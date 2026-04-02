#!/usr/bin/env python3

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import pyds
import cv2
import numpy as np
import json
from redis_helper import RedisHelper
from moving_average import MovingAverage

ma = MovingAverage(60)

redis = RedisHelper()


RESOLUTION = (960, 540)  # Set resolution as per your config

display = False

FPS = 59

# Global variables for FPS calculation
frame_count = 0
start_time = time.time()
fps_interval = 1  # Calculate FPS every 30 frames

def calculate_fps():
    """Calculate and return current FPS"""
    global frame_count, start_time
    
    frame_count += 1
    
    if frame_count % fps_interval == 0:
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = fps_interval / elapsed_time
        start_time = current_time
        return fps

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe function to extract bounding box data, draw a line from center to
    the target bbox, and calculate FPS.
    """
    frame_number = 0
    num_rects = 0
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
        
    current_fps = calculate_fps()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
            
        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
            
        # 1. Adım: Tüm sınırlayıcı kutuları bir listede topla
        bboxes = []
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            bboxes.append({
                'class_id': obj_meta.class_id,
                'confidence': obj_meta.confidence,
                'left': obj_meta.rect_params.left,
                'top': obj_meta.rect_params.top,
                'width': obj_meta.rect_params.width,
                'height': obj_meta.rect_params.height
            })
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        # 2. Adım: En yüksek güven skoruna sahip hedef kutusunu bul
        target_bbox = None
        max_conf = 0
        if bboxes:
            for bbox in bboxes:
                print(f"Class={bbox['class_id']}, "
                      f"Confidence={bbox['confidence']:.2f}, "
                      f"BBox=({bbox['left']:.0f},{bbox['top']:.0f},"
                      f"{bbox['width']:.0f},{bbox['height']:.0f})")
                if bbox['confidence'] > max_conf:
                    max_conf = bbox['confidence']
                    target_bbox = bbox # En iyi hedefi sakla
        
        # Redis'i en iyi bulunan kutu ile güncelle
        if target_bbox:
            redis.r.set("bbox_6mm", json.dumps(target_bbox))
        else:
            redis.r.set("bbox_6mm", json.dumps({}))

        # Görüntü metadatasını al
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        
        # Metin bilgilerini yapılandır (Mevcut kodunuz)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # ... (Redis'ten veri alma ve metin oluşturma kodunuz burada yer alıyor) ...
        offset = redis.r.get("offset").decode('utf-8')
        depth_virt = redis.r.get("depth_virt").decode('utf-8')
        drone_mode = redis.r.get("drone_mode").decode('utf-8')
        distance = redis.r.get("distance").decode('utf-8')
        attitude = redis.r.get("attitude").decode('utf-8')
        velocity = redis.r.get("velocity").decode('utf-8')
        interceptor_location = redis.r.get("interceptor_location").decode('utf-8')
        target_location = redis.r.get("target_location").decode('utf-8')
        pixel_error = redis.r.get("pixel_errors").decode('utf-8')
        filter_roll = redis.r.get("filter_roll").decode('utf-8')
        kp_yaw = redis.r.get("kp_yaw").decode('utf-8')
        cbf = redis.r.get("cbf").decode('utf-8')
        roll_rate = redis.r.get("roll_rate").decode('utf-8')
        pitch_rate = redis.r.get("pitch_rate").decode('utf-8')
        yaw_rate = redis.r.get("yaw_rate").decode('utf-8')
        throttle_command = redis.r.get("throttle_command").decode('utf-8')
        redis.r.set("frame_number", frame_number)
        redis.r.set("cam_fps", current_fps)
        fps_vis = ma.update(current_fps)
        display_text2 = f"Drone Mode: {drone_mode}\nDistance: {distance} m\nAttitude: {attitude}\nVelocity: {velocity} m/s\nInterceptor Location: {interceptor_location}\nTarget Location: {target_location}\nPixel Error: {pixel_error}\nFilter Roll: {filter_roll}\nKp Yaw: {kp_yaw}\nCBF: {cbf}\nRoll Rate: {roll_rate} rad/s\nPitch Rate: {pitch_rate} rad/s\nYaw Rate: {yaw_rate} rad/s\nThrottle Command: {throttle_command}\nOffset: {offset}, Depth: {depth_virt}"
        fps_text = f" FPS: {fps_vis:.1f}" if fps_vis is not None else ""
        display_text = f"Frame Number={frame_number} Number of Objects={num_rects}{fps_text} CAM: 6mm\n{display_text2}"
        py_nvosd_text_params.display_text = display_text
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 3 if RESOLUTION[0] <= 1280 else 20
        py_nvosd_text_params.font_params.font_color.set(1.0, 0.0, 0.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.1)

        # 3. Adım: Eğer bir hedef bulunduysa, çizgiyi çiz
        # if target_bbox:
        # Çizilecek çizgi sayısını 1 olarak ayarla
        display_meta.num_lines = 1
        line_params = display_meta.line_params[0]
        
        # Çizginin başlangıç noktası: ekranın merkezi
        screen_center_x = RESOLUTION[0] // 2
        screen_center_y = RESOLUTION[1] // 2
        
        # Çizginin bitiş noktası: hedef kutusunun merkezi
        virt_x = float(redis.r.get("pixel_x_virt").decode('utf-8'))
        virt_y = float(redis.r.get("pixel_y_virt").decode('utf-8'))
        bbox_center_x = int(virt_x)
        bbox_center_y = int(virt_y)
        
        # Çizgi parametrelerini ayarla
        line_params.x1 = screen_center_x
        line_params.y1 = screen_center_y
        line_params.x2 = bbox_center_x
        line_params.y2 = bbox_center_y
        line_params.line_width = 2
        line_params.line_color.set(1.0, 1.0, 0.0, 0.8)  # Sarı, hafif şeffaf

        # Metadatayı (metin ve çizgi) kareye ekle
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    return Gst.PadProbeReturn.OK

def create_pipeline():
    """
    Create a DeepStream pipeline similar to your deepstream-app configuration
    """
    # Initialize GStreamer
    Gst.init(None)
    
    # Create pipeline
    pipeline = Gst.Pipeline()
    
    # --- Original Elements ---
    source = Gst.ElementFactory.make("nvarguscamerasrc", "camera-source")
    caps_filter = Gst.ElementFactory.make("capsfilter", "caps-filter")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    
    # Tee for splitting after OSD (for display and HUD recording)
    tee = Gst.ElementFactory.make("tee", "tee-hud")
    queue1 = Gst.ElementFactory.make("queue", "queue-display")
    queue2 = Gst.ElementFactory.make("queue", "queue-hud-record")
    
    # Display sink
    if display:
        print("Creating nv3dsink \n")
        display_sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")

    # HUD video sink elements
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder-hud")
    parser = Gst.ElementFactory.make("h264parse", "parser-hud")
    container = Gst.ElementFactory.make("mp4mux", "container-hud")
    file_sink = Gst.ElementFactory.make("filesink", "file-sink-hud")

    # --- YENİ: Ham Video Kaydı İçin Elemanlar ---
    tee_raw_split = Gst.ElementFactory.make("tee", "tee-raw-split")
    queue_raw = Gst.ElementFactory.make("queue", "queue-raw-record")
    encoder_raw = Gst.ElementFactory.make("nvv4l2h264enc", "encoder-raw")
    parser_raw = Gst.ElementFactory.make("h264parse", "parser-raw")
    container_raw = Gst.ElementFactory.make("mp4mux", "container-raw")
    file_sink_raw = Gst.ElementFactory.make("filesink", "file-sink-raw")

    if not all([source, streammux, pgie, nvvidconv, nvosd, tee]):
        sys.stderr.write("Unable to create essential elements\n")
        sys.exit(1)
    
    if not all([queue1, queue2, encoder, parser, container, file_sink]):
        sys.stderr.write("Unable to create HUD recording elements\n")
        sys.exit(1)
    
    # --- YENİ: Ham Kayıt Elemanları için Kontrol ---
    if not all([tee_raw_split, queue_raw, encoder_raw, parser_raw, container_raw, file_sink_raw]):
        sys.stderr.write("Unable to create Raw recording elements\n")
        sys.exit(1)

    # --- Özellikleri Ayarlama ---
    source.set_property("sensor-id", 1)
    source.set_property("tnr-mode", 0)
    source.set_property("tnr-strength", 0)
    source.set_property("ee-mode", 0)
    source.set_property("ee-strength", 0)
    caps = Gst.Caps.from_string(f"video/x-raw(memory:NVMM), width={RESOLUTION[0]}, height={RESOLUTION[1]}, format=NV12, framerate={FPS}/1")
    caps_filter.set_property("caps", caps)
    nvvidconv.set_property("flip-method", 2)
    
    streammux.set_property("width", RESOLUTION[0])
    streammux.set_property("height", RESOLUTION[1])
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)
    
    pgie.set_property("config-file-path", "/home/ituarc/DeepStream-Yolo/config_infer_primary_yolo11.txt")
    
    if display:
        display_sink.set_property("sync", False)
    
    queue1.set_property("max-size-buffers", 1)
    queue1.set_property("leaky", 2)
    queue2.set_property("max-size-buffers", 10)
    queue2.set_property("max-size-time", 0)
    queue2.set_property("max-size-bytes", 0)
    
    # --- YENİ: Ham Kayıt Kuyruğu Özellikleri ---
    queue_raw.set_property("max-size-buffers", 10)
    
    # HUD video kayıt özellikleri
    encoder.set_property("bitrate", 4000000)
    
    # --- YENİ: Ham Video Kodlayıcı Özellikleri ---
    encoder_raw.set_property("bitrate", 4000000)
    
    log_number = redis.r.get("log_number").decode('utf-8')
    test_number = redis.r.get("test_number").decode('utf-8')
    # DEĞİŞTİRİLDİ: Dosya adları güncellendi
    file_sink.set_property("location", f"/home/ituarc/Documents/Visual-Guidance/logs/visual/output_6mm_{log_number}_{test_number}.mp4")
    file_sink_raw.set_property("location", f"/home/ituarc/Documents/Visual-Guidance/logs/visual/output_6mm_{log_number}_{test_number}_raw.mp4")
    
    # --- Boru Hattına Eleman Ekleme ---
    pipeline.add(source)
    pipeline.add(caps_filter)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    # YENİ: Ham akışı ayırmak için tee eklendi
    pipeline.add(tee_raw_split)
    pipeline.add(nvosd)
    pipeline.add(tee)
    pipeline.add(queue1)
    pipeline.add(queue2)
    if display:
        pipeline.add(display_sink)
    pipeline.add(encoder)
    pipeline.add(parser)
    pipeline.add(container)
    pipeline.add(file_sink)
    
    # YENİ: Ham kayıt elemanları eklendi
    pipeline.add(queue_raw)
    pipeline.add(encoder_raw)
    pipeline.add(parser_raw)
    pipeline.add(container_raw)
    pipeline.add(file_sink_raw)

    # --- Elemanları Birbirine Bağlama ---
    source.link(caps_filter)
    
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_filter.get_static_pad("src")
    srcpad.link(sinkpad)
    
    streammux.link(pgie)
    pgie.link(nvvidconv)
    
    # DEĞİŞTİRİLDİ: Akışı OSD'den önce ayır
    nvvidconv.link(tee_raw_split)

    # YENİ: Ham Kayıt Dalı
    tee_raw_split.link(queue_raw)
    queue_raw.link(encoder_raw)
    encoder_raw.link(parser_raw)
    parser_raw.link(container_raw)
    container_raw.link(file_sink_raw)
    
    # DEĞİŞTİRİLDİ: HUD Dalı tee'den besleniyor
    tee_raw_split.link(nvosd)
    nvosd.link(tee)
    
    # Görüntüleme dalı
    tee.link(queue1)
    if display:
        queue1.link(display_sink)
    
    # HUD'lu kayıt dalı
    tee.link(queue2)
    queue2.link(encoder)
    encoder.link(parser)
    parser.link(container)
    container.link(file_sink)
    
    # Sınırlayıcı kutuları çıkarmak için prob ekle
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    return pipeline

def main():
    # Initialize global timing variables
    global start_time
    start_time = time.time()
    
    # Create and start pipeline
    pipeline = create_pipeline()
    
    # Create loop
    loop = GObject.MainLoop()
    
    # Add bus message handler for proper cleanup
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    
    def on_message(bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, Debug: {debug}")
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
            
        elif redis.r.get("stop").decode('utf-8') == "True":
            print("Stopping pipeline as per Redis command")
            loop.quit()
    
    bus.connect("message", on_message)
    
    # Start pipeline
    print("Starting pipeline...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Failed to start pipeline")
        return
    
    if redis.r.get("stop").decode('utf-8') == "True":
        print("Stopping pipeline as per Redis command")
        loop.quit()
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        # Send EOS event to properly close MP4 file
        print("Sending EOS event to the pipeline...")
        pipeline.send_event(Gst.Event.new_eos())
        # Wait a bit for EOS to be processed
        time.sleep(2)
        # Print final statistics
        total_time = time.time() - start_time
        if total_time > 0:
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {frame_count / total_time:.2f}")
        redis.r.set("stop", "False")
        
        # Cleanup
        print("Setting pipeline to NULL state...")
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped.")


if __name__ == '__main__':
    main()