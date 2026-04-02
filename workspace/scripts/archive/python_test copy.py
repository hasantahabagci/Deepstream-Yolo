#!/usr/bin/env python3
import sys
import time
import gi
gi.require_version('Gst', '1.0')
# GObject yerine GLib kullanılıyor
from gi.repository import GObject, Gst, GLib
import pyds
import cv2
import numpy as np
import json
from redis_helper import RedisHelper

# --- Yapılandırma ---
redis = RedisHelper()
RESOLUTION = (960, 540)
display = False
FPS = 59
current_active_camera = 0

# FPS hesaplaması için global değişkenler
frame_count = 0
start_time = time.time()
fps_interval = 1

def calculate_fps():
    """Mevcut FPS'yi hesaplar ve döndürür"""
    global frame_count, start_time
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time >= fps_interval:
        fps = frame_count / elapsed_time
        start_time = current_time
        frame_count = 0
        return fps
    return None

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Sınırlayıcı kutu verilerini çıkarmak, merkezden hedefe bir çizgi çizmek
    ve FPS'yi hesaplamak için prob fonksiyonu.
    """
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("GstBuffer alınamadı")
        return Gst.PadProbeReturn.OK

    current_fps_val = calculate_fps()
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
        bboxes = []
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            bboxes.append({
                'class_id': obj_meta.class_id, 'confidence': obj_meta.confidence,
                'left': obj_meta.rect_params.left, 'top': obj_meta.rect_params.top,
                'width': obj_meta.rect_params.width, 'height': obj_meta.rect_params.height
            })
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        target_bbox = None
        max_conf = 0
        if bboxes:
            for bbox in bboxes:
                if bbox['confidence'] > max_conf:
                    max_conf = bbox['confidence']
                    target_bbox = bbox
        
        if target_bbox:
            redis.r.set("bbox", json.dumps(target_bbox))
        else:
            redis.r.set("bbox", json.dumps({}))

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        
        # Redis'ten verileri al
        try:
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
            virt_x = float(redis.r.get("pixel_x_virt").decode('utf-8'))
            virt_y = float(redis.r.get("pixel_y_virt").decode('utf-8'))
        except redis.exceptions.RedisError as e:
            print(f"OSD probunda Redis verisi okunurken hata: {e}")
            offset, depth_virt, drone_mode, distance, attitude, velocity, interceptor_location, target_location, pixel_error, filter_roll, kp_yaw, cbf, roll_rate, pitch_rate, yaw_rate, throttle_command, virt_x, virt_y = ["N/A"] * 16 + [0.0, 0.0]

        redis.r.set("frame_number", frame_number)
        if current_fps_val is not None:
             redis.r.set("cam_fps", f"{current_fps_val:.1f}")

        display_text2 = f"Drone Mode: {drone_mode}\nDistance: {distance} m\nAttitude: {attitude}\nVelocity: {velocity} m/s\nInterceptor Location: {interceptor_location}\nTarget Location: {target_location}\nPixel Error: {pixel_error}\nFilter Roll: {filter_roll}\nKp Yaw: {kp_yaw}\nCBF: {cbf}\nRoll Rate: {roll_rate} rad/s\nPitch Rate: {pitch_rate} rad/s\nYaw Rate: {yaw_rate} rad/s\nThrottle Command: {throttle_command}\nOffset: {offset}, Depth: {depth_virt}"
        fps_text = f" FPS: {float(redis.r.get('cam_fps').decode('utf-8')):.1f}" if redis.r.exists('cam_fps') else ""
        display_text = f"Frame Number={frame_number} Number of Objects={num_rects}{fps_text}\n{display_text2}"
        
        py_nvosd_text_params.display_text = display_text
        py_nvosd_text_params.x_offset, py_nvosd_text_params.y_offset = 10, 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 12
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.4)

        display_meta.num_lines = 1
        line_params = display_meta.line_params[0]
        line_params.x1, line_params.y1 = RESOLUTION[0] // 2, RESOLUTION[1] // 2
        line_params.x2, line_params.y2 = int(virt_x), int(virt_y)
        line_params.line_width = 2
        line_params.line_color.set(1.0, 1.0, 0.0, 0.8)
        
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK

def create_pipeline():
    """İki kamera kaynağı ve dinamik durum yönetimi ile boru hattı oluşturur."""
    Gst.init(None)
    pipeline = Gst.Pipeline()
    
    # --- Elemanlar ---
    source0 = Gst.ElementFactory.make("nvarguscamerasrc", "camera-source-0")
    source1 = Gst.ElementFactory.make("nvarguscamerasrc", "camera-source-1")
    caps_filter0 = Gst.ElementFactory.make("capsfilter", "caps-filter-0")
    caps_filter1 = Gst.ElementFactory.make("capsfilter", "caps-filter-1")
    input_selector = Gst.ElementFactory.make("input-selector", "input-selector")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    tee = Gst.ElementFactory.make("tee", "tee-hud")
    queue1, queue2 = Gst.ElementFactory.make("queue", "queue-display"), Gst.ElementFactory.make("queue", "queue-hud-record")
    encoder, parser, container, file_sink = [Gst.ElementFactory.make(factory, name) for factory, name in [
        ("nvv4l2h264enc", "encoder-hud"), ("h264parse", "parser-hud"),
        ("mp4mux", "container-hud"), ("filesink", "file-sink-hud")]]
    tee_raw_split = Gst.ElementFactory.make("tee", "tee-raw-split")
    queue_raw, encoder_raw, parser_raw, container_raw, file_sink_raw = [Gst.ElementFactory.make(factory, name) for factory, name in [
        ("queue", "queue-raw-record"), ("nvv4l2h264enc", "encoder-raw"),
        ("h264parse", "parser-raw"), ("mp4mux", "container-raw"), ("filesink", "file-sink-raw")]]
    
    elements = [source0, source1, caps_filter0, caps_filter1, input_selector, streammux, pgie, nvvidconv, nvosd, tee, queue1, queue2, encoder, parser, container, file_sink, tee_raw_split, queue_raw, encoder_raw, parser_raw, container_raw, file_sink_raw]
    if display:
        display_sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        elements.append(display_sink)

    if not all(elements):
        sys.stderr.write("Temel elemanlar oluşturulamadı\n")
        sys.exit(1)

    # --- Özellikleri Ayarlama ---
    source0.set_property("sensor-id", 0)
    source1.set_property("sensor-id", 1)
    caps_str = f"video/x-raw(memory:NVMM), width={RESOLUTION[0]}, height={RESOLUTION[1]}, format=NV12, framerate={FPS}/1"
    caps_filter0.set_property("caps", Gst.Caps.from_string(caps_str))
    caps_filter1.set_property("caps", Gst.Caps.from_string(caps_str))
    
    streammux.set_property("width", RESOLUTION[0])
    streammux.set_property("height", RESOLUTION[1])
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)
    
    pgie.set_property("config-file-path", "config_infer_primary_yolo11.txt")
    log_number = redis.r.get("log_number").decode('utf-8')
    file_sink.set_property("location", f"/home/ituarc/Documents/Visual-Guidance/logs/visual/output_{log_number}.mp4")
    file_sink_raw.set_property("location", f"/home/ituarc/Documents/Visual-Guidance/logs/visual/output_{log_number}_raw.mp4")

    # --- Boru Hattına Eleman Ekleme ---
    for el in elements: pipeline.add(el)

    # --- DÜZELTME: Elemanları Açık ve Net Bir Şekilde Bağlama ---
    # Kamera 0 -> caps_filter0 -> input_selector (sink_0)
    source0.link(caps_filter0)
    sinkpad0 = input_selector.get_request_pad("sink_%u")
    srcpad0 = caps_filter0.get_static_pad("src")
    srcpad0.link(sinkpad0)

    # Kamera 1 -> caps_filter1 -> input_selector (sink_1)
    source1.link(caps_filter1)
    sinkpad1 = input_selector.get_request_pad("sink_%u")
    srcpad1 = caps_filter1.get_static_pad("src")
    srcpad1.link(sinkpad1)
    
    # Boru hattının geri kalanı
    input_selector.link(streammux)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(tee_raw_split)
    
    tee_raw_split.link(queue_raw)
    queue_raw.link(encoder_raw)
    encoder_raw.link(parser_raw)
    parser_raw.link(container_raw)
    container_raw.link(file_sink_raw)
    
    tee_raw_split.link(nvosd)
    nvosd.link(tee)
    
    tee.link(queue2)
    queue2.link(encoder)
    encoder.link(parser)
    parser.link(container)
    container.link(file_sink)
    
    if display:
        tee.link(queue1)
        queue1.link(display_sink)
    
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # DÜZELTME: Geçiş fonksiyonu için pad'leri de döndür
    return pipeline, input_selector, source0, source1, sinkpad0, sinkpad1

def check_camera_switch(data):
    """Redis'i kontrol eder ve kamera kaynaklarının durumunu yönetir."""
    global current_active_camera
    # DÜZELTME: Fonksiyona gelen veriyi güncelle
    input_selector, source0, source1, sinkpad0, sinkpad1 = data
    try:
        desired_camera_str = redis.r.get("active_camera")
        if desired_camera_str:
            desired_camera = int(desired_camera_str.decode('utf-8'))
            if desired_camera != current_active_camera and desired_camera in [0, 1]:
                print(f"--- Kamera Geçiş İsteği: {current_active_camera} -> {desired_camera} ---")
                
                active_source = source0 if desired_camera == 0 else source1
                inactive_source = source1 if desired_camera == 0 else source0
                # DÜZELTME: Aktif pedi doğrudan kullan
                new_pad = sinkpad0 if desired_camera == 0 else sinkpad1
                
                inactive_source.set_state(Gst.State.READY)
                active_source.set_state(Gst.State.PLAYING)
                
                input_selector.set_property("active-pad", new_pad)
                
                print(f"--- Kamera Başarıyla Değiştirildi: {desired_camera} ---")
                current_active_camera = desired_camera

    except (ValueError, redis.exceptions.RedisError, AttributeError) as e:
        print(f"Kamera geçişi kontrol edilirken hata: {e}")
    
    return True

def main():
    global start_time
    start_time = time.time()
    
    try:
        redis.r.set("active_camera", "0")
        redis.r.set("stop", "False")
    except redis.exceptions.RedisError as e:
        sys.stderr.write(f"Redis başlangıç ayarları yapılamadı: {e}\n")
        sys.exit(1)

    # DÜZELTME: Yeni dönüş değerlerini al
    pipeline, input_selector, source0, source1, sinkpad0, sinkpad1 = create_pipeline()
    
    # DÜZELTME: GObject.MainLoop -> GLib.MainLoop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    
    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Hata: {err}, Hata ayıklama: {debug}")
            loop.quit()
        elif t == Gst.MessageType.EOS:
            print("Akış sonu")
            loop.quit()
        try:
            if redis.r.get("stop").decode('utf-8') == "True":
                print("Redis komutuyla boru hattı durduruluyor")
                loop.quit()
        except redis.exceptions.RedisError: pass

    bus.connect("message", on_message)
    
    # DÜZELTME: GObject.timeout_add -> GLib.timeout_add ve yeni veri demetini gönder
    GLib.timeout_add(250, check_camera_switch, (input_selector, source0, source1, sinkpad0, sinkpad1))

    print("Boru hattı başlatılıyor...")
    source1.set_state(Gst.State.READY)
    ret = pipeline.set_state(Gst.State.PLAYING)
    
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Boru hattı başlatılamadı")
        return
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nBoru hattı durduruluyor...")
    finally:
        print("Boru hattına EOS olayı gönderiliyor...")
        pipeline.send_event(Gst.Event.new_eos())
        time.sleep(2)
        print("Boru hattı NULL durumuna ayarlanıyor...")
        pipeline.set_state(Gst.State.NULL)
        print("Boru hattı durduruldu.")
        try:
            redis.r.set("stop", "False")
        except redis.exceptions.RedisError: pass

if __name__ == '__main__':
    main()
