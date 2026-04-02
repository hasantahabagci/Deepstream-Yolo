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

redis = RedisHelper()


RESOLUTION = (960, 540)  # Set resolution as per your config

display = False  # Set to False if you don't want to display the video stream

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
    return None

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe function to extract bounding box data from the pipeline and calculate FPS
    """
    frame_number = 0
    num_rects = 0
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Calculate FPS
    current_fps = calculate_fps()
    # if current_fps is not None:
    #     print(f"Current FPS: {current_fps:.2f}")

    # Retrieve batch metadata from the gst_buffer
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
            
        # Extract bounding boxes
        bboxes = []
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            # Extract bounding box coordinates
            bbox_info = {
                'class_id': obj_meta.class_id,
                'confidence': obj_meta.confidence,
                'left': obj_meta.rect_params.left,
                'top': obj_meta.rect_params.top,
                'width': obj_meta.rect_params.width,
                'height': obj_meta.rect_params.height,
                'right': obj_meta.rect_params.left + obj_meta.rect_params.width,
                'bottom': obj_meta.rect_params.top + obj_meta.rect_params.height
            }
            bboxes.append(bbox_info)
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
            
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        
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
        
        # Prepare the display text
        display_text2 = f"Drone Mode: {drone_mode}\nDistance: {distance} m\nAttitude: {attitude}\nVelocity: {velocity} m/s\nInterceptor Location: {interceptor_location}\nTarget Location: {target_location}\nPixel Error: {pixel_error}\nFilter Roll: {filter_roll}\nKp Yaw: {kp_yaw}\nCBF: {cbf}\nRoll Rate: {roll_rate} rad/s\nPitch Rate: {pitch_rate} rad/s\nYaw Rate: {yaw_rate} rad/s\nThrottle Command: {throttle_command}\n"
        
        # Include FPS in the display text
        fps_text = f" FPS: {current_fps:.1f}" if current_fps is not None else ""
        display_text = f"Frame Number={frame_number} Number of Objects={num_rects}{fps_text}\n{display_text2}"
        
        
        # Setting display text to be shown on screen
        py_nvosd_text_params.display_text = display_text

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 3 if RESOLUTION[0] <= 1280 else 20
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 0.0, 0.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.1)
        # Using pyds.get_string() to get display_text as string
        #print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # Print or process bounding boxes
        if bboxes:
            #print(f"Frame {frame_number}: Found {len(bboxes)} objects")
            max_conf = 0
            for i, bbox in enumerate(bboxes):
                print(f"  Object {i}: Class={bbox['class_id']}, "
                      f"Confidence={bbox['confidence']:.2f}, "
                      f"BBox=({bbox['left']:.0f},{bbox['top']:.0f},"
                      f"{bbox['width']:.0f},{bbox['height']:.0f})")
                if bbox['confidence'] > max_conf:
                    max_conf = bbox['confidence']
                    redis.r.set("bbox", json.dumps(bbox))
                    print("Selected:", i)
        else:
            redis.r.set("bbox", json.dumps({}))
            
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    return Gst.PadProbeReturn.OK

def create_pipeline():
    """
    Create a DeepStream pipeline with dual recording: raw video and annotated video
    """
    # Initialize GStreamer
    Gst.init(None)
    
    # Create pipeline
    pipeline = Gst.Pipeline()
    
    # Create elements based on your config
    source = Gst.ElementFactory.make("nvarguscamerasrc", "camera-source")
    caps_filter = Gst.ElementFactory.make("capsfilter", "caps-filter")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    
    # Create tee after nvvidconv (before OSD) to split the stream
    tee_raw = Gst.ElementFactory.make("tee", "tee-raw")
    
    # Create another tee after OSD for annotated video
    tee_annotated = Gst.ElementFactory.make("tee", "tee-annotated")
    
    # Create queues for different branches
    queue_raw = Gst.ElementFactory.make("queue", "queue-raw")
    queue_display = Gst.ElementFactory.make("queue", "queue-display")
    queue_annotated = Gst.ElementFactory.make("queue", "queue-annotated")
    queue_osd = Gst.ElementFactory.make("queue", "queue-osd")
    
    # Display sink
    print("Creating nv3dsink \n")
    if display:
        display_sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")

    # Annotated video recording elements
    encoder_annotated = Gst.ElementFactory.make("nvv4l2h264enc", "encoder-annotated")
    parser_annotated = Gst.ElementFactory.make("h264parse", "parser-annotated")
    container_annotated = Gst.ElementFactory.make("mp4mux", "container-annotated")
    file_sink_annotated = Gst.ElementFactory.make("filesink", "file-sink-annotated")
    
    # Raw video recording elements
    encoder_raw = Gst.ElementFactory.make("nvv4l2h264enc", "encoder-raw")
    parser_raw = Gst.ElementFactory.make("h264parse", "parser-raw")
    container_raw = Gst.ElementFactory.make("mp4mux", "container-raw")
    file_sink_raw = Gst.ElementFactory.make("filesink", "file-sink-raw")

    # Check if all elements were created successfully
    if not source or not streammux or not pgie or not nvvidconv or not nvosd:
        sys.stderr.write("Unable to create primary elements\n")
        sys.exit(1)
    
    if not tee_raw or not tee_annotated or not queue_raw or not queue_display or not queue_annotated or not queue_osd:
        sys.stderr.write("Unable to create tee and queue elements\n")
        sys.exit(1)
        
    if not encoder_annotated or not parser_annotated or not container_annotated or not file_sink_annotated:
        sys.stderr.write("Unable to create annotated recording elements\n")
        sys.exit(1)
        
    if not encoder_raw or not parser_raw or not container_raw or not file_sink_raw:
        sys.stderr.write("Unable to create raw recording elements\n")
        sys.exit(1)

    # Set properties based on your config
    source.set_property("sensor-id", 0)
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
    
    pgie.set_property("config-file-path", "config_infer_primary_yolo11.txt")
    
    if display:
        display_sink.set_property("sync", False)
    
    # Set queue properties to prevent blocking
    queue_raw.set_property("max-size-buffers", 10)
    queue_raw.set_property("max-size-time", 0)
    queue_raw.set_property("max-size-bytes", 0)
    
    queue_display.set_property("max-size-buffers", 1)
    queue_display.set_property("leaky", 2)  # Leak downstream
    
    queue_annotated.set_property("max-size-buffers", 10)
    queue_annotated.set_property("max-size-time", 0)
    queue_annotated.set_property("max-size-bytes", 0)
    
    queue_osd.set_property("max-size-buffers", 1)
    queue_osd.set_property("leaky", 2)  # Leak downstream
    
    # Set encoder properties
    encoder_annotated.set_property("bitrate", 4000000)
    encoder_raw.set_property("bitrate", 4000000)
    
    # Set file sink properties
    log_number = redis.r.get("log_number").decode('utf-8')
    file_sink_annotated.set_property("location", f"/home/ituarc/Documents/Visual-Guidance/logs/visual/output_{log_number}.mp4")
    file_sink_raw.set_property("location", f"/home/ituarc/Documents/Visual-Guidance/logs/visual/output_raw_{log_number}.mp4")
    
    # Add elements to pipeline
    pipeline.add(source)
    pipeline.add(caps_filter)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(tee_raw)
    pipeline.add(queue_raw)
    pipeline.add(queue_osd)
    pipeline.add(nvosd)
    pipeline.add(tee_annotated)
    pipeline.add(queue_display)
    pipeline.add(queue_annotated)
    if display:
        pipeline.add(display_sink)
    
    # Add annotated recording elements
    pipeline.add(encoder_annotated)
    pipeline.add(parser_annotated)
    pipeline.add(container_annotated)
    pipeline.add(file_sink_annotated)
    
    # Add raw recording elements
    pipeline.add(encoder_raw)
    pipeline.add(parser_raw)
    pipeline.add(container_raw)
    pipeline.add(file_sink_raw)
    
    # Link main pipeline up to the first tee
    source.link(caps_filter)
    
    # Create sink pad for streammux
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_filter.get_static_pad("src")
    srcpad.link(sinkpad)
    
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(tee_raw)
    
    # Link raw video recording branch (before OSD)
    tee_raw.link(queue_raw)
    queue_raw.link(encoder_raw)
    encoder_raw.link(parser_raw)
    parser_raw.link(container_raw)
    container_raw.link(file_sink_raw)
    
    # Link to OSD branch
    tee_raw.link(queue_osd)
    queue_osd.link(nvosd)
    nvosd.link(tee_annotated)
    
    # Link display branch
    tee_annotated.link(queue_display)
    if display:
        queue_display.link(display_sink)
    
    # Link annotated video recording branch
    tee_annotated.link(queue_annotated)
    queue_annotated.link(encoder_annotated)
    encoder_annotated.link(parser_annotated)
    parser_annotated.link(container_annotated)
    container_annotated.link(file_sink_annotated)
    
    # Add probe to extract bounding boxes
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
        # Send EOS event to properly close MP4 files
        pipeline.send_event(Gst.Event.new_eos())
        # Wait a bit for EOS to be processed
        time.sleep(1)
        # Print final statistics
        total_time = time.time() - start_time
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count / total_time:.2f}")
        redis.r.set("stop", "False")
    
    # Cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()