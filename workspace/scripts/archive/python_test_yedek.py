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
    Create a DeepStream pipeline similar to your deepstream-app configuration
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
    
    # Add tee and queue elements for branching
    tee = Gst.ElementFactory.make("tee", "tee")
    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    
    # Display sink
    print("Creating nv3dsink \n")
    if display:
        display_sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")

    # MP4 file sink elements (following your gst-launch pipeline)
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    parser = Gst.ElementFactory.make("h264parse", "parser")
    container = Gst.ElementFactory.make("mp4mux", "container")
    file_sink = Gst.ElementFactory.make("filesink", "file-sink")

    if not source or not streammux or not pgie or not nvvidconv or not nvosd :
        sys.stderr.write("Unable to create elements\n")
        sys.exit(1)
    
    if not tee or not queue1 or not queue2 or not encoder or not parser or not container or not file_sink:
        sys.stderr.write("Unable to create MP4 recording elements\n")
        sys.exit(1)

    # Set properties based on your config
    source.set_property("sensor-id", 0)
    source.set_property("tnr-mode", 0)  # Set temporal noise reduction mode if needed
    source.set_property("tnr-strength", 0)  # Set temporal noise reduction strength if needed
    source.set_property("ee-mode", 0)  # Set edge enhancement mode if needed
    source.set_property("ee-strength", 0)  # Set edge enhancement strength if needed
    caps = Gst.Caps.from_string(f"video/x-raw(memory:NVMM), width={RESOLUTION[0]}, height={RESOLUTION[1]}, format=NV12, framerate={FPS}/1")
    caps_filter.set_property("caps", caps)
    nvvidconv.set_property("flip-method", 2)  # Set flip method if needed
    
    streammux.set_property("width", RESOLUTION[0])
    streammux.set_property("height", RESOLUTION[1])
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)
    
    pgie.set_property("config-file-path", "config_infer_primary_yolo11.txt")
    
    if display:
        display_sink.set_property("sync", False)
    
    # Set queue properties to prevent blocking
    queue1.set_property("max-size-buffers", 1)
    queue1.set_property("leaky", 2)  # Leak downstream
    queue2.set_property("max-size-buffers", 10)
    queue2.set_property("max-size-time", 0)
    queue2.set_property("max-size-bytes", 0)
    
    # Set MP4 recording properties (following your gst-launch pipeline)
    encoder.set_property("bitrate", 4000000)
    
    log_number = redis.r.get("log_number").decode('utf-8')
    file_sink.set_property("location", f"/home/ituarc/Documents/Visual-Guidance/logs/visual/output_{log_number}.mp4")
    
    # Add elements to pipeline
    pipeline.add(source)
    pipeline.add(caps_filter)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
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
    
    # Link elements
    source.link(caps_filter)
    
    # Create sink pad for streammux
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_filter.get_static_pad("src")
    srcpad.link(sinkpad)
    
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(tee)
    
    # Link display branch
    tee.link(queue1)
    if display:
        queue1.link(display_sink)
    
    # Link recording branch (following your gst-launch pipeline structure)
    tee.link(queue2)
    queue2.link(encoder)
    encoder.link(parser)
    parser.link(container)
    container.link(file_sink)
    
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
        # Send EOS event to properly close MP4 file
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