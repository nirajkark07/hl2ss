#------------------------------------------------------------------------------
# This script receives video from the HoloLens front RGB camera and plays it.
# The camera supports various resolutions and framerates. See
# https://github.com/jdibenes/hl2ss/blob/main/etc/pv_configurations.txt
# for a list of supported formats. The default configuration is 1080p 30 FPS. 
# The stream supports three operating modes: 0) video, 1) video + camera pose, 
# 2) query calibration (single transfer).
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import os
import time
import pyrealsense2 as rs
import numpy as np
import threading

# Settings --------------------------------------------------------------------

# HoloLens address
host = "169.254.50.249"

# Operating mode
# 0: video
# 1: video + camera pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Enable Mixed Reality Capture (Holograms)
enable_mrc = False

# Enable Shared Capture
# If another program is already using the PV camera, you can still stream it by
# enabling shared mode, however you cannot change the resolution and framerate
shared = False

# Camera parameters
# Ignored in shared mode
width     = 1920
height    = 1080
framerate = 15

# Framerate denominator (must be > 0)
# Effective FPS is framerate / divisor
divisor = 1 

# Video encoding profile
profile = hl2ss.VideoProfile.H265_MAIN

# Decoded format
# Options include:
# 'bgr24'
# 'rgb24'
# 'bgra'
# 'rgba'
# 'gray8'
decoded_format = 'bgr24'

# Setup directories to store calibration data
#------------------------------------------------------------------------------
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'calibration'))
realsense_dir = os.path.join(script_dir, "calib_data/realsense_calib")
hololens_dir = os.path.join(script_dir, "calib_data/hololens_calib")

# Check if the subfolder exists and if not create it for hololens and realsense cameras
if not os.path.exists(realsense_dir):
    os.makedirs(realsense_dir)
if not os.path.exists(hololens_dir):
    os.makedirs(hololens_dir)

#------------------------------------------------------------------------------

# Setup pipeline for realsense camera
#------------------------------------------------------------------------------
pipe = rs.pipeline()
config = rs.config()
config.enable_device('f1370224')
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipe.start(config)

#------------------------------------------------------------------------------

hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)
# Disable camera autofocus.
focus_value = 2000
client_rc = hl2ss_lnm.ipc_rc(host, hl2ss.IPCPort.REMOTE_CONFIGURATION)
client_rc.open()
client_rc.wait_for_pv_subsystem(True)
client_rc.set_pv_focus(hl2ss.PV_FocusMode.Manual, hl2ss.PV_AutoFocusRange.Normal, hl2ss.PV_ManualFocusDistance.Infinity, focus_value, hl2ss.PV_DriverFallback.Disable)
client_rc.close()

if (mode == hl2ss.StreamMode.MODE_2):
    data = hl2ss_lnm.download_calibration_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width, height, framerate)
    print('Calibration')
    print(f'Focal length: {data.focal_length}')
    print(f'Principal point: {data.principal_point}')
    print(f'Radial distortion: {data.radial_distortion}')
    print(f'Tangential distortion: {data.tangential_distortion}')
    print('Projection')
    print(data.projection)
    print('Intrinsics')
    print(data.intrinsics)
    print('RigNode Extrinsics')
    print(data.extrinsics)

else:
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)
    client.open()
    RecordStream = False
    last_frame_time = time.time() # Initialize frame time

    while (enable):
        data = client.get_next_packet()
        frame = pipe.wait_for_frames()
        color_frame = frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        # print(f'Pose at time {data.timestamp}')
        # print(data.pose)
        # print(f'Focal length: {data.payload.focal_length}')
        # print(f'Principal point: {data.payload.principal_point}')
        
        if data.payload.image is not None:
            cv2.imshow('HoloLens2', data.payload.image)
            cv2.imshow('Realsense', color_image)
            key = cv2.waitKey(1)
        
        # Start saving the frames if space is pressed once until it is pressed again
        if key & 0xFF == ord(" "):
            if not RecordStream:
                time.sleep(0.2)
                RecordStream = True

                # Intrinsic parameters intel realsense
                with open(os.path.join(script_dir, "calib_data/realsense_calib/K_f1370224.txt"), "w") as f:
                    f.write(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
                    f.write(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
                    f.write(f"{0.0} {0.0} {1.0}\n")
                
                with open(os.path.join(script_dir, "calib_data/hololens_calib/K_hl2.txt"), "w") as f:
                    f.write(f"{data.payload.focal_length[0]} {0.0} {data.payload.principal_point[0]}\n")
                    f.write(f"{0.0} {data.payload.focal_length[1]} {data.payload.principal_point[1]}\n")
                    f.write(f"{0.0} {0.0} {1.0}\n")

                print("Recording started")

            else:
                RecordStream = False
                print("Recording stopped")
        if RecordStream:
                current_time = time.time()
                if current_time - last_frame_time >= 5:
                    framename = int(round(time.time() * 1000))
                    hololens_img_path = os.path.join(hololens_dir, f"{framename}.png")
                    realsense_img_path = os.path.join(realsense_dir, f"{framename}.png")
                    cv2.imwrite(hololens_img_path, data.payload.image)
                    cv2.imwrite(realsense_img_path, color_image)
                    last_frame_time = current_time
                    print(f"{framename}.png captured.")

        # press esc or 'q' to close image window
        if key & 0xFF == ord("q") or key == 27: 

            cv2.destroyAllWindows()
            break

    client.close()
    listener.join()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
