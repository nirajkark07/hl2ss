import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
config = rs.config()
config.enable_device('f1370224')
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipe.start(config)
