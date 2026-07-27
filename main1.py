import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from primesense import openni2
from primesense import _openni2 as c_api

# ==========================================
# 1. INISIALISASI MEDIAPIPE POSE
# ==========================================
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
model_path = "pose_landmarker_lite.task"

if not os.path.exists(model_path):
    print("Mengunduh model MediaPipe Pose...")
    urllib.request.urlretrieve(MODEL_URL, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
landmarker = vision.PoseLandmarker.create_from_options(options)

# Koneksi tulang skeleton
connections = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Lengan
    (11, 23), (12, 24), (23, 24), # Torso
    (23, 25), (25, 27), (24, 26), (26, 28) # Kaki
]

# ==========================================
# 2. INISIALISASI OPENNI2 (KODE ANDA)
# ==========================================
openni2.initialize(r"C:\Program Files\OpenNI2\Redist")
dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640, resolutionY=480, fps=30))
depth_stream.start()

color_stream = dev.create_color_stream()
color_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))

ir_stream = dev.create_ir_stream()
ir_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_GRAY16, resolutionX=640, resolutionY=480, fps=30))

cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
cv2.namedWindow('Color / Infrared', cv2.WINDOW_NORMAL)

mode = 'color'
color_stream.start()

print("Tekan 'c' untuk mode Color (Skeleton), 'i' untuk mode Infrared, 'q' untuk keluar")

try:
    while True:
        # --- Depth (selalu aktif) ---
        depth_frame = depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)
        depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = np.uint8(depth_display)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        cv2.imshow('Depth', depth_display)

        # --- Color atau Infrared ---
        if mode == 'color':
            color_frame = color_stream.read_frame()
            color_data = color_frame.get_buffer_as_uint8()
            color_image = np.frombuffer(color_data, dtype=np.uint8).reshape(480, 640, 3)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            # --- PROSES POSE ESTIMATION (MEDIAPIPE) ---
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if results.pose_landmarks:
                for landmark in results.pose_landmarks:
                    h, w, _ = color_image.shape
                    # Gambar garis
                    for start, end in connections:
                        p1 = (int(landmark[start].x * w), int(landmark[start].y * h))
                        p2 = (int(landmark[end].x * w), int(landmark[end].y * h))
                        cv2.line(color_image, p1, p2, (0, 255, 0), 3)
                    # Gambar titik
                    for point in landmark:
                        cx, cy = int(point.x * w), int(point.y * h)
                        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

            cv2.imshow('Color / Infrared', color_image)
        else:
            ir_frame = ir_stream.read_frame()
            ir_data = ir_frame.get_buffer_as_uint16()
            ir_image = np.frombuffer(ir_data, dtype=np.uint16).reshape(480, 640)
            ir_display = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
            ir_display = np.uint8(ir_display)
            cv2.imshow('Color / Infrared', ir_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i') and mode != 'ir':
            color_stream.stop()
            ir_stream.start()
            mode = 'ir'
            print("Mode: Infrared")
        elif key == ord('c') and mode != 'color':
            ir_stream.stop()
            color_stream.start()
            mode = 'color'
            print("Mode: Color + Skeleton")

finally:
    depth_stream.stop()
    if mode == 'color':
        color_stream.stop()
    else:
        ir_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()