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
# 1. INISIALISASI MEDIAPIPE TASKS API
# ==========================================
# Subbab 7.3: Inisialisasi Algoritma Pelacakan Pose
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
model_path = "pose_landmarker_lite.task"

if not os.path.exists(model_path):
    print("Mengunduh model MediaPipe Pose...")
    urllib.request.urlretrieve(MODEL_URL, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# Subbab 7.6: Definisi koneksi tulang untuk Visualisasi
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Wajah
    (9, 10), # Mulut
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Lengan
    (11, 23), (12, 24), (23, 24), # Torso
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), # Kaki Kiri
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)  # Kaki Kanan
]

# ==========================================
# 2. INISIALISASI OPENNI2 & KINECT
# ==========================================
openni2.initialize(r"C:\Program Files\OpenNI2\Redist")
dev = openni2.Device.open_any()

color_stream = dev.create_color_stream()
color_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
    resolutionX=640,
    resolutionY=480,
    fps=30
))
color_stream.start()

print("BAB 7: Mengakses Skeleton Image")
print("Berdirilah di depan Kinect. Tekan 'q' untuk keluar.")

try:
    while True:
        # --- Ambil frame RGB dari Kinect ---
        color_frame = color_stream.read_frame()
        color_data = color_frame.get_buffer_as_uint8()
        color_image = np.frombuffer(color_data, dtype=np.uint8).reshape(480, 640, 3)
        bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # --- Proses Pelacakan Pose ---
        image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Subbab 7.4 & 7.5: State Machine dan Ekstraksi Koordinat
        if results.pose_landmarks:
            for landmark in results.pose_landmarks:
                h, w, _ = bgr_image.shape
                
                # Gambar garis tulang (Subbab 7.6)
                for start, end in connections:
                    p1 = (int(landmark[start].x * w), int(landmark[start].y * h))
                    p2 = (int(landmark[end].x * w), int(landmark[end].y * h))
                    cv2.line(bgr_image, p1, p2, (0, 255, 0), 3)
                
                # Gambar titik sendi utama
                for point in landmark:
                    cx, cy = int(point.x * w), int(point.y * h)
                    cv2.circle(bgr_image, (cx, cy), 4, (0, 0, 255), -1)
                
                # Ekstraksi koordinat Tangan Kanan (Indeks 16) dan Kepala (Indeks 0)
                # landmark[i].visibility adalah representasi "Tracked" vs "Inferred"
                head = landmark[0]
                right_hand = landmark[16]
                
                # Jika tangan kanan terlacak dengan baik (visibility > 0.5)
                if right_hand.visibility > 0.5:
                    hand_x = int(right_hand.x * w)
                    hand_y = int(right_hand.y * h)
                    cv2.circle(bgr_image, (hand_x, hand_y), 8, (255, 0, 0), 2)
                    cv2.putText(bgr_image, f"R.Hand: ({hand_x},{hand_y})", (hand_x + 10, hand_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Tampilkan status pelacakan
                cv2.putText(bgr_image, "Status: SKELETON TRACKED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Subbab 7.4: Jika tidak ada user (Untracked)
            cv2.putText(bgr_image, "Status: WAITING FOR USER...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Skeleton Image Stream', bgr_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Mematikan sistem...")
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()