import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api

# Inisialisasi OpenNI2
openni2.initialize(r"C:\Program Files\OpenNI2\Redist")
dev = openni2.Device.open_any()

# Setup Stream RGB
color_stream = dev.create_color_stream()
color_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))
color_stream.start()

print("BAB 5: Mengakses Data RGB. Tekan 'q' untuk keluar.")

try:
    while True:
        # Baca frame RGB
        color_frame = color_stream.read_frame()
        color_data = color_frame.get_buffer_as_uint8()
        color_image = np.frombuffer(color_data, dtype=np.uint8).reshape(480, 640, 3)
        
        # OpenNI menghasilkan RGB, OpenCV butuh BGR
        bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        # Konversi ruang warna (Subbab 5.5)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Tampilkan
        cv2.imshow('RGB Stream (BGR)', bgr_image)
        cv2.imshow('Grayscale', gray_image)
        cv2.imshow('HSV', hsv_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()