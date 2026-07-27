import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api

# ==========================================
# INISIALISASI OPENNI2 & KINECT
# ==========================================
openni2.initialize(r"C:\Program Files\OpenNI2\Redist")
dev = openni2.Device.open_any()

# Subbab 8.3: Membaca Stream IR menggunakan Driver
# Membuat stream Inframerah
ir_stream = dev.create_ir_stream()

# Subbab 8.2: Format Data Citra IR (16-bit Grayscale)
ir_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_GRAY16, 
    resolutionX=640, 
    resolutionY=480, 
    fps=30
))
ir_stream.start()

print("BAB 8: Mengakses Infrared (IR) Image Stream")
print("Matikan lampu ruangan untuk menguji Night Vision. Tekan 'q' untuk keluar.")

try:
    while True:
        # Membaca buffer IR mentah
        ir_frame = ir_stream.read_frame()
        ir_data = ir_frame.get_buffer_as_uint16()
        
        # Konversi ke Numpy Array 2D (640x480)
        ir_image = np.frombuffer(ir_data, dtype=np.uint16).reshape(480, 640)
        
        # Normalisasi 16-bit ke 8-bit (0-255) agar bisa dirender di layar
        ir_display = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
        ir_display = np.uint8(ir_display)
        
        # Konversi ke BGR agar kita bisa memberi teks berwarna pada citra grayscale
        ir_colored = cv2.cvtColor(ir_display, cv2.COLOR_GRAY2BGR)
        
        # Subbab 8.4: Visualisasi Pola IR (Kinect V1 memancarkan pola titik/titik speckle)
        # Anda akan melihat titik-titik cahaya di layar. Itu adalah pola proyektor IR.
        
        # Subbab 8.5: Aplikasi Citra IR pada Ruang Gelap (Night Vision)
        cv2.putText(ir_colored, "Infrared Stream (Night Vision)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Ambil nilai intensitas di pusat layar untuk analisis sederhana
        center_intensity = ir_image[240, 320]
        cv2.putText(ir_colored, f"Center Intensity: {center_intensity}", (10, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.circle(ir_colored, (320, 240), 3, (0, 0, 255), -1)
        
        # Tampilkan hasil
        cv2.imshow('Infrared Stream', ir_colored)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup saat program ditutup
    print("Mematikan stream...")
    ir_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()