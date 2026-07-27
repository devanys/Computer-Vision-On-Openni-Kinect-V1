import numpy as np
import cv2
from primesense import openni2
from primesense import _openni2 as c_api

# ==========================================
# INISIALISASI OPENNI2 & KINECT
# ==========================================
openni2.initialize(r"C:\Program Files\OpenNI2\Redist")
dev = openni2.Device.open_any()

# Setup Stream Depth
depth_stream = dev.create_depth_stream()

# Subbab 6.1: Format Data Depth (16-bit Unsigned Integer, 1 MM)
depth_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, 
    resolutionX=640, 
    resolutionY=480, 
    fps=30
))
depth_stream.start()

print("BAB 6: Mengakses Depth Image Stream")
print("Tekan 'q' pada jendela gambar untuk keluar.")

try:
    while True:
        # Subbab 6.2: Membaca Depth Map menggunakan OpenCV
        depth_frame = depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        # Mengubah buffer memori menjadi Numpy Array 2D (640x480)
        depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)
        
        # Subbab 6.5: Penanganan "Holes" (Nilai 0) pada Depth Map
        # Nilai 0 (noise/tidak terdeteksi) diganti dengan 10000 agar tidak dibaca sebagai dinding terdekat
        depth_clean = np.where(depth_image == 0, 10000, depth_image)
        
        # Subbab 6.3: Normalisasi dan Visualisasi Depth Map
        # Ubah nilai 16-bit (0-10000 mm) ke 8-bit (0-255) agar bisa ditampilkan di layar monitor
        depth_display = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = np.uint8(depth_display)
        
        # Terapkan colormap agar perbedaan jarak terlihat jelas (Merah=dekat, Biru=jauh)
        depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        
        # Subbab 6.4: Pemetaan Nilai Kedalaman ke Milimeter
        # Ambil nilai jarak mentah di piksel tengah layar (koordinat 320, 240)
        center_y, center_x = 240, 320
        center_dist_mm = depth_clean[center_y, center_x]
        
        # Logika teks untuk ditampilkan di layar
        if center_dist_mm < 10000:
            teks_jarak = f"Jarak Pusat: {center_dist_mm} mm ({center_dist_mm/1000:.2f} m)"
            warna_teks = (255, 255, 255)
        else:
            teks_jarak = "Jarak Pusat: N/A (Out of Range / Noise)"
            warna_teks = (0, 0, 0)
            
        # Gambar titik pusat dan teks informasi jarak
        cv2.circle(depth_colored, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.putText(depth_colored, teks_jarak, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_teks, 2)
        
        # Tampilkan hasil
        cv2.imshow('Depth Map Stream (Colormap)', depth_colored)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup saat program ditutup
    print("Mematikan stream...")
    depth_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()