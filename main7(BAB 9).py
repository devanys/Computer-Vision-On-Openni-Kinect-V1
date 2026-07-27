import numpy as np
import cv2
from collections import deque
from primesense import openni2
from primesense import _openni2 as c_api

# ==========================================
# INISIALISASI OPENNI2 & KINECT
# ==========================================
openni2.initialize(r"C:\Program Files\OpenNI2\Redist")
dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
depth_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, 
    resolutionX=640, 
    resolutionY=480, 
    fps=30
))
depth_stream.start()

# Subbab 9.4: Variabel untuk Filter Median (Sliding Window)
# Menyimpan 5 frame nilai jarak terakhir untuk merata-ratakan fluktuasi (noise)
distance_history = deque(maxlen=5)

print("BAB 9: Mengestimasi Jarak dengan Kinect")
print("Arahkan tangan/objek ke kotak tengah layar. Tekan 'q' untuk keluar.")

try:
    while True:
        # Subbab 9.2: Membaca Depth Map (u,v,z)
        depth_frame = depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)
        
        # Penanganan noise: Nilai 0 (titik buta) diganti sementara dengan 10000
        depth_clean = np.where(depth_image == 0, 10000, depth_image)
        
        # Subbab 9.3: Mengambil Nilai Kedalaman pada Region of Interest (ROI)
        # Buat kotak ROI berukuran 100x100 piksel tepat di tengah layar
        roi_x1, roi_y1 = 270, 190
        roi_x2, roi_y2 = 370, 290
        roi = depth_clean[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Ambil nilai kedalaman terdekat (minimum) di dalam ROI
        # Ini diasumsikan sebagai objek yang paling menonjol ke arah kamera
        current_dist_mm = np.min(roi)
        
        # Subbab 9.4: Filter Median untuk Stabilisasi Pembacaan Jarak
        distance_history.append(current_dist_mm)
        median_dist_mm = np.median(distance_history)
        
        # Normalisasi untuk visualisasi (16-bit ke 8-bit)
        depth_display = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(np.uint8(depth_display), cv2.COLORMAP_JET)
        
        # Gambar kotak ROI di atas citra kedalaman
        cv2.rectangle(depth_colored, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)
        
        # Subbab 9.1 & 9.6: Evaluasi dan Penampilan Jarak
        # Karena objek berada di pusat optik (u=cx, v=cy), Jarak Euclidean ~= Z
        if median_dist_mm < 10000:
            teks_jarak = f"Jarak (Median): {int(median_dist_mm)} mm ({median_dist_mm/1000:.2f} m)"
            warna_teks = (255, 255, 255)
        else:
            teks_jarak = "Jarak: N/A (Out of Range / Kaca / Material Hitam)"
            warna_teks = (0, 0, 0)
            
        cv2.putText(depth_colored, teks_jarak, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_teks, 2)
        
        # Tampilkan perbandingan nilai mentah (raw) sebelum difilter
        teks_raw = f"Raw Min: {current_dist_mm} mm"
        cv2.putText(depth_colored, teks_raw, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Tampilkan hasil
        cv2.imshow('Estimasi Jarak (ROI & Filter Median)', depth_colored)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup saat program ditutup
    print("Mematikan stream...")
    depth_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()