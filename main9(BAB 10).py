import numpy as np
import cv2
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

# Subbab 10.1 & 10.2: Parameter Kamera & Sudut Pandang
# Nilai focal length (fx) dan pusat optik (cx) standar Kinect V1
FX = 525.0
CX = 320.0

# Subbab 10.4: Ambang batas jarak aman (Virtual Bumper)
D_SAFE = 800  # 800 mm (80 cm)

print("BAB 10: Mengestimasi Arah dengan Kinect")
print("Arahkan sensor ke dinding/rintangan. Tekan 'q' untuk keluar.")

try:
    while True:
        # Membaca Depth Map
        depth_frame = depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)
        
        # Subbab 10.5: Implementasi Pseudo-LiDAR
        # Ganti noise (nilai 0) dengan 10000 agar tidak terdeteksi sebagai rintangan
        depth_clean = np.where(depth_image == 0, 10000, depth_image)
        
        # Irisan baris tengah (misal di baris piksel 240)
        slice_row = depth_clean[240, :] 
        
        # Subbab 10.4: Segmentasi Arah (Kiri, Tengah, Kanan)
        # Membagi 640 piksel menjadi 3 zona
        left_zone = slice_row[0:213]
        center_zone = slice_row[214:426]
        right_zone = slice_row[427:640]
        
        # Cari jarak terdekat (minimum) di tiap zona
        min_left = np.min(left_zone)
        min_center = np.min(center_zone)
        min_right = np.min(right_zone)
        
        # Subbab 10.6: Integrasi Estimasi Arah ke Kendali (State Machine)
        if min_center < D_SAFE:
            # Rintangan terdeteksi di depan, robot harus berhenti dan mencari arah aman
            status = "STOP! "
            warna_teks = (0, 0, 255) # Merah
            
            if min_left > min_right:
                status += "BELOK KIRI" # Zona kiri lebih lapang
            else:
                status += "BELOK KANAN" # Zona kanan lebih lapang
        else:
            # Jalan di depan aman
            status = "MAJU LURUS"
            warna_teks = (0, 255, 0) # Hijau
            
        # Subbab 10.2: Perhitungan Sudut Horizontal (Azimuth) Rintangan Terdekat
        # Cari indeks piksel dari rintangan terdekat di seluruh baris
        min_idx = np.argmin(slice_row)
        u_obstacle = min_idx
        z_obstacle = slice_row[min_idx]
        
        # Jika rintangan valid (bukan 10000), hitung sudutnya
        if z_obstacle < 10000:
            theta_rad = np.arctan((u_obstacle - CX) / FX)
            theta_deg = np.degrees(theta_rad)
            teks_sudut = f"Sudut Rintangan: {theta_deg:.1f} derajat"
        else:
            teks_sudut = "Sudut Rintangan: N/A"
            
        # Normalisasi untuk visualisasi
        depth_display = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(np.uint8(depth_display), cv2.COLORMAP_JET)
        
        # Gambar garis pembagi zona dan garis irisan Pseudo-LiDAR
        cv2.line(depth_colored, (213, 0), (213, 480), (255, 255, 255), 1)
        cv2.line(depth_colored, (426, 0), (426, 480), (255, 255, 255), 1)
        cv2.line(depth_colored, (0, 240), (640, 240), (0, 0, 0), 2)
        
        # Tandai posisi rintangan terdekat dengan lingkaran
        if z_obstacle < 10000:
            cv2.circle(depth_colored, (u_obstacle, 240), 8, (255, 255, 255), 2)
        
        # Tampilkan teks status arah dan sudut
        cv2.putText(depth_colored, f"Arah: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_teks, 2)
        cv2.putText(depth_colored, teks_sudut, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Tampilkan hasil
        cv2.imshow('Estimasi Arah (Pseudo-LiDAR)', depth_colored)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup saat program ditutup
    print("Mematikan stream...")
    depth_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()