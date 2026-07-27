import numpy as np
import cv2
import open3d as o3d
from primesense import openni2
from primesense import _openni2 as c_api

# ==========================================
# 1. PARAMETER INTRINSIK KAMERA KINECT V1
# Subbab 12.2: Persamaan Transformasi Koordinat
# ==========================================
FX = 525.0  # Focal length X (piksel)
FY = 525.0  # Focal length Y (piksel)
CX = 319.5  # Pusat optik X (piksel)
CY = 239.5  # Pusat optik Y (piksel)

# ==========================================
# 2. INISIALISASI OPENNI2 & KINECT
# ==========================================
openni2.initialize(r"C:\Program Files\OpenNI2\Redist")
dev = openni2.Device.open_any()

# Setup Stream Depth
depth_stream = dev.create_depth_stream()
depth_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640, resolutionY=480, fps=30))
depth_stream.start()

# Setup Stream RGB
color_stream = dev.create_color_stream()
color_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))
color_stream.start()

# Subbab 12.4: Sinkronisasi Depth ke RGB (Agar warna pas dengan bentuk 3D)
dev.set_image_registration_mode(1)

print("BAB 12: Membangun Colored Point Cloud 3D")
print("Tunggu sebentar, jendela Open3D akan terbuka. Tekan 'q' pada jendela Open3D atau OpenCV untuk keluar.")

# ==========================================
# 3. SETUP VISUALISASI OPEN3D
# Subbab 12.6: Visualisasi 3D Point Cloud
# ==========================================
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Colored Point Cloud 3D", width=800, height=600)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# Atur sudut pandang kamera virtual
ctr = vis.get_view_control()
ctr.set_front([0, 0, -1])
ctr.set_up([0, -1, 0])
ctr.set_zoom(0.5)

try:
    while True:
        # --- Baca Frame Kinect ---
        depth_frame = depth_stream.read_frame()
        depth_data = depth_frame.get_buffer_as_uint16()
        depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

        color_frame = color_stream.read_frame()
        color_data = color_frame.get_buffer_as_uint8()
        color_image = np.frombuffer(color_data, dtype=np.uint8).reshape(480, 640, 3)

        # Subbab 12.3: Implementasi Pembangkitan (Vektorisasi NumPy)
        # Buat meshgrid koordinat piksel (u, v)
        u, v = np.meshgrid(np.arange(640), np.arange(480))

        # Filter noise: Ambil hanya piksel dengan jarak valid (antara 500mm - 4000mm)
        mask = (depth_image > 500) & (depth_image < 4000)
        z = depth_image[mask] / 1000.0  # Konversi mm ke meter
        
        # Persamaan Matematis Back-projection (Subbab 12.2)
        x = ((u[mask] - CX) * z) / FX
        y = ((v[mask] - CY) * z) / FY

        # Subbab 12.4: Fusi Warna (RGB-D)
        # Ambil warna dari citra RGB pada koordinat yang sama, normalisasi ke 0-1
        colors = color_image[mask] / 255.0

        points = np.vstack((x, y, z)).T

        # --- Update Geometri Open3D ---
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Subbab 12.5: Down-sampling Voxel Grid (Ukuran voxel 1 cm)
        # Mengurangi jumlah titik dari ~200rb menjadi ~30rb agar grafis mulus
        pcd_down = pcd.voxel_down_sample(voxel_size=0.01)

        vis.remove_geometry(pcd)
        vis.add_geometry(pcd_down)
        vis.poll_events()
        vis.update_renderer()

        # --- Tampilkan monitor 2D (Hanya untuk monitoring) ---
        depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(np.uint8(depth_display), cv2.COLORMAP_JET)
        rgb_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        # Gabungkan RGB dan Depth untuk monitoring
        combined_2d = np.hstack((rgb_bgr, depth_colored))
        cv2.putText(combined_2d, f"Points: {len(points)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Monitoring RGB & Depth (Tekan Q di sini)', combined_2d)

        # Tekan 'q' pada jendela OpenCV untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Mematikan sistem...")
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    vis.destroy_window()
    cv2.destroyAllWindows()