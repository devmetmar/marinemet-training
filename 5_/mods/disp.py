import io
import base64
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np

def show_centered(fig, dpi=100):
    """Tampilkan figure Matplotlib di tengah halaman notebook."""
    # Simpan fig ke buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)

    # Encode ke base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # HTML render
    html = f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" />
    </div>
    """
    display(HTML(html))

def compute_extent(lons, lats, fig_width=10, fig_height=6, padding_frac=0.1):
    """
    Menghitung extent (lon_min, lon_max, lat_min, lat_max) dari data lons dan lats
    dengan penyesuaian rasio figure dan padding.

    Parameters:
    - lons: list atau array longitudes
    - lats: list atau array latitudes
    - fig_width: lebar figure (default: 10)
    - fig_height: tinggi figure (default: 6)
    - padding_frac: proporsi padding dari total rentang (default: 0.1 → 10%)

    Returns:
    - exts: list [lon_min, lon_max, lat_min, lat_max] yang siap digunakan sebagai extent
    """

    # Bounding box awal dari data
    lon_min, lon_max = np.min(lons), np.max(lons)
    lat_min, lat_max = np.min(lats), np.max(lats)

    # Rasio figure (lebar / tinggi)
    fig_ratio = fig_width / fig_height

    # Hitung span data
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    data_ratio = lon_span / lat_span if lat_span != 0 else np.inf

    if data_ratio > fig_ratio:
        # Terlalu lebar → sesuaikan lat agar cocok dengan rasio
        new_lat_span = lon_span / fig_ratio
        lat_center = (lat_min + lat_max) / 2
        lat_min = lat_center - new_lat_span / 2
        lat_max = lat_center + new_lat_span / 2
    else:
        # Terlalu tinggi → sesuaikan lon agar cocok dengan rasio
        new_lon_span = lat_span * fig_ratio
        lon_center = (lon_min + lon_max) / 2
        lon_min = lon_center - new_lon_span / 2
        lon_max = lon_center + new_lon_span / 2

    # Tambahkan padding
    lat_padding = (lat_max - lat_min) * padding_frac
    lon_padding = (lon_max - lon_min) * padding_frac
    lat_min -= lat_padding
    lat_max += lat_padding
    lon_min -= lon_padding
    lon_max += lon_padding

    return [lon_min, lon_max, lat_min, lat_max]

def animate_argo_trajectory(lons, lats, dts, pts, exts, interval=500):
    """
    Membuat animasi lintasan pada peta menggunakan Cartopy dan Matplotlib.

    Parameters:
    - lons: list atau array longitude
    - lats: list atau array latitude
    - dts: list datetime untuk label waktu
    - pts: list label siklus/titik
    - exts: tuple (lon_min, lon_max, lat_min, lat_max) untuk extent peta
    - interval: waktu antar frame dalam milidetik

    Returns:
    - HTML object berisi animasi yang bisa ditampilkan di Jupyter Notebook
    """

    # Setup figure & axis
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection=proj))
    ax.set_extent(exts, crs=proj)
    ax.add_feature(cfeature.GSHHSFeature(scale="high", levels=[1, 2, 3, 4], facecolor="linen"), linewidth=.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.OCEAN)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.left_labels = True
    gl.right_labels = True
    gl.bottom_labels = True

    # Inisialisasi objek grafik
    traj_line, = ax.plot([], [], color='blue', linewidth=1.5, marker='o', markersize=5, transform=proj)
    start_dot, = ax.plot([], [], 'x', color='yellow', markersize=12, transform=proj)
    end_dot, = ax.plot([], [], 'x', color='red', markersize=12, transform=proj)
    text_label = ax.text(0, 0, '', transform=proj, fontsize=10)

    # Fungsi init
    def init():
        traj_line.set_data([], [])
        start_dot.set_data([], [])
        end_dot.set_data([], [])
        text_label.set_text('')
        return traj_line, start_dot, end_dot, text_label

    # Fungsi update untuk animasi
    def update(frame):
        traj_line.set_data(lons[:frame+1], lats[:frame+1])
        start_dot.set_data([lons[0]], [lats[0]])
        end_dot.set_data([lons[frame]], [lats[frame]])
        text_label.set_position((lons[frame] + 0.03, lats[frame] + 0.03))
        text_label.set_text(f"Cycle: {pts[frame]}\n{dts[frame].strftime('%d-%m-%Y')}")
        return traj_line, start_dot, end_dot, text_label

    # Buat animasi
    ani = FuncAnimation(fig, update, frames=len(lons), init_func=init, blit=True, interval=interval)
    return ani, HTML(ani.to_jshtml())

def animate_drifter_trajectory(df, idd, exts, p=10, l=6, pj=6):
    """
    Membuat animasi lintasan drifter pada peta dengan menggunakan Matplotlib dan Cartopy.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame berisi data posisi drifter dengan kolom 'ID', 'longitude', 'latitude', dan 'time'.
    idd : str atau int
        ID drifter yang ingin dianimasikan lintasannya.
    exts : list atau tuple
        Daftar 4 elemen untuk batas peta: [lon_min, lon_max, lat_min, lat_max].
    p : int, default=10
        Nilai lebar relatif untuk menghitung rasio aspek (digunakan untuk ukuran figure).
    l : int, default=6
        Nilai tinggi relatif untuk menghitung rasio aspek (digunakan untuk ukuran figure).
    pj : int, default=6
        Lebar figure output dalam satuan inchi (inch), digunakan untuk menentukan `figsize`.

    Returns:
    -------
    ani : matplotlib.animation.FuncAnimation
        Objek animasi yang bisa disimpan sebagai file (GIF/MP4) atau diproses lebih lanjut.
    HTML : IPython.display.HTML
        Objek HTML untuk menampilkan animasi secara langsung di Jupyter Notebook.

    Notes:
    -----
    - Fungsi ini menyaring DataFrame berdasarkan `ID` drifter, lalu menganimasikan lintasannya di atas peta.
    - Titik awal ditandai dengan 'x' kuning, titik saat ini dengan 'x' merah.
    - Label teks menunjukkan ID dan tanggal posisi saat ini.
    - Gunakan `ani.save('nama.gif', writer='pillow')` untuk menyimpan animasi sebagai GIF.
    """
    
    # Filter data berdasarkan ID
    df_id = df[df["ID"] == idd]
    
    # Setup figure & axis
    proj = ccrs.PlateCarree()
    rat = p / l
    fig, ax = plt.subplots(figsize=(pj, pj / rat), subplot_kw=dict(projection=proj))
    ax.set_extent(exts, crs=proj)
    ax.add_feature(cfeature.GSHHSFeature(scale="high", levels=[1, 2, 3, 4], facecolor="linen"), linewidth=.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.OCEAN)

    # Inisialisasi line dan titik
    traj_line, = ax.plot([], [], color='blue', linewidth=1.5, marker='o', markersize=5, transform=proj)
    start_dot, = ax.plot([], [], 'x', color='yellow', markersize=12, transform=proj)
    end_dot, = ax.plot([], [], 'x', color='red', markersize=12, transform=proj)
    text_label = ax.text(0, 0, '', transform=proj, fontsize=10)

    # Fungsi init
    def init():
        traj_line.set_data([], [])
        start_dot.set_data([], [])
        end_dot.set_data([], [])
        text_label.set_text('')
        return traj_line, start_dot, end_dot, text_label

    # Fungsi update untuk animasi
    def update(frame):
        lons = df_id['longitude'].values[:frame+1]
        lats = df_id['latitude'].values[:frame+1]
        traj_line.set_data(lons, lats)
        start_dot.set_data([lons[0]], [lats[0]])
        end_dot.set_data([lons[frame]], [lats[frame]])
        text_label.set_position((lons[frame]+0.03, lats[frame]+0.03))
        text_label.set_text(f"{idd}\n\n\n{pd.to_datetime(df_id['time'].values[frame]).strftime('%d-%m-%Y')}")
        return traj_line, start_dot, end_dot, text_label

    # Buat animasi
    ani = FuncAnimation(fig, update, frames=len(df_id), init_func=init, blit=True, interval=500)
    return ani, HTML(ani.to_jshtml())



    
