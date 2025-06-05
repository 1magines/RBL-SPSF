# visualization_python.py
import matplotlib.pyplot as plt
import numpy as np

def plot_dynamic_response(time, displacements, title="Respons Dinamik Struktur", y_label="Perpindahan (m)"):
    """
    Memplot respons dinamik semua DOF.
    :param time: Vektor waktu.
    :param displacements: Matriks perpindahan (n_steps x n_dof).
    """
    plt.figure(figsize=(10, 6))
    num_dof = displacements.shape
    for i in range(num_dof):
        plt.plot(time, displacements[:, i], label=f'DOF {i+1}')
    
    plt.title(title)
    plt.xlabel("Waktu (s)")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_floor_displacement_comparison(time_arrays, displacement_arrays, labels, floor_idx, title_suffix=""):
    """
    Membandingkan perpindahan satu lantai tertentu dari beberapa simulasi.
    :param time_arrays: List array waktu.
    :param displacement_arrays: List matriks perpindahan.
    :param labels: List label untuk setiap plot.
    :param floor_idx: Indeks lantai yang akan diplot.
    """
    plt.figure(figsize=(12, 7))
    for i, time_vec in enumerate(time_arrays):
        disp_floor = displacement_arrays[i][:, floor_idx]
        max_abs_disp = np.max(np.abs(disp_floor))
        plt.plot(time_vec, disp_floor, label=f"{labels[i]} (Maks: {max_abs_disp:.4f} m)")
    
    plt.title(f"Perbandingan Respons Perpindahan Lantai {floor_idx + 1} {title_suffix}")
    plt.xlabel("Waktu (s)")
    plt.ylabel("Perpindahan (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Contoh penggunaan (data dummy)
    t_dummy = np.linspace(0, 10, 500)
    u1_dummy = np.sin(t_dummy) * np.exp(-0.1*t_dummy)
    u2_dummy = 0.5 * np.sin(t_dummy*1.5 + 0.5) * np.exp(-0.15*t_dummy)
    
    disp_matrix_dummy1 = np.vstack((u1_dummy, u1_dummy*0.8, u1_dummy*0.6)).T # 3 DOF
    disp_matrix_dummy2 = np.vstack((u2_dummy, u2_dummy*0.7, u2_dummy*0.5)).T # 3 DOF

    # plot_dynamic_response(t_dummy, disp_matrix_dummy1, title="Contoh Respons Dinamik")
    
    plot_floor_displacement_comparison(
        [t_dummy, t_dummy],
        [disp_matrix_dummy1, disp_matrix_dummy2],
        ["Kasus 1", "Kasus 2"],
        floor_idx=0, # Lantai pertama
        title_suffix=" (Contoh)"
    )