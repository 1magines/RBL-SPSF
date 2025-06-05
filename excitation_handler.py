# excitation_handler.py
import numpy as np

def load_el_centro_data(filepath="el_centro_NS.txt", dt_provided=0.02, target_dt=0.01, total_duration_sec=None):
    """
    Memuat data gempa El Centro (atau data serupa dalam format kolom tunggal).
    Melakukan interpolasi jika target_dt berbeda dari dt_provided.
    :param filepath: Path ke file data (.txt,.dat)
    :param dt_provided: Interval waktu data asli (detik)
    :param target_dt: Interval waktu yang diinginkan untuk output (detik)
    :param total_duration_sec: Durasi total data yang akan digunakan (detik). Jika None, gunakan semua data.
    :return: Tuple (time_vector, acceleration_vector_m_s2)
    """
    try:
        # Asumsi data adalah percepatan dalam g, satu nilai per baris
        # Konversi ke m/s^2
        raw_data = np.loadtxt(filepath) * 9.81 # gravitasi m/s^2
    except FileNotFoundError:
        print(f"File data gempa tidak ditemukan di {filepath}. Menggunakan sinyal dummy.")
        return generate_sinusoidal_excitation(target_dt, 10.0, 2.0, 0.5) # Dummy
    except Exception as e:
        print(f"Error saat memuat data gempa: {e}. Menggunakan sinyal dummy.")
        return generate_sinusoidal_excitation(target_dt, 10.0, 2.0, 0.5) # Dummy

    num_points_raw = len(raw_data)
    time_raw = np.arange(0, num_points_raw * dt_provided, dt_provided)

    if total_duration_sec is not None:
        num_points_to_keep = int(total_duration_sec / dt_provided)
        raw_data = raw_data[:num_points_to_keep]
        time_raw = time_raw[:num_points_to_keep]
        num_points_raw = len(raw_data)

    if dt_provided == target_dt:
        final_time_vector = time_raw
        final_accel_vector = raw_data
    else:
        # Interpolasi
        duration = time_raw[-1]
        final_time_vector = np.arange(0, duration, target_dt)
        final_accel_vector = np.interp(final_time_vector, time_raw, raw_data)
    
    return final_time_vector, final_accel_vector

def generate_sinusoidal_excitation(dt, total_duration_sec, amplitude_m_s2, frequency_hz):
    """
    Menghasilkan eksitasi sinusoidal.
    """
    t = np.arange(0, total_duration_sec, dt)
    accel = amplitude_m_s2 * np.sin(2 * np.pi * frequency_hz * t)
    return t, accel

def generate_random_excitation(dt, total_duration_sec, rms_amplitude_m_s2):
    """
    Menghasilkan eksitasi random (white noise filtered or simple random).
    """
    t = np.arange(0, total_duration_sec, dt)
    # Sinyal random sederhana, bisa diperhalus dengan filter jika perlu
    accel = rms_amplitude_m_s2 * (2 * np.random.rand(len(t)) - 1)
    return t, accel

if __name__ == '__main__':
    # Contoh: Buat file dummy el_centro_NS.txt
    dummy_data = np.random.randn(1000) * 0.1 # percepatan dalam g
    np.savetxt("el_centro_NS.txt", dummy_data)

    dt_sim = 0.01
    time_elcentro, accel_elcentro = load_el_centro_data(dt_provided=0.02, target_dt=dt_sim, total_duration_sec=15)
    print(f"El Centro: {len(accel_elcentro)} poin, durasi {time_elcentro[-1]:.2f} s")

    time_sine, accel_sine = generate_sinusoidal_excitation(dt_sim, 10.0, 2.0, 1.0)
    print(f"Sinusoidal: {len(accel_sine)} poin, durasi {time_sine[-1]:.2f} s")
    
    # Hapus file dummy setelah selesai
    import os
    os.remove("el_centro_NS.txt")