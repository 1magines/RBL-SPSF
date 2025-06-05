# data_generation_for_ann.py
import numpy as np
import pandas as pd
from building_model import Building
from tmd_model import TMD
from coupled_system import get_coupled_matrices
from simulation_engine import newmark_beta_solver
from excitation_handler import load_el_centro_data # atau eksitasi lain
from scipy.stats import qmc # For Latin Hypercube Sampling

def run_single_simulation_for_data_gen(building_params, tmd_params, excitation_signal, dt):
    """
    Menjalankan satu simulasi dan mengembalikan metrik performa.
    :param building_params: Dict properti gedung.
    :param tmd_params: Dict properti TMD (mass_kg, stiffness_N_m, damping_Ns_m).
    :param excitation_signal: Array percepatan tanah.
    :param dt: Langkah waktu simulasi.
    :return: Metrik performa (misal, max displacement lantai atas).
    """
    num_stories = building_params['num_stories']
    bldg = Building(num_stories, 
                    building_params['masses_kg'], 
                    building_params['stiffnesses_N_m'], 
                    building_params['dampings_Ns_m'])
    
    # TMD selalu di lantai atas untuk kasus ini
    tmd_instance = TMD(mass_kg=tmd_params['mass_kg'], 
                       stiffness_N_m=tmd_params['stiffness_N_m'], 
                       damping_Ns_m=tmd_params['damping_Ns_m'], 
                       attachment_floor_idx=num_stories - 1)

    M_sys, C_sys, K_sys = get_coupled_matrices(bldg, tmd_instance)
    
    n_dof_total = M_sys.shape
    u0 = np.zeros((n_dof_total, 1))
    v0 = np.zeros((n_dof_total, 1))

    # Bentuk vektor gaya eksternal P(t) = -M_eff * ug_ddot
    # M_eff untuk gedung adalah M_b * 1_vector
    # M_eff untuk TMD (jika x_d relatif) adalah m_d * (influence vector dari ug_ddot ke ddot_x_N)
    # Untuk formulasi M_coupled, C_coupled, K_coupled yang digunakan:
    # P(t) =
    # Atau jika P_tmd = -m_d * ddot_x_N, maka perlu iterasi atau formulasi beda.
    # Sesuai  M*ä(t) +... = -M*1*üg(t)
    # Untuk sistem terkopel, F_ext = -[M_b*1; m_d*1_scalar_for_tmd_dof] * ug_ddot
    # Vektor pengaruh untuk gaya gempa pada sistem terkopel
    # q = [x_building, x_d_relative]^T
    # F_load_vector = np.zeros(n_dof_total)
    # F_load_vector[:num_stories] = -bldg.m_i # Ini salah, harusnya M_b * 1
    # F_load_vector[num_stories] = -tmd_instance.m_d # Jika TMD juga merasakan ground motion
    
    # F_ext = - M_sys @ influence_vector_ones * excitation_signal
    # Untuk M_coupled yang blok diagonal M_b dan m_d:
    influence_vector = np.ones(n_dof_total) # Semua DOF merasakan percepatan tanah (model dasar)
                                            # Ini berarti x_d adalah relatif terhadap tanah, bukan lantai N.
                                            # Jika x_d relatif terhadap lantai N, maka F_ext_tmd = 0.
    # Mari konsisten dengan formulasi M,C,K coupled dimana x_d adalah relatif thd lantai N.
    # Maka gaya hanya bekerja pada DOF gedung.
    F_ext_building_part = -np.diag(bldg.m_i) @ np.ones((num_stories,1)) # -M_b * 1
    
    F_ext_history = np.zeros((len(excitation_signal), n_dof_total))
    for i in range(len(excitation_signal)):
        F_ext_history[i, :num_stories] = F_ext_building_part.flatten() * excitation_signal[i]
        # DOF TMD tidak menerima gaya eksternal langsung (karena x_d relatif)
        # F_ext_history[i, num_stories] = -tmd_instance.m_d * excitation_signal[i] # JIKA x_d relatif ke tanah

    _, u_hist, _, _ = newmark_beta_solver(M_sys, C_sys, K_sys, F_ext_history, dt, u0, v0)
    
    # Metrik performa: perpindahan maksimum absolut lantai teratas
    max_disp_top_floor = np.max(np.abs(u_hist[:, num_stories - 1]))
    return max_disp_top_floor


def generate_training_data(building_params, tmd_param_ranges, num_samples, 
                           excitation_filepath, excitation_dt_provided, sim_dt, sim_duration):
    """
    Menghasilkan dataset pelatihan (input: TMD params, output: performance_metric).
    :param tmd_param_ranges: Dict {'mass_kg': (min,max), 'stiffness_N_m': (min,max), 'damping_Ns_m': (min,max)}
    :return: pandas DataFrame dengan kolom 'md', 'kd', 'cd', 'max_disp'
    """
    _, excitation_data = load_el_centro_data(
        filepath=excitation_filepath, 
        dt_provided=excitation_dt_provided, 
        target_dt=sim_dt, 
        total_duration_sec=sim_duration
    )

    # Gunakan Latin Hypercube Sampling untuk parameter TMD
    sampler = qmc.LatinHypercube(d=3, seed=42) # 3 parameter TMD
    samples_unit_hypercube = sampler.random(n=num_samples)
    
    # Scale samples ke rentang parameter yang diinginkan
    md_samples = qmc.scale(samples_unit_hypercube[:, 0], tmd_param_ranges['mass_kg'], tmd_param_ranges['mass_kg'])
    kd_samples = qmc.scale(samples_unit_hypercube[:, 1], tmd_param_ranges['stiffness_N_m'], tmd_param_ranges['stiffness_N_m'])
    cd_samples = qmc.scale(samples_unit_hypercube[:, 2], tmd_param_ranges['damping_Ns_m'], tmd_param_ranges['damping_Ns_m'])

    results =
    print(f"Memulai generasi {num_samples} sampel data pelatihan...")
    for i in range(num_samples):
        current_tmd_params = {
            'mass_kg': md_samples[i],
            'stiffness_N_m': kd_samples[i],
            'damping_Ns_m': cd_samples[i]
        }
        
        performance_metric = run_single_simulation_for_data_gen(
            building_params, current_tmd_params, excitation_data, sim_dt
        )
        
        results.append({
            'md': current_tmd_params['mass_kg'],
            'kd': current_tmd_params['stiffness_N_m'],
            'cd': current_tmd_params['damping_Ns_m'],
            'max_disp_top_floor': performance_metric
        })
        if (i + 1) % (num_samples // 10) == 0: # Cetak progres setiap 10%
             print(f"Selesai {i+1}/{num_samples} sampel...")

    return pd.DataFrame(results)

if __name__ == '__main__':
    # Parameter gedung (contoh)
    building_props = {
        'num_stories': 3,
        'masses_kg': ,      # kg
        'stiffnesses_N_m': [2e7, 1.8e7, 1.5e7], # N/m
        'dampings_Ns_m': [1e4, 9e3, 8e3]        # Ns/m (estimasi)
    }

    # Rentang parameter TMD untuk sampling
    tmd_ranges = {
        'mass_kg': (0.01 * sum(building_props['masses_kg']), 0.05 * sum(building_props['masses_kg'])), # 1-5% massa total gedung
        'stiffness_N_m': (1e4, 5e6),  # N/m
        'damping_Ns_m': (100, 2e4)    # Ns/m
    }
    
    num_training_samples = 100 # Jumlah kecil untuk tes cepat, idealnya ratusan/ribuan
    
    # Buat file dummy el_centro_NS.txt jika belum ada
    if not os.path.exists("el_centro_NS.txt"):
        dummy_data = np.random.randn(2000) * 0.1 # percepatan dalam g, 2000 poin @ 0.02s = 40s
        np.savetxt("el_centro_NS.txt", dummy_data)

    dataset_df = generate_training_data(
        building_props, 
        tmd_ranges, 
        num_training_samples,
        excitation_filepath="el_centro_NS.txt", # Pastikan file ini ada
        excitation_dt_provided=0.02,
        sim_dt=0.01,
        sim_duration=20.0 # Durasi gempa yang digunakan untuk tiap simulasi
    )
    
    print("\nDataset Pelatihan yang Dihasilkan:")
    print(dataset_df.head())
    dataset_df.to_csv("tmd_training_data.csv", index=False)
    print("\nDataset disimpan ke tmd_training_data.csv")

    # if os.path.exists("el_centro_NS.txt"): # Hapus jika dummy
    #    if np.allclose(np.loadtxt("el_centro_NS.txt"), dummy_data):
    #         os.remove("el_centro_NS.txt")