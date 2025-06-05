# main_optimization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from building_model import Building
from tmd_model import TMD
from coupled_system import get_coupled_matrices
from simulation_engine import newmark_beta_solver
from excitation_handler import load_el_centro_data
from ann_surrogate_model import create_ann_model, train_ann_model, predict_performance_ann
from optimization_engine import pso_optimizer, objective_function_ann
from data_generation_for_ann import run_single_simulation_for_data_gen # Untuk verifikasi

def run_verification_simulation(building_params, tmd_optimal_params, excitation_signal, dt):
    """ Mirip run_single_simulation_for_data_gen tapi mengembalikan history lengkap """
    num_stories = building_params['num_stories']
    bldg = Building(num_stories, 
                    building_params['masses_kg'], 
                    building_params['stiffnesses_N_m'], 
                    building_params['dampings_Ns_m'])
    
    if tmd_optimal_params: # Jika ada TMD
        tmd_instance = TMD(mass_kg=tmd_optimal_params['mass_kg'], 
                           stiffness_N_m=tmd_optimal_params['stiffness_N_m'], 
                           damping_Ns_m=tmd_optimal_params['damping_Ns_m'], 
                           attachment_floor_idx=num_stories - 1)
        M_sys, C_sys, K_sys = get_coupled_matrices(bldg, tmd_instance)
        n_dof_total = M_sys.shape
    else: # Tanpa TMD
        M_sys = bldg.get_mass_matrix()
        C_sys = bldg.get_damping_matrix()
        K_sys = bldg.get_stiffness_matrix()
        n_dof_total = num_stories

    u0 = np.zeros((n_dof_total, 1))
    v0 = np.zeros((n_dof_total, 1))

    F_ext_building_part = -np.diag(bldg.m_i) @ np.ones((num_stories,1))
    F_ext_history = np.zeros((len(excitation_signal), n_dof_total))
    for i in range(len(excitation_signal)):
        F_ext_history[i, :num_stories] = F_ext_building_part.flatten() * excitation_signal[i]
        # Jika ada TMD, DOF TMD tidak menerima gaya eksternal langsung

    time_vec, u_hist, v_hist, a_hist = newmark_beta_solver(M_sys, C_sys, K_sys, F_ext_history, dt, u0, v0)
    return time_vec, u_hist, v_hist, a_hist

def main():
    # 1. Definisi Properti Gedung
    building_props = {
        'num_stories': 3,
        'masses_kg': ,
        'stiffnesses_N_m': [2e7, 1.8e7, 1.5e7],
        'dampings_Ns_m': [1e4, 9e3, 8e3] 
    }
    top_floor_idx = building_props['num_stories'] - 1

    # 2. Pemuatan Data Eksitasi
    sim_dt = 0.01
    sim_duration = 20.0 # Durasi gempa untuk optimasi dan verifikasi
    
    # Buat file dummy el_centro_NS.txt jika belum ada
    excitation_file = "el_centro_NS.txt"
    if not os.path.exists(excitation_file):
        dummy_data_el_centro = np.random.randn(int(sim_duration/0.02) + 100) * 0.1 
        np.savetxt(excitation_file, dummy_data_el_centro)

    time_excitation, ug_ddot = load_el_centro_data(
        filepath=excitation_file, 
        dt_provided=0.02, 
        target_dt=sim_dt, 
        total_duration_sec=sim_duration
    )

    # 3. Pelatihan JST atau Muat Model JST yang Sudah Dilatih
    # Coba muat data pelatihan jika ada, jika tidak, jalankan data_generation_for_ann.py
    training_data_file = "tmd_training_data.csv"
    ann_model_file = "ann_surrogate_tmd.h5"
    scalers_file = "ann_scalers.json"

    if os.path.exists(ann_model_file) and os.path.exists(scalers_file):
        print("Memuat model JST dan scaler yang sudah ada...")
        ann_model = tf.keras.models.load_model(ann_model_file)
        with open(scalers_file, 'r') as f:
            scaler_params = json.load(f)
        scaler_X = StandardScaler()
        scaler_X.mean_ = np.array(scaler_params['X_mean'])
        scaler_X.scale_ = np.array(scaler_params['X_scale'])
        scaler_y = StandardScaler()
        scaler_y.mean_ = np.array(scaler_params['y_mean'])
        scaler_y.scale_ = np.array(scaler_params['y_scale'])
    else:
        print(f"Model JST atau scaler tidak ditemukan. Memulai generasi data dan pelatihan JST...")
        if not os.path.exists(training_data_file):
            print(f"File data pelatihan {training_data_file} tidak ditemukan. Harap jalankan data_generation_for_ann.py terlebih dahulu.")
            # Untuk demonstrasi, bisa dipanggil di sini, tapi idealnya terpisah.
            # from data_generation_for_ann import generate_training_data, tmd_ranges # (perlu didefinisikan)
            # print("Menjalankan generasi data...")
            # df_train = generate_training_data(...)
            # df_train.to_csv(training_data_file, index=False)
            return 
        
        df_train = pd.read_csv(training_data_file)
        X_train_data = df_train[['md', 'kd', 'cd']].values
        y_train_data = df_train['max_disp_top_floor'].values.reshape(-1,1)

        ann_model = create_ann_model(input_dim=3, output_dim=1, hidden_layers=, dropout_rate=0.1)
        ann_model, _, scaler_X, scaler_y = train_ann_model(ann_model, X_train_data, y_train_data, epochs=200, batch_size=32) # Epochs lebih banyak
        
        ann_model.save(ann_model_file)
        scaler_params_to_save = {
            'X_mean': scaler_X.mean_.tolist(), 'X_scale': scaler_X.scale_.tolist(),
            'y_mean': scaler_y.mean_.tolist(), 'y_scale': scaler_y.scale_.tolist()
        }
        with open(scalers_file, 'w') as f:
            json.dump(scaler_params_to_save, f)
        print("Model JST dan scaler baru telah dilatih dan disimpan.")

    # 4. Jalankan Optimasi PSO
    # Rentang parameter TMD untuk optimasi (bisa sama atau beda dengan rentang sampling data JST)
    # Contoh: m_d (1-5% massa total gedung), k_d, c_d (rentang yang wajar)
    total_building_mass = sum(building_props['masses_kg'])
    opt_bounds = [
        (0.01 * total_building_mass, 0.05 * total_building_mass), # m_d
        (1e4, 5e6),  # k_d
        (100, 2e4)   # c_d
    ]
    
    print("\nMemulai optimasi parameter TMD menggunakan PSO-JST...")
    optimal_tmd_params_array, optimal_fitness_predicted = pso_optimizer(
        objective_function_ann,
        opt_bounds,
        num_particles=30, # Jumlah partikel
        max_iterations=100, # Jumlah iterasi
        ann_model=ann_model,
        scaler_X=scaler_X,
        scaler_y=scaler_y
    )
    
    optimal_tmd_values = {
        'mass_kg': optimal_tmd_params_array,
        'stiffness_N_m': optimal_tmd_params_array,
        'damping_Ns_m': optimal_tmd_params_array
    }
    print("\n--- Parameter TMD Optimal (dari JST-PSO) ---")
    print(optimal_tmd_values)
    print(f"Prediksi performa (max_disp_top_floor) oleh JST: {optimal_fitness_predicted:.4f} m")

    # 5. Verifikasi Parameter TMD Optimal dengan Simulasi Penuh
    print("\nMemverifikasi parameter TMD optimal dengan simulasi penuh...")
    # Simulasi Tanpa TMD
    time_no_tmd, u_no_tmd, _, _ = run_verification_simulation(building_props, None, ug_ddot, sim_dt)
    max_disp_no_tmd = np.max(np.abs(u_no_tmd[:, top_floor_idx]))
    print(f"Respons Tanpa TMD: Perpindahan Maks Lantai Atas = {max_disp_no_tmd:.4f} m")

    # Simulasi Dengan TMD Optimal
    time_opt_tmd, u_opt_tmd, _, _ = run_verification_simulation(building_props, optimal_tmd_values, ug_ddot, sim_dt)
    max_disp_opt_tmd_actual = np.max(np.abs(u_opt_tmd[:, top_floor_idx]))
    print(f"Respons Dengan TMD Optimal: Perpindahan Maks Lantai Atas (aktual) = {max_disp_opt_tmd_actual:.4f} m")
    
    reduction_percentage = ((max_disp_no_tmd - max_disp_opt_tmd_actual) / max_disp_no_tmd) * 100
    print(f"Reduksi perpindahan maksimum: {reduction_percentage:.2f}%")

    # 6. Simpan Parameter TMD Optimal untuk Antarmuka Web
    with open("optimal_tmd_params.json", "w") as f:
        json.dump(optimal_tmd_values, f, indent=4)
    print("\nParameter TMD optimal disimpan ke optimal_tmd_params.json")

    # 7. Plot Hasil Perbandingan
    plt.figure(figsize=(12, 8))
    plt.plot(time_no_tmd, u_no_tmd[:, top_floor_idx], label=f"Tanpa TMD (Maks: {max_disp_no_tmd:.4f} m)")
    plt.plot(time_opt_tmd, u_opt_tmd[:, top_floor_idx], label=f"Dengan TMD Optimal (Maks: {max_disp_opt_tmd_actual:.4f} m)")
    if building_props['num_stories'] < u_opt_tmd.shape: # Jika ada DOF TMD (x_d_rel)
        plt.plot(time_opt_tmd, u_opt_tmd[:, -1], label="Perpindahan Relatif TMD", linestyle=':')
    
    plt.title(f"Respons Perpindahan Lantai Atas (Reduksi {reduction_percentage:.2f}%)")
    plt.xlabel("Waktu (s)")
    plt.ylabel("Perpindahan (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("response_comparison_optimal_tmd.png")
    plt.show()

if __name__ == '__main__':
    main()