# app_flask.py
from flask import Flask, request, jsonify
from flask_cors import CORS # Untuk mengizinkan request dari domain berbeda (localhost:port lain)
import numpy as np
import os

# Import modul-modul yang diperlukan dari proyek
from building_model import Building
from tmd_model import TMD
from coupled_system import get_coupled_matrices
from simulation_engine import newmark_beta_solver
from excitation_handler import load_el_centro_data

app = Flask(__name__)
CORS(app) # Mengizinkan semua origin secara default, bisa dikonfigurasi lebih ketat

# --- Konfigurasi Global (bisa dimuat dari file atau didefinisikan di sini) ---
# Properti gedung default (bisa di-override oleh request dari frontend)
DEFAULT_BUILDING_PROPS = {
    'num_stories': 3,
    'masses_kg': ,
    'stiffnesses_N_m': [2e7, 1.8e7, 1.5e7],
    'dampings_Ns_m': [1e4, 9e3, 8e3]
}
# Eksitasi default
SIM_DT = 0.01
SIM_DURATION = 20.0 # Durasi simulasi untuk antarmuka web
EXCITATION_FILE = "el_centro_NS.txt"

# Muat data eksitasi sekali saat server start
if not os.path.exists(EXCITATION_FILE):
    print(f"PERINGATAN: File eksitasi '{EXCITATION_FILE}' tidak ditemukan. Membuat dummy.")
    dummy_data_el_centro = np.random.randn(int(SIM_DURATION/0.02) + 200) * 0.1 
    np.savetxt(EXCITATION_FILE, dummy_data_el_centro)

TIME_EXCITATION_GLOBAL, UG_DDOT_GLOBAL = load_el_centro_data(
    filepath=EXCITATION_FILE,
    dt_provided=0.02, # Asumsi dt data El Centro
    target_dt=SIM_DT,
    total_duration_sec=SIM_DURATION
)
print(f"Data eksitasi global dimuat: {len(UG_DDOT_GLOBAL)} poin.")

@app.route('/simulate_html', methods=)
def simulate_for_html():
    try:
        data = request.get_json()
        
        # Ambil parameter gedung dari request, atau gunakan default
        num_stories = int(data.get('num_stories', DEFAULT_BUILDING_PROPS['num_stories']))
        # Untuk kesederhanaan, asumsikan struktur massa, kekakuan, redaman gedung tetap dari default
        # atau bisa juga dikirim dari frontend jika slidernya ada.
        # Di sini, kita fokus pada parameter TMD yang diubah-ubah.
        
        bldg_props_current = DEFAULT_BUILDING_PROPS # Gunakan default untuk gedung
        # Jika frontend mengirim parameter gedung:
        # bldg_props_current['masses_kg'] =)]
        #... dan seterusnya untuk stiffnesses, dampings

        tmd_params_req = data.get('tmd_params') # { 'mass_kg': val, 'stiffness_N_m': val, 'damping_Ns_m': val }
        use_tmd = data.get('use_tmd', False)

        # --- Logika Simulasi (mirip run_verification_simulation) ---
        bldg = Building(bldg_props_current['num_stories'], 
                        bldg_props_current['masses_kg'], 
                        bldg_props_current['stiffnesses_N_m'], 
                        bldg_props_current['dampings_Ns_m'])
        
        top_floor_idx_sim = bldg_props_current['num_stories'] - 1

        if use_tmd and tmd_params_req:
            tmd_instance = TMD(mass_kg=float(tmd_params_req['mass_kg']), 
                               stiffness_N_m=float(tmd_params_req['stiffness_N_m']), 
                               damping_Ns_m=float(tmd_params_req['damping_Ns_m']), 
                               attachment_floor_idx=top_floor_idx_sim)
            M_sys, C_sys, K_sys = get_coupled_matrices(bldg, tmd_instance)
            n_dof_total = M_sys.shape
        else: # Tanpa TMD
            M_sys = bldg.get_mass_matrix()
            C_sys = bldg.get_damping_matrix()
            K_sys = bldg.get_stiffness_matrix()
            n_dof_total = bldg_props_current['num_stories']

        u0 = np.zeros((n_dof_total, 1))
        v0 = np.zeros((n_dof_total, 1))

        F_ext_building_part = -np.diag(bldg.m_i) @ np.ones((bldg.n_stories,1))
        F_ext_history = np.zeros((len(UG_DDOT_GLOBAL), n_dof_total))
        for i in range(len(UG_DDOT_GLOBAL)):
            F_ext_history[i, :bldg.n_stories] = F_ext_building_part.flatten() * UG_DDOT_GLOBAL[i]

        time_vec, u_hist, _, _ = newmark_beta_solver(M_sys, C_sys, K_sys, F_ext_history, SIM_DT, u0, v0)
        
        # Siapkan data untuk dikirim kembali ke frontend
        # Kirim hanya perpindahan lantai dan TMD (jika ada)
        response_data = {
            'time': time_vec.tolist(),
            'displacements_building': u_hist[:, :bldg.n_stories].tolist(), # Semua lantai gedung
        }
        if use_tmd and tmd_params_req and u_hist.shape > bldg.n_stories:
            response_data['displacement_tmd_relative'] = u_hist[:, bldg.n_stories].tolist() # DOF terakhir adalah x_d_rel

        return jsonify(response_data)

    except Exception as e:
        print(f"Error di endpoint /simulate_html: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/get_optimal_tmd', methods=)
def get_optimal_tmd_params():
    optimal_params_file = "optimal_tmd_params.json"
    if os.path.exists(optimal_params_file):
        with open(optimal_params_file, 'r') as f:
            params = json.load(f)
        return jsonify(params)
    else:
        return jsonify({'error': 'File parameter optimal tidak ditemukan.'}), 404

if __name__ == '__main__':
    print("Pastikan file eksitasi (misal, el_centro_NS.txt) ada di direktori yang sama.")
    print("Jalankan server Flask pada http://127.0.0.1:5000")
    app.run(debug=True, port=5000) # debug=True hanya untuk pengembangan