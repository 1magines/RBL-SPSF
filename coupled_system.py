# coupled_system.py
import numpy as np

def get_coupled_matrices(building_obj, tmd_obj):
    """
    Membentuk matriks massa, redaman, dan kekakuan untuk sistem gedung-TMD terkopel.
    Asumsi TMD dipasang di lantai teratas (building_obj.n_stories - 1).
    DOF terakhir adalah untuk TMD.
    """
    M_b = building_obj.get_mass_matrix()
    C_b = building_obj.get_damping_matrix()
    K_b = building_obj.get_stiffness_matrix()

    n_dof_building = building_obj.n_stories
    n_dof_total = n_dof_building + 1 # Tambahan 1 DOF untuk TMD

    M_sys = np.zeros((n_dof_total, n_dof_total))
    C_sys = np.zeros((n_dof_total, n_dof_total))
    K_sys = np.zeros((n_dof_total, n_dof_total))

    # Blok gedung
    M_sys[:n_dof_building, :n_dof_building] = M_b
    C_sys[:n_dof_building, :n_dof_building] = C_b
    K_sys[:n_dof_building, :n_dof_building] = K_b

    # Coupling dan TMD
    # Diasumsikan TMD dipasang di lantai 'tmd_obj.attach_floor'
    idx_attach = tmd_obj.attach_floor

    M_sys[idx_attach, idx_attach] += tmd_obj.m_d # Massa TMD ditambahkan ke massa lantai tempat terpasang
                                                # Ini untuk formulasi perpindahan absolut TMD.
                                                # Jika x_d adalah relatif, formulasi M akan berbeda.
                                                # Proposal  menggunakan x_d relatif. Mari sesuaikan.
    
    # Mengacu pada , persamaan gerak massa TMD:
    # m_d * (ddot(x_N) + ddot(x_d)) + c_d * dot(x_d) + k_d * x_d = 0
    # Persamaan gerak lantai N: M_N * ddot(x_N) +... - (c_d * dot(x_d) + k_d * x_d) =...
    # Vektor keadaan: [x_1,..., x_N, x_d]^T

    # Matriks Massa Sistem (dengan x_d relatif)
    M_sys[n_dof_total-1, n_dof_total-1] = tmd_obj.m_d
    M_sys[n_dof_total-1, idx_attach] = tmd_obj.m_d # Coupling term m_d * ddot(x_N)

    # Matriks Redaman Sistem
    C_sys[idx_attach, idx_attach] += tmd_obj.c_d
    C_sys[idx_attach, n_dof_total-1] = -tmd_obj.c_d
    C_sys[n_dof_total-1, idx_attach] = -tmd_obj.c_d # Ini salah jika x_d relatif.
                                                    # Untuk m_d*ddot(x_N) + m_d*ddot(x_d) + c_d*dot(x_d) + k_d*x_d = -m_d*ddot(u_g) (jika TMD juga merasakan ground motion)
                                                    # atau = 0 jika x_d relatif terhadap lantai yg bergerak.
                                                    # Proposal: m_d(ä_N + ä_d) + c_d*ȧ_d + k_d*x_d = 0
                                                    # Ini berarti gaya pada TMD adalah inersia lantai N.
                                                    # Dan gaya pada lantai N adalah - (c_d*ȧ_d + k_d*x_d)

    # Koreksi berdasarkan formulasi umum sistem gedung-TMD dengan x_d relatif:
    # M_sys:
    # M_b 0
    # 0   m_d
    # C_sys:
    # C_b     0
    # 0       c_d
    # K_sys:
    # K_b     0
    # 0       k_d
    # Kemudian coupling terms ditambahkan pada matriks gaya atau persamaan gerak.
    # Alternatifnya, matriks sistem yang diperbesar:
    # [ M_b | 0   ][ddot_x_b]   [ C_b + C_link | -C_link ][dot_x_b]   [ K_b + K_link | -K_link ][x_b]   [ F_b ]
    # [ --- | --- ][--------] + [--------------|---------][-------] + [--------------|---------][---] = [ --- ]
    #[ddot_x_d]  [dot_x_d]  [x_d]   [ F_d ]
    # dimana x_d adalah perpindahan TMD relatif terhadap lantai pemasangan.
    # C_link dan K_link adalah matriks kolom yang menghubungkan dof TMD ke dof lantai pemasangan.

    # Mari gunakan formulasi yang lebih standar untuk sistem N+1 DOF:
    # x_sys = [x_1,..., x_N, x_d_abs]^T dimana x_d_abs adalah perpindahan absolut TMD.
    # Atau x_sys = [x_1,..., x_N, x_d_rel]^T dimana x_d_rel adalah perpindahan TMD relatif thd lantai N.
    # Proposal menggunakan x_d relatif. 

    # Matriks Massa (M_sys) - (N+1) x (N+1)
    # M_sys_corrected = np.zeros((n_dof_total, n_dof_total))
    # M_sys_corrected[:n_dof_building, :n_dof_building] = M_b
    # M_sys_corrected[n_dof_building, n_dof_building] = tmd_obj.m_d
    # M_sys_corrected[n_dof_building, idx_attach] = tmd_obj.m_d # m_d * ddot(x_N)
    # M_sys_corrected[idx_attach, n_dof_building] = tmd_obj.m_d # symmetric

    # Ini adalah formulasi yang lebih umum diterima untuk x_d relatif:
    M_final = np.zeros((n_dof_total, n_dof_total))
    M_final[:n_dof_building, :n_dof_building] = M_b
    M_final[n_dof_building, n_dof_building] = tmd_obj.m_d
    # Tambahkan massa TMD ke lantai tempat ia terpasang untuk gaya inersia lantai tersebut
    M_final[idx_attach, idx_attach] += tmd_obj.m_d # Ini jika x_d adalah perpindahan absolut TMD.
                                                 # Jika x_d relatif, maka M_b[idx_attach,idx_attach] tetap.
                                                 # Persamaan TMD: m_d * (ddot_x_N + ddot_x_d_rel) +... = 0
                                                 # Persamaan lantai N: m_N * ddot_x_N +... - (k_d*x_d_rel + c_d*dot_x_d_rel) = F_N
    # Mari kita gunakan matriks sistem yang jelas untuk N DOF gedung dan 1 DOF TMD (relatif)
    # M_sys has M_b and m_d on diagonal blocks.
    # M_sys[:n_dof_building, :n_dof_building] = M_b
    # M_sys[n_dof_building, n_dof_building] = tmd_obj.m_d

    # C_sys
    C_final = np.zeros((n_dof_total, n_dof_total))
    C_final[:n_dof_building, :n_dof_building] = C_b
    C_final[idx_attach, idx_attach] += tmd_obj.c_d
    C_final[idx_attach, n_dof_building] = -tmd_obj.c_d
    C_final[n_dof_building, idx_attach] = -tmd_obj.c_d
    C_final[n_dof_building, n_dof_building] = tmd_obj.c_d

    # K_sys
    K_final = np.zeros((n_dof_total, n_dof_total))
    K_final[:n_dof_building, :n_dof_building] = K_b
    K_final[idx_attach, idx_attach] += tmd_obj.k_d
    K_final[idx_attach, n_dof_building] = -tmd_obj.k_d
    K_final[n_dof_building, idx_attach] = -tmd_obj.k_d
    K_final[n_dof_building, n_dof_building] = tmd_obj.k_d
    
    # Vektor gaya eksternal juga perlu penyesuaian.
    # Jika F_ext = -M_b * 1 * u_g_ddot, maka untuk sistem gabungan:
    # F_sys =
    # atau F_sys =
    # Proposal: m_d(ä_N + ä_d) + c_d*ȧ_d + k_d*x_d = 0. Ini berarti tidak ada gaya eksternal langsung pada TMD.
    # Gaya efektif pada TMD adalah -m_d * ä_N.
    # Maka, F_sys untuk TMD adalah -m_d * (vektor pengaruh percepatan lantai N).

    # Untuk persamaan M_sys * ddot(q) + C_sys * dot(q) + K_sys * q = P(t)
    # q = [x_1,..., x_N, x_d_rel]^T
    # M_sys_final:
    # [ M_b | 0   ]
    # [ m_d*L | m_d ]  dimana L adalah vektor baris [0...1...0] menunjuk lantai attachment
    
    # Mari sederhanakan dengan asumsi umum:
    M_s = np.zeros((n_dof_total, n_dof_total))
    M_s[:n_dof_building, :n_dof_building] = M_b
    M_s[n_dof_building,n_dof_building] = tmd_obj.m_d
    # Ini adalah massa untuk DOF [x_building, x_d_relative_to_ground]
    # Jika x_d adalah relatif terhadap lantai N, maka persamaan menjadi lebih kompleks.
    # Untuk konsistensi dengan banyak literatur:
    # M_true_coupled = M_s (M_b diagonal, m_d diagonal)
    # C_true_coupled = C_final
    # K_true_coupled = K_final
    # Dan vektor gaya P(t) = [-M_b*1*ug_ddot, 0]^T
    # Namun, persamaan m_d(ä_N + ä_d) +... = 0 menyiratkan matriks massa yang terkopel.

    # Mengikuti formulasi standar Chopra untuk TMD (atau lampiran buku):
    # M* = [[M_b, 0], [0, m_d]]
    # C* =, [-c_d*L, c_d]]
    # K* =, [-k_d*L, k_d]]
    # dimana L adalah vektor [0,..., 1,..., 0] yang menunjukkan lantai tempat TMD terpasang (idx_attach).
    # Vektor keadaan adalah [x_building_displacements, x_d_relative_to_attachment_floor]^T

    L = np.zeros((1, n_dof_building))
    L[0, idx_attach] = 1

    M_coupled = np.block([M_b,                          np.zeros((n_dof_building, 1))],
        [np.zeros((1, n_dof_building)), tmd_obj.m_d                 ])

    C_coupled = np.block(,
        [-tmd_obj.c_d * L,             tmd_obj.c_d       ])

    K_coupled = np.block(,
        [-tmd_obj.k_d * L,             tmd_obj.k_d       ])
    
    return M_coupled, C_coupled, K_coupled

if __name__ == '__main__':
    from building_model import Building
    from tmd_model import TMD
    num_stories = 3
    masses = 
    stiffnesses = [2e6, 1.8e6, 1.5e6]
    dampings = 
    building = Building(num_stories, masses, stiffnesses, dampings)
    
    # TMD dipasang di lantai teratas (indeks 2 untuk 3 lantai)
    tmd = TMD(mass_kg=100, stiffness_N_m=2e4, damping_Ns_m=200, attachment_floor_idx=num_stories-1)
    
    M_sys, C_sys, K_sys = get_coupled_matrices(building, tmd)
    print("Matriks Massa Sistem Terkopel (M_sys):\n", M_sys)
    print("\nMatriks Redaman Sistem Terkopel (C_sys):\n", C_sys)
    print("\nMatriks Kekakuan Sistem Terkopel (K_sys):\n", K_sys)
    print(f"\nDimensi Matriks: M={M_sys.shape}, C={C_sys.shape}, K={K_sys.shape}")