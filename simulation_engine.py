# simulation_engine.py
import numpy as np

def newmark_beta_solver(M, C, K, F_ext_time_history, dt, u0, v0, gamma=0.5, beta=0.25):
    """
    Menyelesaikan persamaan gerak M*u_ddot + C*u_dot + K*u = F(t) menggunakan metode Newmark-Beta.
    :param M: Matriks massa (n_dof x n_dof)
    :param C: Matriks redaman (n_dof x n_dof)
    :param K: Matriks kekakuan (n_dof x n_dof)
    :param F_ext_time_history: Array gaya eksternal vs waktu (n_steps x n_dof)
    :param dt: Langkah waktu (detik)
    :param u0: Vektor perpindahan awal (n_dof x 1)
    :param v0: Vektor kecepatan awal (n_dof x 1)
    :param gamma: Parameter Newmark
    :param beta: Parameter Newmark
    :return: Tuple (time_vector, displacement_history, velocity_history, acceleration_history)
    """
    n_steps = F_ext_time_history.shape
    n_dof = M.shape

    u = np.zeros((n_steps, n_dof))
    v = np.zeros((n_steps, n_dof))
    a = np.zeros((n_steps, n_dof))
    time_vector = np.arange(0, n_steps * dt, dt)

    # Kondisi awal
    u[0, :] = u0.flatten()
    v[0, :] = v0.flatten()
    
    # Hitung percepatan awal a0 dari persamaan gerak: M*a0 = F(0) - C*v0 - K*u0
    # Perlu invers matriks M. Jika M diagonal, lebih mudah.
    try:
        M_inv = np.linalg.inv(M)
        a[0, :] = M_inv @ (F_ext_time_history[0, :].reshape(-1,1) - C @ v0 - K @ u0).flatten()
    except np.linalg.LinAlgError: # Jika M singular
        print("Error: Matriks massa singular. Tidak dapat menghitung percepatan awal.")
        # Fallback: jika M diagonal (umumnya iya untuk model massa terkonsentrasi)
        if np.all(M == np.diag(np.diag(M))):
             a[0, :] = (F_ext_time_history[0, :].reshape(-1,1) - C @ v0 - K @ u0).flatten() / np.diag(M)
        else: # Jika M tidak diagonal dan singular, ini masalah besar
            raise

    # Koefisien Newmark
    a1 = 1 / (beta * dt**2)
    a2 = 1 / (beta * dt)
    a3 = (1 / (2 * beta)) - 1
    a4 = gamma / (beta * dt)
    a5 = gamma / beta - 1
    a6 = dt * (gamma / (2 * beta) - 1)

    # Matriks efektif K_hat = K + a1*M + a4*C
    K_hat = K + a1 * M + a4 * C
    try:
        K_hat_inv = np.linalg.inv(K_hat)
    except np.linalg.LinAlgError:
        print("Error: Matriks kekakuan efektif K_hat singular.")
        raise

    for i in range(n_steps - 1):
        F_i_plus_1 = F_ext_time_history[i+1, :].reshape(-1,1)
        
        # Prediktor gaya efektif P_hat
        term_M = M @ (a1 * u[i, :].reshape(-1,1) + a2 * v[i, :].reshape(-1,1) + a3 * a[i, :].reshape(-1,1))
        term_C = C @ (a4 * u[i, :].reshape(-1,1) + a5 * v[i, :].reshape(-1,1) + a6 * a[i, :].reshape(-1,1))
        P_hat = F_i_plus_1 + term_M + term_C

        # Solve untuk u[i+1]
        u[i+1, :] = (K_hat_inv @ P_hat).flatten()

        # Update percepatan dan kecepatan
        a[i+1, :] = (a1 * (u[i+1, :].reshape(-1,1) - u[i, :].reshape(-1,1)) - a2 * v[i, :].reshape(-1,1) - a3 * a[i, :].reshape(-1,1)).flatten()
        v[i+1, :] = (v[i, :].reshape(-1,1) + dt * ( (1-gamma)*a[i, :].reshape(-1,1) + gamma*a[i+1, :].reshape(-1,1) )).flatten()
        
    return time_vector, u, v, a

if __name__ == '__main__':
    # Contoh SDOF sederhana untuk tes solver
    m_val = 1.0  # kg
    k_val = (2 * np.pi)**2  # N/m (f_n = 1 Hz)
    c_val = 0.1 * 2 * np.sqrt(m_val * k_val) # 5% redaman

    M_sdof = np.array([[m_val]])
    K_sdof = np.array([[k_val]])
    C_sdof = np.array([[c_val]])

    dt_sdof = 0.01
    duration_sdof = 10.0
    time_sdof, F_sdof_hist = generate_sinusoidal_excitation(dt_sdof, duration_sdof, 1.0, 1.0) # Eksitasi resonan
    F_sdof_hist = F_sdof_hist.reshape(-1,1) # n_steps x n_dof

    u0_sdof = np.array([[0.0]])
    v0_sdof = np.array([[0.0]])

    t_resp, u_resp, v_resp, a_resp = newmark_beta_solver(M_sdof, C_sdof, K_sdof, F_sdof_hist, dt_sdof, u0_sdof, v0_sdof)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t_resp, u_resp[:, 0], label="Perpindahan (m)")
    plt.title("Respons SDOF dengan Newmark-Beta")
    plt.xlabel("Waktu (s)")
    plt.ylabel("Perpindahan (m)")
    plt.grid(True)
    plt.legend()
    plt.show()