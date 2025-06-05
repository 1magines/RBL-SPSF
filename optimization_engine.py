# optimization_engine.py
import numpy as np
from ann_surrogate_model import predict_performance_ann # Untuk fungsi objektif

# --- Implementasi PSO Sederhana ---
class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([np.random.uniform(-abs(high-low)*0.1, abs(high-low)*0.1) for low, high in bounds])
        self.best_position = np.copy(self.position)
        self.fitness = float('inf')
        self.best_fitness = float('inf')
        self.bounds = bounds

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

        # Batasi kecepatan untuk mencegah ledakan
        for i in range(len(self.velocity)):
            max_vel = (self.bounds[i] - self.bounds[i]) * 0.5 # Batas kecepatan 50% dari rentang
            self.velocity[i] = np.clip(self.velocity[i], -max_vel, max_vel)


    def update_position(self):
        self.position += self.velocity
        # Terapkan batasan (clipping)
        for i in range(len(self.position)):
            self.position[i] = np.clip(self.position[i], self.bounds[i], self.bounds[i])

def pso_optimizer(objective_func, bounds, num_particles, max_iterations, ann_model, scaler_X, scaler_y):
    """
    Particle Swarm Optimization.
    :param objective_func: Fungsi yang akan diminimalkan. Signature: objective_func(params_array, ann_model, scaler_X, scaler_y)
    :param bounds: List of tuples [(low1, high1), (low2, high2),...] untuk setiap parameter.
    :param num_particles: Jumlah partikel.
    :param max_iterations: Jumlah iterasi maksimum.
    :param ann_model, scaler_X, scaler_y: Diperlukan oleh objective_func.
    :return: Tuple (best_global_position, best_global_fitness)
    """
    swarm = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = np.copy(swarm.position) # Inisialisasi
    global_best_fitness = float('inf')

    # Inisialisasi fitness partikel awal
    initial_positions = np.array([p.position for p in swarm])
    initial_fitnesses = objective_func(initial_positions, ann_model, scaler_X, scaler_y).flatten()

    for i, particle in enumerate(swarm):
        particle.fitness = initial_fitnesses[i]
        if particle.fitness < particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_position = np.copy(particle.position)
        if particle.fitness < global_best_fitness:
            global_best_fitness = particle.fitness
            global_best_position = np.copy(particle.position)
            
    print(f"PSO Iter 0: Best Fitness = {global_best_fitness:.4f}")

    for iteration in range(max_iterations):
        for particle in swarm:
            particle.update_velocity(global_best_position)
            particle.update_position()

            # Evaluasi fitness partikel (satu per satu untuk PSO standar)
            current_fitness = objective_func(particle.position.reshape(1, -1), ann_model, scaler_X, scaler_y)
            particle.fitness = current_fitness
            
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = np.copy(particle.position)
            
            if particle.fitness < global_best_fitness:
                global_best_fitness = particle.fitness
                global_best_position = np.copy(particle.position)
        
        if (iteration + 1) % 10 == 0 or iteration == max_iterations -1 : # Cetak progres setiap 10 iterasi
            print(f"PSO Iter {iteration + 1}/{max_iterations}: Best Fitness = {global_best_fitness:.4f}, Params = {global_best_position}")
            
    return global_best_position, global_best_fitness

# Fungsi objektif yang menggunakan JST
def objective_function_ann(tmd_params_array, ann_model, scaler_X, scaler_y):
    """
    Fungsi objektif untuk optimasi, menggunakan model JST untuk prediksi.
    :param tmd_params_array: NumPy array parameter TMD (n_samples x 3 untuk [md, kd, cd]).
    :param ann_model, scaler_X, scaler_y: Model JST terlatih dan scaler-nya.
    :return: Nilai performa yang diprediksi (misalnya, max displacement), untuk diminimalkan.
    """
    # ann_model memprediksi metrik performa (misal, max displacement)
    # Kita ingin meminimalkan metrik ini.
    predicted_metric = predict_performance_ann(ann_model, tmd_params_array, scaler_X, scaler_y)
    return predicted_metric # Asumsi JST dilatih untuk memprediksi nilai yang ingin diminimalkan

if __name__ == '__main__':
    # Contoh penggunaan PSO dengan dummy ANN
    # Buat dan latih model JST dummy (seperti di ann_surrogate_model.py __main__)
    num_samples = 200
    X_dummy = np.random.rand(num_samples, 3) * np.array([100, 1e5, 1e3]) 
    y_dummy = (0.5 * X_dummy[:,0]/50 - 0.2 * (X_dummy[:,1]/5e4 -1)**2 - 0.1 * (X_dummy[:,2]/5e2 -1)**2 + 
               np.random.randn(num_samples) * 0.01) # Fungsi lebih kompleks
    y_dummy = y_dummy.reshape(-1,1)

    from ann_surrogate_model import create_ann_model, train_ann_model
    ann_dummy = create_ann_model(input_dim=3, output_dim=1, hidden_layers=)
    ann_trained_dummy, _, sc_X_dummy, sc_y_dummy = train_ann_model(ann_dummy, X_dummy, y_dummy, epochs=30, batch_size=16)

    # Definisikan batasan untuk parameter TMD [m_d, k_d, c_d]
    # Contoh: m_d (10-200 kg), k_d (1e3-2e5 N/m), c_d (50-2e3 Ns/m)
    param_bounds = [(10, 200), (1e3, 2e5), (50, 2e3)] 

    num_particles_pso = 20
    max_iterations_pso = 50 # Iterasi sedikit untuk tes cepat

    print("\nMemulai Optimasi PSO dengan JST sebagai fungsi objektif...")
    best_params, best_fitness_val = pso_optimizer(
        objective_function_ann, 
        param_bounds, 
        num_particles_pso, 
        max_iterations_pso,
        ann_trained_dummy, sc_X_dummy, sc_y_dummy
    )

    print("\n--- Hasil Optimasi PSO ---")
    print(f"Parameter TMD Optimal (m_d, k_d, c_d): {best_params}")
    print(f"Nilai Fungsi Objektif (prediksi JST): {best_fitness_val}")

    # Verifikasi dengan fungsi asli (jika diketahui, dalam kasus nyata ini adalah simulasi penuh)
    # Untuk dummy case ini, kita tidak punya fungsi asli yang mudah, tapi JST adalah aproksimasinya.