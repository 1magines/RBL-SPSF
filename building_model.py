# building_model.py
import numpy as np

class Building:
    def __init__(self, num_stories, story_masses_kg, story_stiffnesses_N_m, story_dampings_Ns_m):
        """
        Inisialisasi model gedung.
        :param num_stories: Jumlah lantai
        :param story_masses_kg: List atau array massa per lantai (kg)
        :param story_stiffnesses_N_m: List atau array kekakuan antar lantai (N/m)
        :param story_dampings_Ns_m: List atau array redaman antar lantai (Ns/m)
        """
        self.n_stories = num_stories
        if len(story_masses_kg)!= num_stories or \
           len(story_stiffnesses_N_m)!= num_stories or \
           len(story_dampings_Ns_m)!= num_stories:
            raise ValueError("Panjang array massa, kekakuan, dan redaman harus sama dengan jumlah lantai.")

        self.m_i = np.array(story_masses_kg)
        self.k_i = np.array(story_stiffnesses_N_m)
        self.c_i = np.array(story_dampings_Ns_m)

    def get_mass_matrix(self):
        M = np.diag(self.m_i)
        return M

    def get_stiffness_matrix(self):
        K = np.zeros((self.n_stories, self.n_stories))
        for i in range(self.n_stories):
            if i == 0:
                K[i, i] = self.k_i[i] + (self.k_i[i+1] if self.n_stories > 1 else 0)
                if self.n_stories > 1:
                    K[i, i+1] = -self.k_i[i+1]
            elif i < self.n_stories - 1:
                K[i, i-1] = -self.k_i[i]
                K[i, i] = self.k_i[i] + self.k_i[i+1]
                K[i, i+1] = -self.k_i[i+1]
            else: # Lantai teratas
                K[i, i-1] = -self.k_i[i]
                K[i, i] = self.k_i[i]
        return K

    def get_damping_matrix(self):
        # Asumsi redaman Rayleigh atau model sederhana serupa
        # Untuk kesederhanaan, kita bisa gunakan model diagonal jika c_i adalah redaman absolut per lantai
        # atau model Caughey jika c_i adalah koefisien untuk matriks proporsional
        # Di sini, kita asumsikan c_i adalah redaman antar lantai, serupa dengan k_i
        C = np.zeros((self.n_stories, self.n_stories))
        for i in range(self.n_stories):
            if i == 0:
                C[i, i] = self.c_i[i] + (self.c_i[i+1] if self.n_stories > 1 else 0)
                if self.n_stories > 1:
                    C[i, i+1] = -self.c_i[i+1]
            elif i < self.n_stories - 1:
                C[i, i-1] = -self.c_i[i]
                C[i, i] = self.c_i[i] + self.c_i[i+1]
                C[i, i+1] = -self.c_i[i+1]
            else: # Lantai teratas
                C[i, i-1] = -self.c_i[i]
                C[i, i] = self.c_i[i]
        return C

if __name__ == '__main__':
    # Contoh penggunaan
    num_stories = 3
    masses =   # kg
    stiffnesses = [2e6, 1.8e6, 1.5e6] # N/m
    dampings =  # Ns/m

    building = Building(num_stories, masses, stiffnesses, dampings)
    M_b = building.get_mass_matrix()
    K_b = building.get_stiffness_matrix()
    C_b = building.get_damping_matrix()

    print("Matriks Massa Gedung (M_b):\n", M_b)
    print("\nMatriks Kekakuan Gedung (K_b):\n", K_b)
    print("\nMatriks Redaman Gedung (C_b):\n", C_b)