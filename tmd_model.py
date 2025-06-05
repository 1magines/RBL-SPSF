# tmd_model.py
class TMD:
    def __init__(self, mass_kg, stiffness_N_m, damping_Ns_m, attachment_floor_idx):
        """
        Inisialisasi model TMD.
        :param mass_kg: Massa TMD (kg)
        :param stiffness_N_m: Kekakuan pegas TMD (N/m)
        :param damping_Ns_m: Koefisien redaman TMD (Ns/m)
        :param attachment_floor_idx: Indeks lantai (0-based) tempat TMD dipasang
        """
        self.m_d = mass_kg
        self.k_d = stiffness_N_m
        self.c_d = damping_Ns_m
        self.attach_floor = attachment_floor_idx # Biasanya lantai teratas (n_stories - 1)

if __name__ == '__main__':
    # Contoh penggunaan
    tmd = TMD(mass_kg=100, stiffness_N_m=2e4, damping_Ns_m=200, attachment_floor_idx=2)
    print(f"TMD: massa={tmd.m_d} kg, kekakuan={tmd.k_d} N/m, redaman={tmd.c_d} Ns/m, terpasang di lantai index {tmd.attach_floor}")