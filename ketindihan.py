import math    # buat sin, cos, tan, exp, sqrt
import random  # buat ngacak populasi, crossover point, mutasi

# ==============================================================================
# GA buat nyari nilai MINIMUM dari fungsi f(x1, x2):
#   f(x1, x2) = -(sin(x1) * cos(x2) * tan(x1+x2) + 0.5 * exp(1 - sqrt(x2^2)))
#   domain: -10 <= x1, x2 <= 10
#
# alurnya:
#   1. bikin 100 kandidat solusi acak (populasi)
#   2. nilai tiap kandidat -> disebut fitness
#   3. yang bagus lebih sering kepilih jadi orangtua
#   4. orangtua kawin -> anak dapet campuran gen keduanya (crossover)
#   5. sesekali ada perubahan acak kecil di anak (mutasi) biar ga nge-stuck
#   6. ulangi N generasi, ambil yang terbaik
# ==============================================================================


# ==============================================================================
# PARAMETER GA — ubah di sini kalau mau eksperimen
# ==============================================================================

POP_SIZE = 100   # jumlah kandidat per generasi
N_BITS   = 16    # tiap variabel dikodekan jadi 16 bit -> 2^16 = 65536 kemungkinan nilai, presisi ~0.0003
N_GEN    = 100   # jumlah generasi (kriteria penghentian)
PC       = 0.80  # peluang crossover 80% — 20% sisanya anak = copy orangtua
PM       = 0.01  # peluang mutasi per bit 1% — biar GA bisa explore area baru
X_MIN    = -10.0 # batas bawah domain
X_MAX    =  10.0 # batas atas domain


# ==============================================================================
# FUNGSI OBJEKTIF — fungsi yang mau dicari nilai minimumnya
# ==============================================================================
def fungsi_objektif(x1, x2):
    # tan() bisa meledak mendekati tak terhingga di titik asimptotnya (x1+x2 ≈ π/2, 3π/2, dst.)
    # kalau itu terjadi, anggap titik ini tidak valid
    try:
        tan_val = math.tan(x1 + x2)              # hitung tan(x1+x2), ini yang rawan meledak

        if abs(tan_val) > 1e6:                   # kalau tan udah gila (> sejuta), skip
            return float('inf')                  # inf = "buang aja, ga keitung"

        exp_val = math.exp(1 - math.sqrt(x2 ** 2))              # hitung exp(1 - sqrt(x2^2))
        hasil   = -(math.sin(x1) * math.cos(x2) * tan_val       # ini rumus f(x1,x2)-nya
                    + 0.5 * exp_val)
        return hasil

    except (ValueError, OverflowError):          # tangkap error matematika lain
        return float('inf')                      # kalau error, anggap ga valid juga


# ==============================================================================
# HITUNG FITNESS
# GA kerjanya dengan memaksimalkan fitness, sedangkan kita mau MINIMUMKAN f
# solusinya: fitness = -f -> f makin kecil, fitness makin gede, GA makin suka
# ==============================================================================
def hitung_fitness(x1, x2):
    f = fungsi_objektif(x1, x2)           # hitung nilai fungsinya dulu

    if f == float('inf'):
        return -1e9                        # kalau ga valid, kasih hukuman gede banget

    return -f                             # balik tanda: minimasi jadi maksimasi


# ==============================================================================
# ENKODE — nilai x (desimal) -> kromosom biner
# GA mainnya di level bit, jadi x1 dan x2 harus diubah ke biner dulu
#
# caranya:
#   1. scale x dari [-10, 10] ke integer [0, 2^n - 1]
#   2. ubah integer itu ke list bit
# ==============================================================================
def enkode(x, n_bits):
    # (x - X_MIN) / (X_MAX - X_MIN) -> normalisasi x ke 0.0–1.0
    # dikali (2^n - 1) -> scale ke range integer
    int_val = round((x - X_MIN) / (X_MAX - X_MIN) * (2 ** n_bits - 1))
    int_val = max(0, min(2 ** n_bits - 1, int_val))  # clamp biar ga keluar rentang

    # baca bit ke-i dari int_val satu per satu pake operasi bitwise
    # >> geser kanan, & 1 ambil bit paling kanan (hasilnya 0 atau 1)
    return [(int_val >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


# ==============================================================================
# DEKODE — kromosom biner -> nilai x1 dan x2 asli
# dipake tiap mau ngitung fitness: fungsi objektif butuh nilai desimal, bukan bit
# ==============================================================================
def dekode(kromosom, n_bits):
    def bits_ke_float(bits):
        int_val = 0
        for bit in bits:
            int_val = int_val * 2 + bit  # baca biner -> desimal: geser kiri (*2) + bit berikutnya

        return X_MIN + int_val * (X_MAX - X_MIN) / (2 ** n_bits - 1)  # scale balik ke [-10, 10]

    x1 = bits_ke_float(kromosom[:n_bits])   # 16 bit pertama = x1
    x2 = bits_ke_float(kromosom[n_bits:])   # 16 bit sisanya = x2
    return x1, x2


# ==============================================================================
# INISIALISASI POPULASI — bikin generasi pertama secara acak
# tiap individu = list 32 bit (16 bit x1 + 16 bit x2)
# input user ikut dimasukin ke populasi biar titik itu ikut berevolusi juga
# ==============================================================================
def inisialisasi_populasi(pop_size, n_bits, x1_seed=None, x2_seed=None):
    populasi = []

    for _ in range(pop_size):
        kromosom = [random.randint(0, 1) for _ in range(n_bits * 2)]  # isi 32 bit secara acak
        populasi.append(kromosom)

    if x1_seed is not None and x2_seed is not None:
        kromosom_seed = enkode(x1_seed, n_bits) + enkode(x2_seed, n_bits)  # encode input user ke biner
        populasi[0]   = kromosom_seed                                        # taruh di slot pertama

    return populasi


# ==============================================================================
# SELEKSI ORANGTUA — Tournament Selection
# pilih k individu acak, yang fitness-nya tertinggi di antara k itu yang menang
# lebih adil dari roulette wheel: yang biasa-biasa pun masih bisa kepilih
# -> populasi tetep beragam, ga cepet-cepet konvergen ke satu titik
# ==============================================================================
def seleksi_turnamen(populasi, fitness_list, k=3):
    kandidat_idx = random.sample(range(len(populasi)), k)                  # pilih k indeks acak
    pemenang_idx = max(kandidat_idx, key=lambda i: fitness_list[i])        # ambil yang fitness-nya paling gede
    return populasi[pemenang_idx][:]                                        # return salinannya biar aman


# ==============================================================================
# CROSSOVER — Single-Point Crossover
# potong dua kromosom di titik yang sama (acak), tukar bagian kanannya
#
# contoh (8 bit):
#   P1: [1,0,1,0 | 1,1,0,1]      C1: [1,0,1,0 | 0,0,1,0]  ← kiri P1 + kanan P2
#   P2: [0,1,0,1 | 0,0,1,0]  ->   C2: [0,1,0,1 | 1,1,0,1]  ← kiri P2 + kanan P1
# ==============================================================================
def crossover(parent1, parent2, pc):
    if random.random() < pc:                          # crossover terjadi kalau acakan < pc (80%)
        titik  = random.randint(1, len(parent1) - 1)  # titik potong acak (bukan di ujung)
        child1 = parent1[:titik] + parent2[titik:]    # kiri P1 + kanan P2
        child2 = parent2[:titik] + parent1[titik:]    # kiri P2 + kanan P1
        return child1, child2

    return parent1[:], parent2[:]                     # kalau ga crossover, anak = copy orangtua


# ==============================================================================
# MUTASI — Bit-Flip Mutation
# tiap bit punya peluang kecil (1%) buat ke-flip: 0->1 atau 1->0
# penting biar GA bisa explore area yang belum pernah dijamah crossover
# ==============================================================================
def mutasi(kromosom, pm):
    # kalau random < pm -> flip bit (1 - bit), kalau engga -> biarin
    return [1 - bit if random.random() < pm else bit for bit in kromosom]


# ==============================================================================
# BUAT GENERASI BARU — Generational Replacement + Elitisme
# 1. individu terbaik langsung dilolosin tanpa diutak-atik (elitisme)
#    -> biar solusi terbaik ga ilang gara-gara kena mutasi
# 2. sisa slot diisi offspring dari seleksi -> crossover -> mutasi
# ==============================================================================
def buat_generasi_baru(populasi, fitness_list, pop_size, pc, pm):
    populasi_baru = []

    # elitisme: lolosin si terbaik langsung ke generasi berikutnya
    idx_elite = max(range(len(populasi)), key=lambda i: fitness_list[i])  # cari indeks yang paling bagus
    populasi_baru.append(populasi[idx_elite][:])                          # masukin ke generasi baru (di-copy)

    while len(populasi_baru) < pop_size:              # isi terus sampai populasi penuh
        parent1 = seleksi_turnamen(populasi, fitness_list)  # pilih orangtua 1
        parent2 = seleksi_turnamen(populasi, fitness_list)  # pilih orangtua 2

        child1, child2 = crossover(parent1, parent2, pc)    # kawinkan -> dapet dua anak
        child1 = mutasi(child1, pm)                          # mutasi anak 1
        child2 = mutasi(child2, pm)                          # mutasi anak 2

        populasi_baru.append(child1)                         # masukin anak 1
        if len(populasi_baru) < pop_size:                    # cek slot masih ada ga
            populasi_baru.append(child2)                     # masukin anak 2

    return populasi_baru


# ==============================================================================
# MAIN — alur utama program dari awal sampai akhir
# ==============================================================================
def main():
    print("=" * 55)
    print("   Genetic Algorithm - Minimisasi f(x1, x2)")
    print("=" * 55)
    print("  f(x1,x2) = -(sin(x1)*cos(x2)*tan(x1+x2)")
    print("             + 0.5*exp(1-sqrt(x2^2)))")
    print(f"  Domain   : {X_MIN} <= x1, x2 <= {X_MAX}")
    print("=" * 55)

    # input x1 dari user — nilainya ikut masuk ke populasi awal
    while True:
        try:
            x1_input = float(input(f"\nMasukkan x1 ({X_MIN} s.d. {X_MAX}): "))  # konversi input ke float
            if X_MIN <= x1_input <= X_MAX:  # cek apakah masuk domain
                break
            print(f"  [!] x1 harus dalam rentang [{X_MIN}, {X_MAX}]")
        except ValueError:                  # tangkap kalau user masukin huruf atau karakter aneh
            print("  [!] input ga valid, masukin angka dong")

    # input x2 dari user
    while True:
        try:
            x2_input = float(input(f"Masukkan x2 ({X_MIN} s.d. {X_MAX}): "))
            if X_MIN <= x2_input <= X_MAX:
                break
            print(f"  [!] x2 harus dalam rentang [{X_MIN}, {X_MAX}]")
        except ValueError:
            print("  [!] input ga valid, masukin angka dong")

    print()
    print(f"  Setting GA: PopSize={POP_SIZE}, Generasi={N_GEN}, Pc={PC}, Pm={PM}, Bits/var={N_BITS}")
    print("-" * 55)

    # LANGKAH 1: bikin populasi awal
    populasi = inisialisasi_populasi(POP_SIZE, N_BITS, x1_input, x2_input)

    # variabel buat nyimpen juara sepanjang masa (bukan cuma juara generasi terakhir)
    kromosom_terbaik_global = None
    fitness_terbaik_global  = float('-inf')  # mulai dari -inf biar pasti keupdate di gen pertama
    x1_terbaik = None
    x2_terbaik = None

    # LANGKAH 2: evolusi sebanyak N_GEN generasi
    for gen in range(1, N_GEN + 1):

        # decode tiap kromosom -> hitung fitness -> simpan ke list
        fitness_list = []
        for kromosom in populasi:
            x1, x2 = dekode(kromosom, N_BITS)   # biner -> nilai x1, x2
            fit    = hitung_fitness(x1, x2)      # hitung fitness-nya
            fitness_list.append(fit)

        # cari yang paling bagus di generasi ini
        idx_gen_terbaik  = max(range(len(populasi)), key=lambda i: fitness_list[i])
        fitness_gen_best = fitness_list[idx_gen_terbaik]

        # update rekor kalau generasi ini lebih bagus dari sebelumnya
        if fitness_gen_best > fitness_terbaik_global:
            fitness_terbaik_global  = fitness_gen_best
            kromosom_terbaik_global = populasi[idx_gen_terbaik][:]          # simpan kromosom (di-copy)
            x1_terbaik, x2_terbaik = dekode(kromosom_terbaik_global, N_BITS)  # decode x1 x2-nya

        print(f"Gen {gen:3d} | Best: {fitness_gen_best:.10f}")  # :3d = 3 digit, :.10f = 10 desimal

        # evolusikan ke generasi berikutnya
        populasi = buat_generasi_baru(populasi, fitness_list, POP_SIZE, PC, PM)

    # LANGKAH 3: tampilin hasil akhir
    f_minimum    = fungsi_objektif(x1_terbaik, x2_terbaik)
    kromosom_str = ''.join(map(str, kromosom_terbaik_global))  # list bit -> string "10110..."

    print()
    print("=" * 55)
    print("HASIL TERBAIK:")
    print(f"  Kromosom : {kromosom_str}")
    print(f"  x1       = {x1_terbaik}")
    print(f"  x2       = {x2_terbaik}")
    print(f"  f(x1,x2) = {f_minimum}  <- ini nilai minimumnya")
    print("=" * 55)


if __name__ == "__main__":
    main()  # jalanin main() kalau file ini dieksekusi langsung (bukan di-import)
