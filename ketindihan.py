import math
import random

# =============================================================================
# PROGRAM: Genetic Algorithm untuk cari nilai MINIMUM dari fungsi f(x1, x2)
#
# Fungsinya:
#   f(x1, x2) = -(sin(x1) * cos(x2) * tan(x1+x2) + 0.5 * exp(1 - sqrt(x2^2)))
#
# Batasannya (domain):
#   -10 <= x1 <= 10  dan  -10 <= x2 <= 10
#
# Cara kerja GA secara garis besar:
#   1. Bikin banyak "kandidat solusi" secara acak (populasi)
#   2. Nilai masing-masing kandidat pakai fungsi di atas (fitness)
#   3. Yang nilainya bagus lebih sering dipilih jadi "orangtua"
#   4. Orangtua "kawin" → anaknya mewarisi gabungan sifat keduanya
#   5. Kadang ada sedikit perubahan acak (mutasi) biar gak stuck di satu titik
#   6. Ulangi proses ini sebanyak N generasi, simpan yang terbaik
# =============================================================================


# =============================================================================
# PARAMETER GA
# Ini adalah "setting" utama yang menentukan bagaimana GA bekerja.
# Kalau mau eksperimen, ubah nilai-nilai di sini.
# =============================================================================

POP_SIZE = 100   # Jumlah kandidat solusi dalam satu generasi.
                 # Makin besar → makin banyak variasi, tapi makin lambat.

N_BITS = 16      # Setiap variabel (x1 dan x2) direpresentasikan dalam 16 bit biner.
                 # 16 bit = 2^16 = 65536 kemungkinan nilai antara -10 s.d. 10.
                 # Presisi yang dihasilkan sekitar 0.0003 — cukup halus.

N_GEN = 100      # GA akan berevolusi sebanyak 100 generasi lalu berhenti.
                 # Ini adalah kriteria penghentian (stopping criteria) kita.

PC = 0.80        # Probabilitas crossover = 80%.
                 # Dari setiap pasang orangtua, 80% kemungkinan mereka
                 # benar-benar menghasilkan anak baru hasil gabungan.
                 # 20% sisanya anaknya ya salinan orangtua aja.

PM = 0.01        # Probabilitas mutasi per bit = 1%.
                 # Setiap bit dalam kromosom punya peluang 1% untuk dibalik.
                 # Fungsinya: supaya GA bisa "kabur" dari titik yang tidak optimal.

X_MIN = -10.0    # Batas bawah domain x1 dan x2
X_MAX =  10.0    # Batas atas domain x1 dan x2


# =============================================================================
# FUNGSI OBJEKTIF — ini fungsi yang ingin kita cari nilai minimumnya
# =============================================================================
def fungsi_objektif(x1, x2):
    # Kita masukkan nilai x1 dan x2, lalu hitung f(x1, x2)
    # Ada potensi masalah: fungsi tan() bisa mendekati tak terhingga
    # (misalnya saat x1+x2 mendekati π/2, 3π/2, dst.)
    # Kalau itu terjadi, kita anggap nilai fungsinya "tidak valid"
    try:
        tan_val = math.tan(x1 + x2)

        # Kalau tan-nya meledak (lebih dari sejuta), skip titik ini
        # — anggap titik ini tidak layak jadi kandidat solusi
        if abs(tan_val) > 1e6:
            return float('inf')  # 'inf' = "buruk banget, jangan pilih ini"

        exp_val = math.exp(1 - math.sqrt(x2 ** 2))
        hasil = -(math.sin(x1) * math.cos(x2) * tan_val + 0.5 * exp_val)
        return hasil

    except (ValueError, OverflowError):
        # Kalau ada error matematika lain (misal overflow), skip juga
        return float('inf')


# =============================================================================
# HITUNG FITNESS
#
# GA bekerja dengan cara "memaksimalkan" sesuatu (fitness).
# Masalahnya, kita mau MINIMUMKAN f. Jadi kita balik tandanya:
#   fitness = -f(x1, x2)
#
# Semakin kecil f  →  semakin besar fitness  →  semakin "bagus" di mata GA
# =============================================================================
def hitung_fitness(x1, x2):
    f = fungsi_objektif(x1, x2)

    # Kalau fungsinya tidak valid (inf), kasih hukuman: fitness sangat kecil
    # supaya individu ini tidak akan pernah terpilih jadi orangtua
    if f == float('inf'):
        return -1e9

    return -f  # balik tanda supaya "minimasi" jadi "maksimasi"


# =============================================================================
# ENKODE — mengubah nilai x (desimal) jadi kromosom biner
#
# Kenapa perlu diubah ke biner?
# Karena GA bekerja di level bit: crossover dan mutasi dilakukan di bit-bit ini.
#
# Caranya:
#   1. Petakan x dari range [X_MIN, X_MAX] ke bilangan bulat [0, 2^n - 1]
#   2. Ubah bilangan bulat itu ke representasi biner sepanjang n bit
# =============================================================================
def enkode(x, n_bits):
    # Petakan nilai x ke bilangan bulat antara 0 sampai (2^n - 1)
    int_val = round((x - X_MIN) / (X_MAX - X_MIN) * (2 ** n_bits - 1))

    # Pastikan tidak keluar dari rentang valid (jaga-jaga dari pembulatan)
    int_val = max(0, min(2 ** n_bits - 1, int_val))

    # Ubah bilangan bulat itu ke list bit [0, 1, 1, 0, ...] sepanjang n_bits
    return [(int_val >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


# =============================================================================
# DEKODE — kebalikan dari enkode: ubah kromosom biner balik ke nilai x1 dan x2
#
# Ini dipakai setiap kita mau hitung fitness:
# kita perlu tahu x1 dan x2 aslinya sebelum bisa dimasukkan ke fungsi objektif.
# =============================================================================
def dekode(kromosom, n_bits):
    def bits_ke_float(bits):
        # Ubah list bit ke bilangan bulat dulu (misal [1,0,1,1] → 11)
        int_val = 0
        for bit in bits:
            int_val = int_val * 2 + bit

        # Lalu petakan bilangan bulat itu balik ke range [X_MIN, X_MAX]
        return X_MIN + int_val * (X_MAX - X_MIN) / (2 ** n_bits - 1)

    # Kromosom kita = 32 bit: 16 bit pertama untuk x1, 16 bit berikutnya untuk x2
    x1 = bits_ke_float(kromosom[:n_bits])
    x2 = bits_ke_float(kromosom[n_bits:])
    return x1, x2


# =============================================================================
# INISIALISASI POPULASI — bikin "generasi pertama" secara acak
#
# Setiap individu (kromosom) adalah list of bit sepanjang 32 bit
# (16 bit untuk x1 + 16 bit untuk x2).
#
# Nilai awal dari user (x1_seed, x2_seed) dimasukkan ke populasi sebagai
# salah satu individu — supaya titik awal yang dipilih user ikut berevolusi juga.
# =============================================================================
def inisialisasi_populasi(pop_size, n_bits, x1_seed=None, x2_seed=None):
    populasi = []

    for _ in range(pop_size):
        # Tiap kromosom diisi bit acak 0 atau 1 sepanjang 32 bit
        kromosom = [random.randint(0, 1) for _ in range(n_bits * 2)]
        populasi.append(kromosom)

    # Kalau user kasih nilai awal, encode dan taruh di slot pertama populasi
    # biar titik awal itu juga ikut berevolusi, bukan cuman diabaikan
    if x1_seed is not None and x2_seed is not None:
        kromosom_seed = enkode(x1_seed, n_bits) + enkode(x2_seed, n_bits)
        populasi[0] = kromosom_seed

    return populasi


# =============================================================================
# SELEKSI ORANGTUA — Tournament Selection
#
# Cara kerjanya:
#   Pilih k individu secara acak dari populasi (kayak bikin "mini kompetisi")
#   Yang fitnessnya paling tinggi di antara k itu yang jadi "orangtua"
#
# Kenapa pakai tournament selection?
#   Karena lebih adil dari roulette wheel: individu bagus punya peluang menang,
#   tapi individu yang biasa-biasa juga masih bisa lolos kadang-kadang.
#   Ini bantu jaga keberagaman populasi supaya tidak cepat konvergen.
# =============================================================================
def seleksi_turnamen(populasi, fitness_list, k=3):
    # Pilih k indeks acak dari populasi (tanpa pengulangan)
    kandidat_idx = random.sample(range(len(populasi)), k)

    # Yang fitnessnya paling gede di antara k kandidat itu yang menang
    pemenang_idx = max(kandidat_idx, key=lambda i: fitness_list[i])

    # Return salinannya (bukan referensi langsung) supaya aman saat dimodifikasi
    return populasi[pemenang_idx][:]


# =============================================================================
# CROSSOVER — Single-Point Crossover
#
# Bayangkan dua orangtua punya "tali DNA" masing-masing sepanjang 32 bit.
# Kita potong keduanya di titik yang sama (acak), lalu tukar ujungnya.
# Hasilnya: dua anak yang masing-masing punya campuran dari kedua orangtua.
#
# Contoh (diperpendek jadi 8 bit):
#   Parent1: [1, 0, 1, 0 | 1, 1, 0, 1]
#   Parent2: [0, 1, 0, 1 | 0, 0, 1, 0]
#   Titik potong di posisi 4 →
#   Child1:  [1, 0, 1, 0 | 0, 0, 1, 0]   ← kiri dari P1, kanan dari P2
#   Child2:  [0, 1, 0, 1 | 1, 1, 0, 1]   ← kiri dari P2, kanan dari P1
# =============================================================================
def crossover(parent1, parent2, pc):
    # Cek dulu apakah crossover terjadi (berdasarkan probabilitas pc)
    if random.random() < pc:
        # Pilih titik potong secara acak (bukan di ujung, supaya ada pertukaran)
        titik = random.randint(1, len(parent1) - 1)

        # Tukar bagian kanan dari masing-masing parent
        child1 = parent1[:titik] + parent2[titik:]
        child2 = parent2[:titik] + parent1[titik:]
        return child1, child2

    # Kalau tidak terjadi crossover (20% kemungkinan), anak = salinan orangtua
    return parent1[:], parent2[:]


# =============================================================================
# MUTASI — Bit-Flip Mutation
#
# Setelah crossover, setiap bit di kromosom anak punya peluang kecil (PM = 1%)
# untuk dibalik: 0 jadi 1, atau 1 jadi 0.
#
# Kenapa perlu mutasi?
#   Kalau cuma crossover, lama-lama semua individu akan mirip-mirip
#   karena terus menurun dari nenek moyang yang sama.
#   Mutasi memperkenalkan keacakan kecil → GA bisa explore area baru
#   yang belum pernah dicoba sama sekali sebelumnya.
# =============================================================================
def mutasi(kromosom, pm):
    # Untuk setiap bit: lempar dadu, kalau < pm → balik bitnya
    return [1 - bit if random.random() < pm else bit for bit in kromosom]


# =============================================================================
# BUAT GENERASI BARU — Pergantian Generasi dengan Elitisme
#
# Setelah semua individu dinilai, kita bikin generasi berikutnya:
#   1. Individu terbaik langsung dipertahankan tanpa perubahan (elitisme)
#      → Tujuannya: pastikan solusi terbaik yang udah ditemukan tidak hilang
#        kena mutasi atau tersisih di generasi berikutnya
#   2. Sisa slot populasi diisi dengan offspring dari proses seleksi-crossover-mutasi
# =============================================================================
def buat_generasi_baru(populasi, fitness_list, pop_size, pc, pm):
    populasi_baru = []

    # --- Elitisme: langsung pertahankan individu dengan fitness tertinggi ---
    # Cari dulu siapa yang paling bagus di generasi sekarang
    idx_elite = max(range(len(populasi)), key=lambda i: fitness_list[i])
    # Masukkan dia ke generasi baru persis apa adanya (tanpa crossover/mutasi)
    populasi_baru.append(populasi[idx_elite][:])

    # --- Isi sisa populasi dengan anak-anak hasil evolusi ---
    while len(populasi_baru) < pop_size:
        # Pilih dua orangtua lewat tournament selection
        parent1 = seleksi_turnamen(populasi, fitness_list)
        parent2 = seleksi_turnamen(populasi, fitness_list)

        # Kawinkan keduanya → dapat dua anak hasil kombinasi gen keduanya
        child1, child2 = crossover(parent1, parent2, pc)

        # Kasih sedikit mutasi ke masing-masing anak
        child1 = mutasi(child1, pm)
        child2 = mutasi(child2, pm)

        # Masukkan ke generasi baru
        populasi_baru.append(child1)
        if len(populasi_baru) < pop_size:  # cek dulu supaya tidak kelebihan slot
            populasi_baru.append(child2)

    return populasi_baru


# =============================================================================
# MAIN — Di sinilah semuanya dijalankan urut dari awal sampai akhir
# =============================================================================
def main():
    print("=" * 55)
    print("   Genetic Algorithm - Minimisasi f(x1, x2)")
    print("=" * 55)
    print("  f(x1,x2) = -(sin(x1)*cos(x2)*tan(x1+x2)")
    print("             + 0.5*exp(1-sqrt(x2^2)))")
    print(f"  Domain   : {X_MIN} <= x1, x2 <= {X_MAX}")
    print("=" * 55)

    # --- Minta input x1 dari user ---
    # x1 ini nanti akan dimasukkan ke populasi awal sebagai salah satu kandidat
    while True:
        try:
            x1_input = float(input(f"\nMasukkan x1 ({X_MIN} s.d. {X_MAX}): "))
            if X_MIN <= x1_input <= X_MAX:
                break
            print(f"  [!] x1 harus dalam rentang [{X_MIN}, {X_MAX}]")
        except ValueError:
            print("  [!] Input tidak valid, masukkan angka.")

    # --- Minta input x2 dari user ---
    while True:
        try:
            x2_input = float(input(f"Masukkan x2 ({X_MIN} s.d. {X_MAX}): "))
            if X_MIN <= x2_input <= X_MAX:
                break
            print(f"  [!] x2 harus dalam rentang [{X_MIN}, {X_MAX}]")
        except ValueError:
            print("  [!] Input tidak valid, masukkan angka.")

    print()
    print(f"  Setting GA: PopSize={POP_SIZE}, Generasi={N_GEN}, Pc={PC}, Pm={PM}, Bits/var={N_BITS}")
    print("-" * 55)

    # --- LANGKAH 1: Buat populasi awal ---
    # 100 kromosom acak + kromosom dari input user dimasukkan di slot pertama
    populasi = inisialisasi_populasi(POP_SIZE, N_BITS, x1_input, x2_input)

    # Siapkan variabel untuk nyimpen kromosom terbaik yang pernah ditemukan
    # (bukan cuma terbaik di generasi terakhir, tapi terbaik sepanjang 100 generasi)
    kromosom_terbaik_global = None
    fitness_terbaik_global  = float('-inf')
    x1_terbaik = None
    x2_terbaik = None

    # --- LANGKAH 2: Loop evolusi sebanyak N_GEN generasi ---
    for gen in range(1, N_GEN + 1):

        # Nilai semua individu di generasi ini:
        # decode dulu kromosom ke (x1, x2), baru hitung fitnessnya
        fitness_list = []
        for kromosom in populasi:
            x1, x2 = dekode(kromosom, N_BITS)
            fit    = hitung_fitness(x1, x2)
            fitness_list.append(fit)

        # Cari individu dengan fitness tertinggi di generasi ini
        idx_gen_terbaik  = max(range(len(populasi)), key=lambda i: fitness_list[i])
        fitness_gen_best = fitness_list[idx_gen_terbaik]

        # Kalau individu terbaik generasi ini lebih baik dari yang pernah ada,
        # simpan sebagai "juara sementara" — nanti dia yang kita tampilkan di akhir
        if fitness_gen_best > fitness_terbaik_global:
            fitness_terbaik_global  = fitness_gen_best
            kromosom_terbaik_global = populasi[idx_gen_terbaik][:]
            x1_terbaik, x2_terbaik = dekode(kromosom_terbaik_global, N_BITS)

        # Tampilkan perkembangan fitness terbaik tiap generasi
        print(f"Gen {gen:3d} | Best: {fitness_gen_best:.10f}")

        # Evolusikan ke generasi berikutnya
        populasi = buat_generasi_baru(populasi, fitness_list, POP_SIZE, PC, PM)

    # --- LANGKAH 3: Tampilkan hasil akhir ---
    # Decode kromosom terbaik yang pernah ditemukan selama 100 generasi
    f_minimum    = fungsi_objektif(x1_terbaik, x2_terbaik)
    kromosom_str = ''.join(map(str, kromosom_terbaik_global))

    print()
    print("=" * 55)
    print("HASIL TERBAIK:")
    print(f"  Kromosom : {kromosom_str}")
    print(f"  x1       = {x1_terbaik}")
    print(f"  x2       = {x2_terbaik}")
    print(f"  f(x1,x2) = {f_minimum}  <- nilai minimum yang ditemukan")
    print("=" * 55)


if __name__ == "__main__":
    main()