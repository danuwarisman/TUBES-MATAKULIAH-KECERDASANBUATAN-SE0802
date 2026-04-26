import math, random

# KONSTANTA GA ( YG GA PERLU DI INPUT)
BITS = 16
LEN  = BITS * 2
XMIN, XMAX = -10.0, 10.0
POP_SIZE = 100      # ukuran populasi
MAX_GEN  = 50       # jumlah generasi
PC       = 0.8      # probabilitas crossover
PM       = 0.01     # probabilitas mutasi
# FUNGSI OBJEKTIF YANG INGIN DIMINIMASI: f(x1,x2) = sin(x1)*cos(x2)*tan(x1+x2) + 0.5*exp(1-abs(x2))
def f(x1, x2):
    try:
        v = -(math.sin(x1)*math.cos(x2)*math.tan(x1+x2) + 0.5*math.exp(1-abs(x2)))
        return v if math.isfinite(v) else float('inf')
    except: return float('inf')

 
# 1. INISIALISASI POPULASI
#    Kromosom pertama = encode dari x1,x2 input user (seed).
#    Sisanya acak, agar eksplorasi tetap luas.
 
def encode_x(x):
    # Ubah nilai riil x → integer → bit list
    int_val = round((x - XMIN) / (XMAX - XMIN) * (2**BITS - 1))
    int_val = max(0, min(2**BITS - 1, int_val))   # clamp agar tidak keluar range
    return [int(b) for b in format(int_val, f'0{BITS}b')]

def inisialisasi(x1_seed, x2_seed):
    
    seed = encode_x(x1_seed) + encode_x(x2_seed)  # individu pertama dari input user
    pop  = [seed]
    pop += [[random.randint(0,1) for _ in range(LEN)] for _ in range(POP_SIZE - 1)]
    return pop

 # 2. DEKODE KROMOSOM
#    Bit list → angka riil x1, x2
 
def dekode(kr):
    to_x = lambda b: XMIN + int(''.join(map(str,b)),2) * (XMAX-XMIN) / (2**BITS-1)
    return to_x(kr[:BITS]), to_x(kr[BITS:])

 
# 3. PERHITUNGAN FITNESS
#    fitness = -f  →  f kecil berarti fitness besar (GA memaksimalkan)
 
def fitness(kr):
    v = f(*dekode(kr))
    return -v if math.isfinite(v) else float('-inf')

 
# 4. PEMILIHAN ORANGTUA — Tournament Selection
#    Adu 3 individu acak, pemenang = fitness tertinggi
 
def seleksi(pop, fit):
    k = random.sample(range(len(pop)), 3)
    return pop[max(k, key=lambda i: fit[i])]

    
# 5. CROSSOVER — Single-Point
#    Tukar gen di titik potong acak dengan probabilitas PC
 
def crossover(p1, p2):
    if random.random() < PC:
        t = random.randint(1, LEN-1)
        return p1[:t]+p2[t:], p2[:t]+p1[t:]
    return p1[:], p2[:]

 
# 6. MUTASI — Bit-Flip
#    Tiap bit bisa terbalik (0↔1) dengan probabilitas PM
 
def mutasi(kr):
    return [1-b if random.random()<PM else b for b in kr]

 
# 7. PERGANTIAN GENERASI — Generational + Elitism
#    1 individu terbaik langsung lolos, sisanya dari crossover+mutasi
 
def generasi_baru(pop, fit):
    elit = pop[max(range(len(pop)), key=lambda i: fit[i])][:]
    anak = [elit]
    while len(anak) < POP_SIZE:
        c1, c2 = crossover(seleksi(pop,fit), seleksi(pop,fit))
        anak += [mutasi(c1), mutasi(c2)]
    return anak[:POP_SIZE]

# MAIN PROGRAM
if __name__ == "__main__":
    print("=== GA — Minimasi f(x1, x2) ===\n")
    x1_in = float(input("Masukkan x1: "))
    x2_in = float(input("Masukkan x2: "))
    print()

    pop = inisialisasi(x1_in, x2_in)              # 1. inisialisasi

    best_kr = None
    best_f  = float('inf')

    for g in range(1, MAX_GEN + 1):
        fit = [fitness(k) for k in pop]           # 2 & 3. dekode + hitung fitness
        pop = generasi_baru(pop, fit)             # 4,5,6,7. seleksi, crossover, mutasi, ganti generasi

        # Cari terbaik generasi ini
        idx = max(range(len(fit)), key=lambda i: fit[i])
        x1g, x2g = dekode(pop[idx])
        fg = f(x1g, x2g)

        print(f"Gen {g:3d} | Best: {fg}")

        if fg < best_f:                           # update solusi terbaik global
            best_f, best_kr = fg, pop[idx][:]

    # Output akhir
    x1_out, x2_out = dekode(best_kr)
    print(f"\nHASIL TERBAIK:")
    print(f"x1      = {x1_out}")
    print(f"x2      = {x2_out}")
    print(f"fitness = {best_f}")