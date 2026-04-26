import math, random

# ── Konstanta GA ───────────────────────────────────────────
BITS     = 8
LEN      = BITS * 2
XMIN, XMAX = -10.0, 10.0
POP_SIZE = 6
MAX_GEN  = 5
PC       = 0.8
PM       = 0.05

# Fungsi objektif yang ingin diminimasi 
def f(x1, x2):
    try:
        v = -(math.sin(x1)*math.cos(x2)*math.tan(x1+x2) + 0.5*math.exp(1-abs(x2)))
        return v if math.isfinite(v) else float('inf')
    except: return float('inf')

# 1. INISIALISASI POPULASI
#    x1,x2 input user di-encode jadi kromosom pertama (seed).
#    Sisanya acak agar eksplorasi tetap luas.
def encode(x):
    iv = round((x-XMIN)/(XMAX-XMIN)*(2**BITS-1))
    return [int(b) for b in format(max(0,min(2**BITS-1,iv)), f'0{BITS}b')]

def inisialisasi(x1s, x2s):
    pop = [encode(x1s)+encode(x2s)]
    pop += [[random.randint(0,1) for _ in range(LEN)] for _ in range(POP_SIZE-1)]
    return pop

# 2. DEKODE KROMOSOM
#    Bit list → nilai riil x1, x2 di domain [XMIN, XMAX]
def dekode(kr):
    to_x = lambda b: XMIN + int(''.join(map(str,b)),2)*(XMAX-XMIN)/(2**BITS-1)
    return to_x(kr[:BITS]), to_x(kr[BITS:])

def kr_str(kr): return ''.join(map(str,kr))

# 3. PERHITUNGAN FITNESS
#    fitness = -f lalu digeser positif (syarat roulette wheel)
def hitung_fitness(pop):
    raw = [-f(*dekode(kr)) if math.isfinite(f(*dekode(kr))) else -1e9 for kr in pop]
    mn  = min(raw)
    return [r - mn + 1 for r in raw]   # geser agar semua > 0


# 4. SELEKSI — Roulette Wheel
#    P[i] = fitness[i] / total, spin random, pilih berdasar interval
def seleksi_roulette(pop, fit):
    total = sum(fit)
    r, c  = random.random(), 0
    for i, fi in enumerate(fit):
        c += fi / total
        if r <= c: return pop[i]
    return pop[-1]


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

# 7. PERGANTIAN GENERASI
#    Seleksi → crossover → mutasi → populasi baru
def generasi_baru(pop, fit):
    anak = []
    while len(anak) < POP_SIZE:
        c1, c2 = crossover(seleksi_roulette(pop,fit), seleksi_roulette(pop,fit))
        anak += [mutasi(c1), mutasi(c2)]
    return anak[:POP_SIZE]

# MAIN 
if __name__ == "__main__":
    print("=== GA — Minimasi f(x1, x2) ===")
    x1_in = float(input("Masukkan x1: "))
    x2_in = float(input("Masukkan x2: "))

    pop = inisialisasi(x1_in, x2_in)        # 1. inisialisasi
    best_kr, best_f = None, float('inf')

    for gen in range(1, MAX_GEN+1):
        print(f"\nGENERASI {gen}")
        print("-" * 55)

        fit = hitung_fitness(pop)            # 2 & 3. dekode + hitung fitness

        for kr in pop:
            x1, x2 = dekode(kr)
            fo = f(x1, x2)
            print(f"  kromosom: {kr_str(kr)}  x1: {x1:6.2f}  x2: {x2:6.2f}  fitness: {fo:.4f}")
            if fo < best_f:                  # update solusi terbaik global
                best_f, best_kr = fo, kr[:]

        pop = generasi_baru(pop, fit)        # 4,5,6,7. seleksi, crossover, mutasi, ganti generasi

    print(f"\n{'='*55}")
    print("HASIL AKHIR")
    print(f"{'='*55}")
    x1_out, x2_out = dekode(best_kr)
    print(f"  kromosom terbaik : {kr_str(best_kr)}")
    print(f"  x1               : {x1_out:.4f}")
    print(f"  x2               : {x2_out:.4f}")
    print(f"  fitness          : {best_f:.4f}")