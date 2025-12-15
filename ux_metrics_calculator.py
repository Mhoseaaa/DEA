"""
UX METRICS CALCULATOR - Organized by PDF Material
Program untuk menghitung rumus-rumus UX Research berdasarkan materi kuliah

Materi yang dicakup:
- Materi 8: Quantifying User Research (08_Quantifying_User_Research2.pdf)
- Materi 9: Sample Size & T-Score
- Materi 10: Performance Based Metrics
- Materi 11: Issues Based Behavioral and Physiological Metrics
- Materi 12: Self-Reported Metrics, Combined and Comparative Metrics
"""

import math
from scipy import stats
import numpy as np

def print_header(title, pdf_ref=""):
    print("\n" + "="*60)
    print(f"  {title}")
    if pdf_ref:
        print(f"  📄 {pdf_ref}")
    print("="*60)

def print_formula(formula, page):
    print(f"\n📖 Rumus (Halaman {page}):")
    print(f"   {formula}")
    print("-"*50)

# ==============================================================================
#                           MATERI 8: QUANTIFYING USER RESEARCH
#                       (08_Quantifying_User_Research2.pdf)
# ==============================================================================

def materi_8_menu():
    print_header("MATERI 8: QUANTIFYING USER RESEARCH", 
                 "08_Quantifying_User_Research2.pdf")
    print("""
COMPLETION RATE COMPARISON:
  1. Small-Sample Test (Binomial)           [Hal. 4]
  2. Mid-Probability Test                   [Hal. 6]
  3. Large-Sample Test (Z-test)             [Hal. 7]

SATISFACTION SCORE:
  4. Satisfaction Score vs Benchmark        [Hal. 9]
  5. Mengubah Rating Kontinyu ke Diskret    [Hal. 12]

WITHIN-SUBJECTS COMPARISON:
  6. Paired t-test                          [Hal. 15-16]
  7. Two-sample t-test                      [Hal. 16]

BETWEEN-SUBJECTS COMPARISON:
  8. Chi-square Test of Independence        [Hal. 20]
  9. Fisher Exact Test                      [Hal. 22]
  10. McNemar Exact Test                    [Hal. 23]

  0. Kembali ke Menu Utama
""")
    return input("Pilihan Anda: ")

# 1. Small-Sample Test (Binomial) - Halaman 4
def small_sample_test():
    print_header("SMALL-SAMPLE TEST (Binomial)", "Materi 8 - Halaman 4")
    print_formula("P(X ≥ x) = Σ C(n,k) × p^k × (1-p)^(n-k)", "4")
    
    print("Digunakan untuk membandingkan completion rate dengan benchmark")
    print("ketika sample size kecil (n < 12)")
    print()
    
    n = int(input("Masukkan jumlah total task (n): "))
    x = int(input("Masukkan jumlah task berhasil (x): "))
    p0_input = float(input("Masukkan benchmark proportion (p0, misal 0.7 atau 70): "))
    p0 = p0_input / 100 if p0_input > 1 else p0_input
    
    # Calculate binomial probability P(X >= x)
    p_value = 1 - stats.binom.cdf(x - 1, n, p0)
    
    # Also calculate exact probabilities for each outcome
    print(f"\n{'='*50}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*50}")
    print(f"Sample size (n): {n}")
    print(f"Successes (x): {x}")
    print(f"Observed proportion: {x/n:.4f} ({x/n*100:.1f}%)")
    print(f"Benchmark (p0): {p0} ({p0*100:.0f}%)")
    print(f"\nP-value (one-tailed, X ≥ {x}): {p_value:.4f}")
    
    confidence = (1 - p_value) * 100
    print(f"\n=== KESIMPULAN ===")
    observed = x / n
    if observed >= p0:
        print(f"Kemungkinan completion rate lebih besar dari {p0*100:.0f}% sebesar {confidence:.1f}% (100%-{p_value*100:.1f}%)")
    else:
        print(f"Kemungkinan completion rate lebih kecil dari {p0*100:.0f}% sebesar {p_value*100:.1f}%")

# 2. Mid-Probability Test - Halaman 6
def mid_probability_test():
    print_header("MID-PROBABILITY TEST", "Materi 8 - Halaman 6")
    print_formula("Mid-P = ½×P(X=x) + P(X>x)", "6")
    
    print("Mid-probabilitas menambahkan setengah probabilitas")
    print("suatu keberhasilan (lebih akurat dari exact probability)")
    print()
    
    n = int(input("Masukkan jumlah total responden (n): "))
    x = int(input("Masukkan jumlah responden berhasil (x): "))
    p0_input = float(input("Masukkan benchmark proportion (p0, misal 0.7 atau 70): "))
    p0 = p0_input / 100 if p0_input > 1 else p0_input
    
    # P(X = x) - probability of exactly x successes
    p_exact_x = stats.binom.pmf(x, n, p0)
    
    # P(X > x) - probability of more than x successes
    p_greater = 1 - stats.binom.cdf(x, n, p0)
    
    # Mid-probability = ½×P(X=x) + P(X>x)
    half_p_exact = 0.5 * p_exact_x
    mid_p = half_p_exact + p_greater
    
    # Confidence = 100% - mid_p
    confidence = (1 - mid_p) * 100
    
    print(f"\n{'='*50}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*50}")
    print(f"\nKemungkinan mendapatkan tepat {x} responden berhasil =")
    print(f"½ × BINOM.DIST({x},{n},{p0},FALSE) = ½ × ({p_exact_x:.5f}) = {half_p_exact:.5f}")
    
    print(f"\nKemungkinan mendapatkan tepat {n} responden berhasil =")
    print(f"BINOM.DIST({n},{n},{p0},FALSE) = {stats.binom.pmf(n, n, p0):.5f}")
    
    print(f"\nKemungkinan mendapatkan {x} atau {n} responden berhasil")
    print(f"= {half_p_exact:.5f} + {p_greater:.5f} = {mid_p:.4f}")
    
    print(f"\n=== KESIMPULAN ===")
    observed = x / n
    if observed >= p0:
        print(f"→ Kemungkinan completion rate lebih besar dari {p0*100:.0f}% sebesar {confidence:.1f}% (100%-{mid_p*100:.1f}%)")
    else:
        print(f"→ Kemungkinan completion rate lebih kecil dari {p0*100:.0f}% sebesar {mid_p*100:.1f}%")

# 3. Large-Sample Test (Z-test) - Halaman 7
def large_sample_test():
    print_header("LARGE-SAMPLE TEST (Z-test)", "Materi 8 - Halaman 7")
    print_formula("Z = (p̂ - p0) / √(p0(1-p0)/n)", "7")
    
    print("Digunakan untuk sample size besar (n ≥ 12)")
    print("Menggunakan normal approximation")
    print()
    
    n = int(input("Masukkan jumlah total responden (n): "))
    x = int(input("Masukkan jumlah responden berhasil (x): "))
    p0_input = float(input("Masukkan benchmark proportion (p0, misal 0.75 atau 75): "))
    
    # Auto-convert percentage to proportion
    p0 = p0_input / 100 if p0_input > 1 else p0_input
    
    p_hat = x / n
    
    # Z-score calculation
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / se
    
    # NORM.S.DIST(z, TRUE) - cumulative distribution
    probability = stats.norm.cdf(z) * 100
    
    print(f"\n{'='*50}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*50}")
    print(f"Observed proportion (p̂): {p_hat:.4f} ({p_hat*100:.1f}%)")
    print(f"Benchmark (p0): {p0:.4f} ({p0*100:.0f}%)")
    print(f"Standard Error: {se:.4f}")
    print(f"Z-score: {z:.4f}")
    print(f"\nNORM.S.DIST({z:.3f}, TRUE) = {stats.norm.cdf(z):.4f}")
    
    print(f"\n=== KESIMPULAN ===")
    print(f"→ {probability:.2f}% probabilitas bahwa setidaknya {p0*100:.0f}% responden")
    print(f"  dapat menyelesaikan tugas")

# 4. Satisfaction Score vs Benchmark - Halaman 9
def satisfaction_vs_benchmark():
    print_header("SATISFACTION SCORE vs BENCHMARK", "Materi 8 - Halaman 9")
    print_formula("t = (X̄ - μ) / (s / √n)", "9")
    
    print("One-sample t-test untuk membandingkan mean satisfaction")
    print("dengan benchmark yang diketahui")
    print()
    
    print("Pilih metode input:")
    print("1. Masukkan summary statistics (mean, std dev, n)")
    print("2. Masukkan raw scores")
    method = input("Pilihan (1/2): ")
    
    if method == "2":
        print("\nMasukkan satisfaction scores (pisahkan dengan koma):")
        scores_str = input("Scores: ")
        scores = [float(x.strip()) for x in scores_str.split(",")]
        n = len(scores)
        x_bar = sum(scores) / n
        s = math.sqrt(sum((x - x_bar)**2 for x in scores) / (n - 1))
    else:
        n = int(input("Masukkan jumlah responden (n): "))
        x_bar = float(input("Masukkan rata-rata score (X̄): "))
        s = float(input("Masukkan standar deviasi (s): "))
    
    benchmark = float(input("Masukkan benchmark/standar industri (μ): "))
    
    se = s / math.sqrt(n)
    t_stat = (x_bar - benchmark) / se
    df = n - 1
    
    # T.DIST(t, df, TRUE) - cumulative distribution
    t_dist_value = stats.t.cdf(t_stat, df)
    probability = t_dist_value * 100
    
    print(f"\n{'='*50}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*50}")
    print(f"Sample size (n): {n}")
    print(f"Sample mean (X̄): {x_bar}")
    print(f"Standard deviation (s): {s}")
    print(f"Standard Error (SE): {se}")
    print(f"Benchmark (μ): {benchmark}")
    print(f"Degrees of freedom (df): {df}")
    print(f"\nt = (X̄ - μ) / (s/√n)")
    print(f"t = ({x_bar} - {benchmark}) / ({s}/√{n})")
    print(f"t = {x_bar - benchmark} / {se} = {t_stat}")
    print(f"\nT.DIST({t_stat}, {df}, TRUE) = {t_dist_value}")
    
    print(f"\n=== KESIMPULAN ===")
    # Truncate to 2 decimal places without rounding
    prob_truncated = math.trunc(probability * 100) / 100
    if x_bar >= benchmark:
        print(f"→ {prob_truncated}% probabilitas bahwa produk memiliki")
        print(f"  skor kepuasan yang lebih baik dibanding standar industri ({benchmark})")
    else:
        lower_prob_truncated = math.trunc((100 - probability) * 100) / 100
        print(f"→ Mean satisfaction ({x_bar}) lebih rendah dari benchmark ({benchmark})")
        print(f"→ {lower_prob_truncated}% probabilitas bahwa skor kepuasan")
        print(f"  lebih rendah dari standar industri")

# 5. Mengubah Rating Kontinyu ke Diskret - Halaman 12
def continuous_to_discrete():
    print_header("MENGUBAH RATING KONTINYU KE DISKRET", "Materi 8 - Halaman 12")
    print_formula("Konversi rating ke binary (1/0) lalu hitung dengan Binomial", "12")
    
    print("Mengkonversi rating kontinyu menjadi diskret (binary)")
    print("kemudian menghitung probabilitas dengan distribusi binomial")
    print()
    
    # Tanyakan jumlah pengguna terlebih dahulu
    n = int(input("Masukkan jumlah pengguna: "))
    scale = int(input("Skala yang digunakan (5 atau 7): "))
    
    print(f"\nPilih metode input rating:")
    print("1. Input rating satu per satu")
    print("2. Input semua rating sekaligus (pisahkan dengan koma)")
    method = input("Pilihan (1/2): ")
    
    ratings = []
    if method == "1":
        print(f"\nMasukkan rating untuk setiap pengguna (1-{scale}):")
        for i in range(1, n + 1):
            rating = int(input(f"  Pengguna {i}: "))
            ratings.append(rating)
    else:
        print(f"\nMasukkan semua rating (pisahkan dengan koma):")
        ratings_str = input("Ratings: ")
        ratings = [int(x.strip()) for x in ratings_str.split(",")]
        if len(ratings) != n:
            print(f"Warning: Jumlah rating ({len(ratings)}) tidak sama dengan jumlah pengguna ({n})")
            n = len(ratings)
    
    # Tanyakan threshold untuk Top-2-Box
    print(f"\nMasukkan threshold untuk 'setuju' (contoh: rating >= 4 dianggap setuju)")
    threshold = int(input(f"Rating minimum untuk dianggap setuju (1-{scale}): "))
    
    # Benchmark proportion
    p0_input = float(input("Masukkan benchmark proportion (misal 0.75 atau 75): "))
    p0 = p0_input / 100 if p0_input > 1 else p0_input
    
    # Konversi ke binary
    binary_conversion = [1 if r >= threshold else 0 for r in ratings]
    successes = sum(binary_conversion)
    
    print(f"\n{'='*50}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*50}")
    
    # Tampilkan rating asli
    print(f"\nHasil rating: {', '.join(map(str, ratings))}")
    
    # Tampilkan konversi binary
    print(f"Hasil konversi (rating >= {threshold} = 1): {', '.join(map(str, binary_conversion))}")
    
    print(f"\nTotal pengguna (n): {n}")
    print(f"Jumlah yang setuju (x): {successes}")
    print(f"Proportion observed: {successes}/{n} = {successes/n}")
    print(f"Benchmark (p0): {p0} ({p0*100}%)")
    
    # Hitung dengan Binomial Distribution
    print(f"\n--- Perhitungan Binomial ---")
    
    # Calculate probabilities for x, x+1, ..., n
    prob_sum = 0
    half_prob_x = 0
    
    for k in range(successes, n + 1):
        prob_k = stats.binom.pmf(k, n, p0)
        if k == successes:
            half_prob_x = 0.5 * prob_k
            print(f"½ × BINOM.DIST({k},{n},{p0},FALSE) = ½ × ({prob_k}) = {half_prob_x}")
        else:
            print(f"BINOM.DIST({k},{n},{p0},FALSE) = {prob_k}")
            prob_sum += prob_k
    
    # Mid-probability
    mid_p = half_prob_x + prob_sum
    confidence = (1 - mid_p) * 100
    
    # Truncate to 2 decimal places
    confidence_truncated = math.trunc(confidence * 100) / 100
    
    print(f"\nMid-probability = {half_prob_x} + {prob_sum} = {mid_p}")
    print(f"Confidence = 100% - {mid_p*100}% = {confidence}%")
    
    print(f"\n=== KESIMPULAN ===")
    print(f"→ Kemungkinan {p0*100}% pengguna setuju dengan respond tersebut {confidence_truncated}%")
    
    # Juga tampilkan distribusi rating
    print(f"\n--- Distribusi Rating ---")
    counts = [0] * scale
    for r in ratings:
        if 1 <= r <= scale:
            counts[r-1] += 1
    
    for i, count in enumerate(counts):
        pct = count/n*100
        bar = "█" * int(pct/2)
        print(f"  Rating {i+1}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Top-Box metrics
    top_box = counts[-1] / n * 100
    top_2_box = sum(counts[-2:]) / n * 100
    mean = sum((i+1) * count for i, count in enumerate(counts)) / n
    
    print(f"\n  Mean Rating: {mean}")
    print(f"  Top-Box (Rating {scale}): {top_box}%")
    print(f"  Top-2-Box (Rating {scale-1}-{scale}): {top_2_box}%")

# 6. Paired t-test - Halaman 15-16
def paired_ttest():
    print_header("PAIRED T-TEST (Within-Subjects)", "Materi 8 - Halaman 15-16")
    print_formula("t = D̄ / (sD / √n)", "15-16")
    print_formula("D̄ = mean of differences", "15-16")
    
    print("Digunakan untuk membandingkan 2 kondisi pada subjek yang SAMA")
    print("(within-subjects / repeated measures)")
    print()
    
    print("Pilih metode input:")
    print("1. Input raw data (skor per responden)")
    print("2. Input summary statistics (rata-rata perbedaan, std dev, n)")
    input_type = input("Pilihan (1/2): ")
    
    if input_type == "2":
        # Input summary statistics
        print("\n--- Input Summary Statistics ---")
        n = int(input("Masukkan jumlah responden (n): "))
        d_bar = float(input("Masukkan rata-rata perbedaan skor (D̄): "))
        s_d = float(input("Masukkan standar deviasi (sD): "))
        
        se = s_d / math.sqrt(n)
        t_stat = d_bar / se
        df = n - 1
        
        # T.DIST for one-tailed
        t_dist_value = stats.t.cdf(abs(t_stat), df)
        probability = t_dist_value * 100
        
        # Truncate to 2 decimal places
        prob_truncated = math.trunc(probability * 100) / 100
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Effect size (Cohen's d)
        cohens_d = d_bar / s_d
        
        print(f"\n{'='*70}")
        print("HASIL PERHITUNGAN:")
        print(f"{'='*70}")
        
        print(f"\nJumlah responden (n): {n}")
        print(f"Rata-rata perbedaan skor (D̄): {d_bar}")
        print(f"Standar Deviasi (sD): {s_d}")
        
        print(f"\n--- Perhitungan ---")
        print(f"\nt = D̄ / (sD / √n)")
        print(f"t = {d_bar} / ({s_d} / √{n})")
        print(f"t = {d_bar} / {se}")
        print(f"t = {t_stat}")
        
        print(f"\n→ T.DIST({t_stat}, {df}, TRUE) = {t_dist_value}")
        
        print(f"\n=== KESIMPULAN ===")
        if d_bar > 0:
            print(f"→ Disimpulkan bahwa {prob_truncated}% skor Kondisi A berbeda dibanding Kondisi B")
            print(f"→ Produk A significantly higher")
        elif d_bar < 0:
            print(f"→ Disimpulkan bahwa {prob_truncated}% skor Kondisi B berbeda dibanding Kondisi A")
            print(f"→ Produk B significantly higher")
        else:
            print(f"→ Tidak ada perbedaan antara kedua kondisi")
        
        print(f"\nP-value (two-tailed): {p_value}")
        print(f"Cohen's d: {cohens_d}")
        return
    
    # Input raw data
    n = int(input("\nMasukkan jumlah responden: "))
    
    print("\nPilih metode input data:")
    print("1. Input data satu per satu")
    print("2. Input semua data sekaligus (pisahkan dengan koma)")
    method = input("Pilihan (1/2): ")
    
    cond1 = []
    cond2 = []
    
    if method == "1":
        print(f"\nMasukkan skor untuk setiap responden:")
        for i in range(1, n + 1):
            a = float(input(f"  User {i} - Kondisi A: "))
            b = float(input(f"  User {i} - Kondisi B: "))
            cond1.append(a)
            cond2.append(b)
    else:
        print("\nMasukkan data Kondisi A (pisahkan dengan koma):")
        cond1_str = input("Kondisi A: ")
        cond1 = [float(x.strip()) for x in cond1_str.split(",")]
        
        print("Masukkan data Kondisi B (pisahkan dengan koma):")
        cond2_str = input("Kondisi B: ")
        cond2 = [float(x.strip()) for x in cond2_str.split(",")]
        
        if len(cond1) != n or len(cond2) != n:
            print(f"Warning: Jumlah data tidak sesuai, menggunakan jumlah data yang ada")
            n = min(len(cond1), len(cond2))
            cond1 = cond1[:n]
            cond2 = cond2[:n]
    
    # Calculate differences (A - B)
    differences = [a - b for a, b in zip(cond1, cond2)]
    
    d_bar = sum(differences) / n
    s_d = math.sqrt(sum((d - d_bar)**2 for d in differences) / (n - 1))
    se = s_d / math.sqrt(n)
    
    t_stat = d_bar / se
    df = n - 1
    
    # T.DIST for one-tailed (to compare which is higher)
    t_dist_value = stats.t.cdf(abs(t_stat), df)
    probability = t_dist_value * 100
    
    # Truncate to 2 decimal places
    prob_truncated = math.trunc(probability * 100) / 100
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Effect size (Cohen's d)
    cohens_d = d_bar / s_d
    
    print(f"\n{'='*70}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*70}")
    
    # Tampilkan tabel data
    print(f"\n{'User':<6} {'A':<10} {'B':<10} {'Difference (A-B)':<15}")
    print("-" * 45)
    for i in range(n):
        print(f"{i+1:<6} {cond1[i]:<10} {cond2[i]:<10} {differences[i]:<15}")
    print("-" * 45)
    print(f"{'Mean':<6} {sum(cond1)/n:<10} {sum(cond2)/n:<10} {d_bar:<15}")
    
    print(f"\n--- Perhitungan ---")
    print(f"Rata-rata perbedaan skor (D̄): {d_bar}")
    print(f"Standar Deviasi: {s_d}")
    
    print(f"\nt = D̄ / (sD / √n)")
    print(f"t = {d_bar} / ({s_d} / √{n})")
    print(f"t = {d_bar} / {se}")
    print(f"t = {t_stat}")
    
    print(f"\n→ T.DIST({t_stat}, {df}, TRUE) = {t_dist_value}")
    
    print(f"\n=== KESIMPULAN ===")
    if d_bar > 0:
        print(f"→ Disimpulkan bahwa {prob_truncated}% skor Kondisi A berbeda dibanding Kondisi B")
        print(f"→ Produk A significantly higher")
    elif d_bar < 0:
        print(f"→ Disimpulkan bahwa {prob_truncated}% skor Kondisi B berbeda dibanding Kondisi A")
        print(f"→ Produk B significantly higher")
    else:
        print(f"→ Tidak ada perbedaan antara kedua kondisi")
    
    print(f"\nP-value (two-tailed): {p_value}")
    print(f"Cohen's d: {cohens_d}")

# 7. Two-sample t-test - Halaman 16
def two_sample_ttest():
    print_header("TWO-SAMPLE T-TEST (Independent)", "Materi 8 - Halaman 16")
    print_formula("t = (X̄1 - X̄2) / √(s1²/n1 + s2²/n2)", "16")
    
    print("Digunakan untuk membandingkan 2 GRUP BERBEDA (between-subjects)")
    print("x̄1, x̄2 = rata-rata dari sampel 1 dan 2")
    print("s1, s2 = standard deviasi sampel 1 dan 2")
    print("n1, n2 = jumlah sampel 1 dan 2")
    print()
    
    print("Pilih metode input data:")
    print("1. Input data satu per satu")
    print("2. Input semua data sekaligus (pisahkan dengan koma)")
    print("3. Sudah ada rerata skor (input summary statistics)")
    method = input("Pilihan (1/2/3): ")
    
    if method == "3":
        # Input summary statistics
        print("\n--- Input Summary Statistics ---")
        print("\nGroup 1:")
        n1 = int(input("  Jumlah responden (n1): "))
        mean1 = float(input("  Rerata skor (x̄1): "))
        std1 = float(input("  Standar deviasi (s1): "))
        var1 = std1 ** 2
        
        print("\nGroup 2:")
        n2 = int(input("  Jumlah responden (n2): "))
        mean2 = float(input("  Rerata skor (x̄2): "))
        std2 = float(input("  Standar deviasi (s2): "))
        var2 = std2 ** 2
        
        # Rumus: t = (x̄1 - x̄2) / √(s1²/n1 + s2²/n2)
        se = math.sqrt((var1 / n1) + (var2 / n2))
        t_stat = (mean1 - mean2) / se
        
        # Degrees of freedom = n1 + n2 - 2
        df = n1 + n2 - 2
        
        # Two-tailed p-value (seperti di gambar: T.DIST menunjukkan two-tailed)
        two_tailed_p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Confidence = 100% - two-tailed p-value (seperti di gambar)
        confidence = (1 - two_tailed_p) * 100
        
        # Truncate to 2 decimal places
        conf_truncated = math.trunc(confidence * 100) / 100
        
        # Cohen's d
        pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std
        
        print(f"\n{'='*70}")
        print("HASIL PERHITUNGAN:")
        print(f"{'='*70}")
        
        print(f"\n--- Data Group 1 ---")
        print(f"n1 = {n1}")
        print(f"x̄1 (Mean) = {mean1}")
        print(f"s1 (Std Dev) = {std1}")
        print(f"s1² (Variance) = {var1}")
        
        print(f"\n--- Data Group 2 ---")
        print(f"n2 = {n2}")
        print(f"x̄2 (Mean) = {mean2}")
        print(f"s2 (Std Dev) = {std2}")
        print(f"s2² (Variance) = {var2}")
        
        print(f"\n--- Perhitungan ---")
        print(f"\nt = (x̄1 - x̄2) / √(s1²/n1 + s2²/n2)")
        print(f"t = ({mean1} - {mean2}) / √({var1}/{n1} + {var2}/{n2})")
        print(f"t = {mean1 - mean2} / √({var1/n1} + {var2/n2})")
        print(f"t = {mean1 - mean2} / √{(var1/n1) + (var2/n2)}")
        print(f"t = {mean1 - mean2} / {se}")
        print(f"t = {t_stat}")
        
        print(f"\nDegrees of freedom: {df}")
        print(f"\n→ T.DIST({t_stat}, {df}, TRUE) = {two_tailed_p}")
        
        print(f"\n=== KESIMPULAN ===")
        print(f"→ Disimpulkan bahwa kemungkinan skor Group 1 berbeda dari Group 2")
        print(f"  hanya sebesar {conf_truncated}%")
        if mean1 > mean2:
            print(f"→ Group 1 higher")
        elif mean1 < mean2:
            print(f"→ Group 2 higher")
        
        print(f"\nCohen's d: {cohens_d}")
        return
    
    # Tanyakan jumlah responden per grup
    print("\nMasukkan jumlah responden:")
    n1 = int(input("  Group 1 (n1): "))
    n2 = int(input("  Group 2 (n2): "))
    
    group1 = []
    group2 = []
    
    if method == "1":
        print(f"\nMasukkan skor untuk Group 1:")
        for i in range(1, n1 + 1):
            score = float(input(f"  User {i}: "))
            group1.append(score)
        
        print(f"\nMasukkan skor untuk Group 2:")
        for i in range(1, n2 + 1):
            score = float(input(f"  User {i}: "))
            group2.append(score)
    else:
        print("\nMasukkan data Group 1 (pisahkan dengan koma):")
        g1_str = input("Group 1: ")
        group1 = [float(x.strip()) for x in g1_str.split(",")]
        
        print("Masukkan data Group 2 (pisahkan dengan koma):")
        g2_str = input("Group 2: ")
        group2 = [float(x.strip()) for x in g2_str.split(",")]
        
        if len(group1) != n1 or len(group2) != n2:
            print(f"Warning: Jumlah data tidak sesuai, menggunakan jumlah data yang ada")
            n1 = len(group1)
            n2 = len(group2)
    
    # Hitung mean
    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2
    
    # Hitung variance dan std dev (sistem hitung sendiri)
    var1 = sum((x - mean1)**2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2)**2 for x in group2) / (n2 - 1)
    std1 = math.sqrt(var1)
    std2 = math.sqrt(var2)
    
    # Rumus yang benar: t = (x̄1 - x̄2) / √(s1²/n1 + s2²/n2)
    se = math.sqrt((var1 / n1) + (var2 / n2))
    t_stat = (mean1 - mean2) / se
    
    # Degrees of freedom = n1 + n2 - 2
    df = n1 + n2 - 2
    
    # Two-tailed p-value
    two_tailed_p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Confidence = 100% - two-tailed p-value
    confidence = (1 - two_tailed_p) * 100
    
    # Truncate to 2 decimal places
    conf_truncated = math.trunc(confidence * 100) / 100
    
    # Cohen's d
    pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std
    
    print(f"\n{'='*70}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*70}")
    
    # Tampilkan tabel data
    print(f"\n--- Data Group 1 ---")
    print(f"Data: {', '.join(map(str, group1))}")
    print(f"n1 = {n1}")
    print(f"x̄1 (Mean) = {mean1}")
    print(f"s1 (Std Dev) = {std1}")
    print(f"s1² (Variance) = {var1}")
    
    print(f"\n--- Data Group 2 ---")
    print(f"Data: {', '.join(map(str, group2))}")
    print(f"n2 = {n2}")
    print(f"x̄2 (Mean) = {mean2}")
    print(f"s2 (Std Dev) = {std2}")
    print(f"s2² (Variance) = {var2}")
    
    print(f"\n--- Perhitungan ---")
    print(f"\nt = (x̄1 - x̄2) / √(s1²/n1 + s2²/n2)")
    print(f"t = ({mean1} - {mean2}) / √({var1}/{n1} + {var2}/{n2})")
    print(f"t = {mean1 - mean2} / √({var1/n1} + {var2/n2})")
    print(f"t = {mean1 - mean2} / √{(var1/n1) + (var2/n2)}")
    print(f"t = {mean1 - mean2} / {se}")
    print(f"t = {t_stat}")
    
    print(f"\nDegrees of freedom: {df}")
    print(f"\n→ T.DIST({t_stat}, {df}, TRUE) = {two_tailed_p}")
    
    print(f"\n=== KESIMPULAN ===")
    print(f"→ Disimpulkan bahwa kemungkinan skor Group 1 berbeda dari Group 2")
    print(f"  hanya sebesar {conf_truncated}%")
    if mean1 > mean2:
        print(f"→ Group 1 higher")
    elif mean1 < mean2:
        print(f"→ Group 2 higher")
    
    print(f"\nCohen's d: {cohens_d}")

# 8. Chi-square Test of Independence - Halaman 20
def chi_square_test():
    print_header("CHI-SQUARE TEST OF INDEPENDENCE", "Materi 8 - Halaman 20")
    print_formula("χ² = (n × (ad - bc)²) / ((a+b)(c+d)(a+c)(b+d))", "20")
    
    print("Digunakan untuk menguji hubungan antara 2 variabel kategorikal")
    print("dalam desain between-subjects (2x2 contingency table)")
    print()
    print("         | Success | Failure |")
    print("---------|---------|---------|")
    print("Group A  |    a    |    b    |")
    print("Group B  |    c    |    d    |")
    print()
    
    print("Masukkan nilai cell:")
    a = int(input("  a (Group A - Success): "))
    b = int(input("  b (Group A - Failure): "))
    c = int(input("  c (Group B - Success): "))
    d = int(input("  d (Group B - Failure): "))
    
    # Hitung totals
    n = a + b + c + d
    row1 = a + b  # Group A total
    row2 = c + d  # Group B total
    col1 = a + c  # Success total
    col2 = b + d  # Failure total
    
    # Rumus chi-square sesuai PPT
    # χ² = (n × (ad - bc)²) / ((a+b)(c+d)(a+c)(b+d))
    ad_bc = (a * d) - (b * c)
    numerator = n * (ad_bc ** 2)
    denominator = row1 * row2 * col1 * col2
    
    chi2 = numerator / denominator
    
    # CHISQ.DIST.RT (right-tail) dengan df = 1 untuk 2x2 table
    df = 1
    p_value = 1 - stats.chi2.cdf(chi2, df)
    
    # Truncate p-value to 4 decimal places
    p_truncated = math.trunc(p_value * 10000) / 10000
    
    print(f"\n{'='*70}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*70}")
    
    print("\nContingency Table:")
    print(f"         | Success | Failure | Total")
    print(f"---------|---------|---------|-------")
    print(f"Group A  |   {a:4d}  |   {b:4d}  |  {row1:4d}")
    print(f"Group B  |   {c:4d}  |   {d:4d}  |  {row2:4d}")
    print(f"---------|---------|---------|-------")
    print(f"Total    |   {col1:4d}  |   {col2:4d}  |  {n:4d}")
    
    print(f"\n--- Perhitungan ---")
    print(f"\nχ² = (n × (ad - bc)²) / ((a+b)(c+d)(a+c)(b+d))")
    print(f"χ² = ({n} × ({a}×{d} - {b}×{c})²) / ({row1} × {row2} × {col1} × {col2})")
    print(f"χ² = ({n} × ({a*d} - {b*c})²) / ({row1} × {row2} × {col1} × {col2})")
    print(f"χ² = ({n} × {ad_bc}²) / {denominator}")
    print(f"χ² = ({n} × {ad_bc**2}) / {denominator}")
    print(f"χ² = {numerator} / {denominator}")
    print(f"χ² = {chi2}")
    
    print(f"\n→ CHISQ.DIST.RT({chi2}, {df}) = {p_value}")
    
    print(f"\n=== KESIMPULAN ===")
    if p_value < 0.05:
        print(f"→ Disimpulkan bahwa terdapat perbedaan yang significant secara")
        print(f"  statistik terhadap completion rate responden Group A dan Group B")
        print(f"  (p-value = {p_truncated} < 0.05)")
    else:
        print(f"→ Tidak terdapat perbedaan yang significant")
        print(f"  (p-value = {p_truncated} >= 0.05)")

# 9. Fisher Exact Test - Halaman 22
def fisher_exact_test():
    print_header("FISHER EXACT TEST", "Materi 8 - Halaman 22")
    print_formula("Odds Ratio = (a×d) / (b×c)", "22")
    
    print("Digunakan saat sample size kecil (expected freq < 5)")
    print("Lebih akurat dari Chi-square untuk sample kecil")
    print()
    print("         | Success | Failure |")
    print("---------|---------|---------|")
    print("Group A  |    a    |    b    |")
    print("Group B  |    c    |    d    |")
    print()
    
    print("Masukkan nilai cell:")
    a = int(input("  a (Group A - Success): "))
    b = int(input("  b (Group A - Failure): "))
    c = int(input("  c (Group B - Success): "))
    d = int(input("  d (Group B - Failure): "))
    
    # Hitung Odds Ratio manual
    if b * c == 0:
        odds_ratio = float('inf') if a * d > 0 else 0
    else:
        odds_ratio = (a * d) / (b * c)
    
    # Fisher exact test - one-tailed (greater) dan two-tailed
    table = [[a, b], [c, d]]
    _, p_two_tailed = stats.fisher_exact(table, alternative='two-sided')
    _, p_one_tailed = stats.fisher_exact(table, alternative='greater')
    
    # Totals
    n = a + b + c + d
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    
    print(f"\n{'='*70}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*70}")
    
    print("\nContingency Table:")
    print(f"         | Success | Failure | Total")
    print(f"---------|---------|---------|-------")
    print(f"Group A  |   {a:4d}  |   {b:4d}  |  {row1:4d}")
    print(f"Group B  |   {c:4d}  |   {d:4d}  |  {row2:4d}")
    print(f"---------|---------|---------|-------")
    print(f"Total    |   {col1:4d}  |   {col2:4d}  |  {n:4d}")
    
    print(f"\n--- Perhitungan ---")
    print(f"\nOdds Ratio = (a × d) / (b × c)")
    print(f"Odds Ratio = ({a} × {d}) / ({b} × {c})")
    print(f"Odds Ratio = {a*d} / {b*c}")
    print(f"Odds Ratio = {odds_ratio}")
    
    print(f"\np-value one-tailed (A > B): {p_one_tailed}")
    print(f"p-value two-tailed (A ≠ B): {p_two_tailed}")
    
    print(f"\n=== KESIMPULAN (α = 0.05) ===")
    if p_one_tailed < 0.05:
        print(f"→ p-value {p_one_tailed} < 0.05")
        print(f"→ Signifikan! Group A significantly higher than Group B")
    else:
        print(f"→ p-value {p_one_tailed} > 0.05")
        print(f"→ Belum signifikan di α = 0.05")
        if p_one_tailed < 0.10:
            print(f"→ Tapi signifikan jika pakai α = 0.10 (p-value < 0.10)")

# 10. McNemar Exact Test - Halaman 23
def mcnemar_test():
    print_header("McNEMAR EXACT TEST", "Materi 8 - Halaman 23")
    print_formula("McNemar focuses on discordant pairs (b and c)", "23")
    
    print("Digunakan untuk within-subjects comparison dengan data binary")
    print("(sama subjek diukur pada 2 kondisi berbeda)")
    print()
    print("Masukkan 2x2 table untuk paired data:")
    print("         Kondisi2=Yes  Kondisi2=No")
    print()
    
    a = int(input("Kondisi1=Yes, Kondisi2=Yes (a): "))
    b = int(input("Kondisi1=Yes, Kondisi2=No (b): "))
    c = int(input("Kondisi1=No, Kondisi2=Yes (c): "))
    d = int(input("Kondisi1=No, Kondisi2=No (d): "))
    
    # McNemar test with exact binomial
    n = b + c  # discordant pairs
    
    if n > 0:
        # Exact test using binomial
        if b > c:
            p_value = 2 * stats.binom.cdf(c, n, 0.5)
        else:
            p_value = 2 * stats.binom.cdf(b, n, 0.5)
        p_value = min(p_value, 1.0)
        
        # Chi-square approximation
        chi2 = (abs(b - c) - 1)**2 / (b + c)  # with continuity correction
    else:
        p_value = 1.0
        chi2 = 0
    
    print(f"\n{'='*50}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*50}")
    print("\nContingency Table:")
    print(f"              Kondisi2=Yes  Kondisi2=No")
    print(f"Kondisi1=Yes:     {a:5d}        {b:5d}")
    print(f"Kondisi1=No:      {c:5d}        {d:5d}")
    
    print(f"\nDiscordant pairs:")
    print(f"  b (Yes→No): {b}")
    print(f"  c (No→Yes): {c}")
    print(f"  Total discordant: {n}")
    
    if n > 0:
        print(f"\nMcNemar χ² (with correction): {chi2:.4f}")
    print(f"Exact P-value (two-tailed): {p_value:.4f}")
    print(f"\nKesimpulan (α = 0.05):")
    if p_value < 0.05:
        print(f"  ✓ Signifikan! Ada perbedaan antara kedua kondisi")
    else:
        print(f"  ✗ Tidak signifikan")

# ==============================================================================
#                                   MAIN MENU
# ==============================================================================

def main_menu():
    print("\n" + "="*60)
    print("       UX METRICS CALCULATOR - BY PDF MATERIAL")
    print("="*60)
    print("""
PILIH MATERI:

  8.  Quantifying User Research         [08_Quantifying_User_Research2.pdf]
  9.  Sample Size & T-Score             [09_Sample_Size.pdf, 09_t-score.pdf]
  10. Performance Based Metrics         [10_Performance_Based_Metrics.pdf]
  11. Issues Based Metrics              [11_Issues_Based_Behavioral...pdf]
  12. Self-Reported & Comparative       [12_Self_Reported_Metrics...pdf]

  0.  Keluar
""")
    return input("Pilihan Anda: ")

def run_materi_8():
    while True:
        choice = materi_8_menu()
        if choice == "0":
            break
        
        options = {
            "1": small_sample_test,
            "2": mid_probability_test,
            "3": large_sample_test,
            "4": satisfaction_vs_benchmark,
            "5": continuous_to_discrete,
            "6": paired_ttest,
            "7": two_sample_ttest,
            "8": chi_square_test,
            "9": fisher_exact_test,
            "10": mcnemar_test
        }
        
        if choice in options:
            try:
                options[choice]()
            except Exception as e:
                print(f"\nError: {e}")
            input("\n[Tekan Enter untuk melanjutkan...]")
        else:
            print("Pilihan tidak valid!")

# ==============================================================================
#                    MATERI 9: SAMPLE SIZE
# ==============================================================================

def materi_9_menu():
    print_header("MATERI 9: SAMPLE SIZE", "09_Sample_Size.pdf")
    print("""
SUMMATIVE Sample Size:
  1. Estimating Values (n = t²s²/d²)
  2. No Estimate of Variability (Effect Size)
  3. Comparing Values (Benchmark)
  4. Within-subjects Comparison
  5. Between-subjects Comparison

FORMATIVE Sample Size:
  6. Problem Discovery

  0. Kembali
""")
    return input("Pilihan: ")

# 1. Estimating Values - Hal 7
def summative_estimating_values():
    print_header("ESTIMATING VALUES", "Materi 9 - Halaman 7")
    print_formula("n = t²s² / d²", "7")
    
    print("n = sample size")
    print("s² = variance")
    print("d = critical difference (error maksimal yang diinginkan)")
    print("t = tingkat kepercayaan (tergantung df)")
    print()
    print("Menggunakan iterasi: z-score → t-score")
    print()
    
    print("Pilih metode input:")
    print("1. Input variance (s²) langsung")
    print("2. Input data mentah (variance dihitung otomatis)")
    method = input("Pilihan (1/2): ")
    
    if method == "2":
        print("\nMasukkan data (pisahkan dengan koma):")
        print("Contoh: 12, 14, 12, 20, 16")
        data_str = input("Data: ")
        data = [float(x.strip()) for x in data_str.split(",")]
        n_data = len(data)
        mean = sum(data) / n_data
        s2 = sum((x - mean)**2 for x in data) / (n_data - 1)
        s = math.sqrt(s2)
        
        print(f"\n--- Data Summary ---")
        print(f"Data: {data}")
        print(f"Mean (x̄): {mean}")
        print(f"Variance (s²): {s2}")
        print(f"Std Dev (s): {s}")
    else:
        s2 = float(input("\nMasukkan variance (s²): "))
        s = math.sqrt(s2)
    
    print("\nCara menentukan d (critical difference):")
    print("  - Berdasarkan pengalaman")
    print("  - what-if: misal 10% dari mean")
    d = float(input("Masukkan critical difference (d): "))
    
    conf_input = float(input("Masukkan tingkat kepercayaan (misal 90, 95, 99): "))
    conf = conf_input / 100 if conf_input > 1 else conf_input
    
    # Untuk estimating values (± d), gunakan two-tailed
    alpha = (1 - conf) / 2
    
    # Step 1: Hitung dengan z-score
    z = stats.norm.ppf(1 - alpha)
    n_raw_initial = (z**2 * s2) / (d**2)
    n_initial = max(2, math.ceil(n_raw_initial))  # Minimal n=2 agar df minimal 1
    
    print(f"\n{'='*70}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*70}")
    
    print(f"\n--- Iterasi ---")
    print(f"\n{'Iterasi':<10} {'t':<10} {'t²':<10} {'s²':<10} {'d':<10} {'d²':<10} {'df':<10} {'n (raw)':<12} {'n (rounded)':<10}")
    print("-" * 100)
    
    # Initial dengan z-score
    print(f"{'Initial':<10} {z:<10.4f} {z**2:<10.4f} {s2:<10} {d:<10} {d**2:<10.4f} {'-':<10} {n_raw_initial:<12.1f} {n_initial:<10}")
    
    n = n_initial
    prev_n = 0
    prev_prev_n = -1  # Untuk deteksi oscillation
    iteration = 1
    
    while n != prev_n and iteration <= 10:
        # Deteksi oscillation (n berputar antara 2 nilai)
        if n == prev_prev_n:
            print(f"\n⚠ Deteksi oscillation antara {prev_n} dan {n}")
            n = max(prev_n, n)  # Ambil yang lebih besar (lebih konservatif)
            print(f"→ Menggunakan nilai lebih besar: {n}")
            break
        
        prev_prev_n = prev_n
        prev_n = n
        df = n - 1
        t = stats.t.ppf(1 - alpha, df)
        n_raw = (t**2 * s2) / (d**2)
        n = max(2, math.ceil(n_raw))  # Minimal n=2 agar df minimal 1
        
        print(f"{iteration:<10} {t:<10.4f} {t**2:<10.4f} {s2:<10} {d:<10} {d**2:<10.4f} {df:<10} {n_raw:<12.1f} {n:<10}")
        
        if n == prev_n:
            break
        iteration += 1
    
    print(f"\n=== KESIMPULAN ===")
    print(f"→ Dibutuhkan sample size: {n} responden")

# 2. No Estimate of Variability - Hal 15
def summative_no_variability():
    print_header("NO ESTIMATE OF VARIABILITY", "Materi 9 - Halaman 15-17")
    print_formula("n = t² / e²", "15")
    
    print("Digunakan jika tidak mengetahui recognition variability (variance)")
    print("dari percobaan sebelumnya.")
    print()
    print("Gunakan effect size (e) dari Cohen (1988):")
    print("  - 0.2 = efek kecil")
    print("  - 0.5 = efek sedang")
    print("  - 0.8 = efek besar")
    print()
    print("e = d/s (effect size = critical difference / std dev)")
    print()
    
    print("Pilih effect size:")
    print("1. Kecil (0.2)")
    print("2. Sedang (0.5)")
    print("3. Besar (0.8)")
    print("4. Custom (masukkan nilai)")
    choice = input("Pilihan (1/2/3/4): ")
    
    if choice == "1":
        e = 0.2
    elif choice == "2":
        e = 0.5
    elif choice == "3":
        e = 0.8
    else:
        e = float(input("Masukkan effect size (misal 0.33): "))
    
    conf_input = float(input("Masukkan tingkat kepercayaan (misal 80, 90, 95): "))
    conf = conf_input / 100 if conf_input > 1 else conf_input
    
    # Two-tailed
    alpha = (1 - conf) / 2
    
    # Step 1: z-score
    z = stats.norm.ppf(1 - alpha)
    n_raw_initial = z**2 / e**2
    n_initial = max(2, math.ceil(n_raw_initial))
    
    print(f"\n{'='*70}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*70}")
    
    print(f"\nEffect size (e) = {e}")
    print(f"e² = {e**2}")
    print(f"Confidence = {conf*100}%")
    print(f"α = {(1-conf):.2f}, α/2 = {alpha:.2f} (two-tailed)")
    
    print(f"\n--- Iterasi ---")
    print(f"\n{'Step':<8} {'Rumus':<50} {'n':<10}")
    print("-" * 70)
    
    # Step 1: z-score
    print(f"{'1':<8} z-score = {z:.3f}")
    print(f"{'':8} n = {z:.3f}² / {e}² = {z**2:.4f} / {e**2:.4f} = {n_raw_initial:.1f} → {n_initial}")
    
    n = n_initial
    prev_n = 0
    prev_prev_n = -1
    step = 2
    
    while n != prev_n and step <= 10:
        # Deteksi oscillation
        if n == prev_prev_n:
            print(f"\n⚠ Deteksi oscillation antara {prev_n} dan {n}")
            n = max(prev_n, n)
            print(f"→ Menggunakan nilai lebih besar: {n}")
            break
        
        prev_prev_n = prev_n
        prev_n = n
        df = n - 1
        t = stats.t.ppf(1 - alpha, df)
        n_raw = t**2 / e**2
        n = max(2, math.ceil(n_raw))
        
        print(f"\n{step:<8} t-score dengan n = {prev_n}, (df={prev_n}-1={df})")
        print(f"{'':8} → t (two-tailed) = {alpha:.2f} → t({df}) = {t:.3f}")
        print(f"{'':8} n = {t:.3f}² / {e}² = {t**2:.4f} / {e**2:.4f} = {n_raw:.1f}, dibulatkan n = {n}")
        
        if n == prev_n:
            break
        step += 1
    
    print(f"\n=== KESIMPULAN ===")
    print(f"→ Dibutuhkan sample size: {n} responden")

# 3. Comparing Values (Benchmark) - Hal 19-20
def summative_comparing_benchmark():
    print_header("COMPARING VALUES (BENCHMARK)", "Materi 9 - Halaman 19-20")
    print_formula("n = (t² × s²) / d²", "19")
    
    print("CONTOH: Product requirement score mewajibkan hasil SUS")
    print("score setidaknya 75. Dalam evaluasi awal, diperoleh")
    print("SUS score 65. Setelah melakukan beberapa perbaikan, akan")
    print("dilakukan evaluasi kembali.")
    print("  • Variability dari percobaan sebelumnya : 5 (s=2.234)")
    print("  • Critical difference (d) = 1 point (SUS score >= 76)")
    print("  • Tingkat kepercayaan yang diharapkan 90%")
    print()
    
    s2 = float(input("Masukkan variance dari percobaan sebelumnya (s²): "))
    d = float(input("Masukkan critical difference (d): "))
    conf_input = float(input("Masukkan tingkat kepercayaan (misal 90, 95): "))
    conf = conf_input / 100 if conf_input > 1 else conf_input
    
    # One-tailed untuk benchmark
    alpha = 1 - conf
    
    # Step 1: z-score
    z = stats.norm.ppf(conf)  # Untuk one-tailed, langsung pakai conf
    n_raw_initial = (z**2 * s2) / (d**2)
    n_initial = max(2, math.ceil(n_raw_initial))
    
    print(f"\n{'='*70}")
    print("JAWAB:")
    print(f"{'='*70}")
    
    print(f"\n1. variance (S²) = {s2}")
    print(f"   z-score dengan {conf*100:.0f}% confidence = {z:.3f}")
    print(f"   → α = {alpha:.2f} → z : {conf:.2f}")
    print(f"   n = ({z:.3f}² x {int(s2)}) / {int(d)}² = {n_raw_initial:.1f} , dibulatkan n = {n_initial}")
    
    n = n_initial
    prev_n = 0
    prev_prev_n = -1
    step = 2
    
    while n != prev_n and step <= 10:
        # Deteksi oscillation
        if n == prev_prev_n:
            print(f"\n⚠ Deteksi oscillation antara {prev_n} dan {n}")
            n = max(prev_n, n)
            print(f"→ Menggunakan nilai lebih besar: {n}")
            break
        
        prev_prev_n = prev_n
        prev_n = n
        df = n - 1
        t = stats.t.ppf(conf, df)  # Untuk one-tailed, langsung pakai conf
        n_raw = (t**2 * s2) / (d**2)
        n = max(2, math.ceil(n_raw))
        
        print(f"\n{step}. t-score dengan n = {prev_n}, (df={prev_n}-1={df})")
        print(f"   → t (one-tailed) = {alpha:.2f} → t({df}) = {t:.3f}")
        print(f"   n = ({t:.3f}² x {int(s2)}) / {int(d)}² = {n_raw:.1f} , dibulatkan n = {n}")
        
        if n == prev_n:
            break
        step += 1
    
    print(f"\n=== KESIMPULAN ===")
    print(f"→ Dibutuhkan sample size: {n} responden")

# 4. Within-subjects Comparison - Hal 21-22
def summative_within_subjects():
    print_header("WITHIN-SUBJECTS COMPARISON", "Materi 9 - Halaman 21-22")
    print_formula("n = t²s² / d²", "21")
    
    print("Gunakan paired t-test (difference scores t-test)")
    print("Two-tailed test")
    print()
    print("Contoh: Membandingkan produk Anda dengan produk saingan")
    print()
    
    s2 = float(input("Masukkan variance dari percobaan sebelumnya (s²): "))
    d = float(input("Masukkan critical difference (d): "))
    conf_input = float(input("Masukkan tingkat kepercayaan (misal 99): "))
    conf = conf_input / 100 if conf_input > 1 else conf_input
    
    # Two-tailed
    alpha = (1 - conf) / 2
    
    # Initial dengan z-score
    z = stats.norm.ppf(1 - alpha)
    s = math.sqrt(s2)
    n_initial = math.ceil((z**2 * s2) / (d**2))
    
    print(f"\n{'='*70}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*70}")
    
    print(f"\ns² = {s2}, s = {s:.4f}")
    print(f"d = {d}, d² = {d**2}")
    
    print(f"\n--- Tabel Iterasi ---")
    print(f"\n{'':>10} {'Initial':>10} ", end="")
    
    # Iterasi
    n = n_initial
    iterations = [('Initial', z, z**2, s2, d, d**2, '-', (z**2*s2)/(d**2), n_initial)]
    
    prev_n = 0
    iter_num = 1
    while n != prev_n and iter_num <= 5:
        prev_n = n
        df = n - 1
        t = stats.t.ppf(1 - alpha, df)
        n_raw = (t**2 * s2) / (d**2)
        n = math.ceil(n_raw)
        iterations.append((str(iter_num), t, t**2, s2, d, d**2, df, n_raw, n))
        print(f"{iter_num:>10} ", end="")
        if n == prev_n:
            break
        iter_num += 1
    
    print()
    print("-" * (10 + 11 * len(iterations)))
    
    # Print rows
    row_labels = ['t', 't²', 's²', 'd', 'd²', 'df', 'Unrounded', 'Rounded up']
    for i, label in enumerate(row_labels):
        print(f"{label:>10} ", end="")
        for it in iterations:
            if i == 0:
                print(f"{it[1]:>10.4f} ", end="")
            elif i == 1:
                print(f"{it[2]:>10.4f} ", end="")
            elif i == 2:
                print(f"{it[3]:>10} ", end="")
            elif i == 3:
                print(f"{it[4]:>10} ", end="")
            elif i == 4:
                print(f"{it[5]:>10.4f} ", end="")
            elif i == 5:
                print(f"{str(it[6]):>10} ", end="")
            elif i == 6:
                print(f"{it[7]:>10.1f} ", end="")
            elif i == 7:
                print(f"{it[8]:>10} ", end="")
        print()
    
    print(f"\n=== KESIMPULAN ===")
    print(f"→ Dibutuhkan sample size: {n} responden")

# 5. Between-subjects Comparison - Hal 23-24
def summative_between_subjects():
    print_header("BETWEEN-SUBJECTS COMPARISON", "Materi 9 - Halaman 23-24")
    print_formula("n = 2t²s² / d²", "23")
    
    print("Lebih kompleks, jumlah sampel tiap kelompok bisa berbeda")
    print("Two-tailed test")
    print("Asumsi: kedua kelompok memiliki performance variability yang sama")
    print("        → peserta dari populasi yang sama dan random assignment")
    print()
    
    s2 = float(input("Masukkan variance dari percobaan sebelumnya (s²): "))
    d = float(input("Masukkan critical difference (d): "))
    conf_input = float(input("Masukkan tingkat kepercayaan (misal 99): "))
    conf = conf_input / 100 if conf_input > 1 else conf_input
    
    # Two-tailed
    alpha = (1 - conf) / 2
    
    # Initial dengan z-score (rumus n = 2z²s²/d²)
    z = stats.norm.ppf(1 - alpha)
    n_raw_initial = (2 * z**2 * s2) / (d**2)
    n_initial = max(2, math.ceil(n_raw_initial))
    df_initial = n_initial - 1  # df = n - 1
    
    print(f"\n{'='*70}")
    print("HASIL PERHITUNGAN:")
    print(f"{'='*70}")
    
    print(f"\ns² = {s2}")
    print(f"d = {d}, d² = {d**2}")
    print(f"Confidence = {conf*100}%, α/2 = {alpha}")
    
    # Iterasi dan simpan semua hasil
    # Format: (label, t, t², s², d, d², df, n_raw, n_rounded)
    iterations = [('Initial', z, z**2, s2, d, d**2, df_initial, n_raw_initial, n_initial)]
    
    n = n_initial
    prev_n = 0
    prev_prev_n = -1
    iter_num = 1
    
    while n != prev_n and iter_num <= 5:
        # Deteksi oscillation
        if n == prev_prev_n and iter_num > 1:
            n = max(prev_n, n)
            break
        
        prev_prev_n = prev_n
        prev_n = n
        df = n - 1  # df = n - 1 untuk iterasi (sesuai slide)
        t = stats.t.ppf(1 - alpha, df)
        n_raw = (2 * t**2 * s2) / (d**2)
        n = max(2, math.ceil(n_raw))
        df_result = n - 1  # df hasil dari n baru
        iterations.append((str(iter_num), t, t**2, s2, d, d**2, df_result, n_raw, n))
        if n == prev_n:
            break
        iter_num += 1
    
    # Print tabel seperti di slide
    print(f"\n--- Between-subjects Comparison of an Alternative ---\n")
    
    # Header
    print(f"{'':>12}", end="")
    for it in iterations:
        print(f"{it[0]:>12}", end="")
    print()
    print("-" * (12 + 12 * len(iterations)))
    
    # Print rows
    row_labels = ['t', 't²', 's²', 'd', 'd²', 'df', 'Unrounded', 'Rounded up']
    for i, label in enumerate(row_labels):
        print(f"{label:>12}", end="")
        for it in iterations:
            if i == 0:
                print(f"{it[1]:>12.3f}", end="")
            elif i == 1:
                print(f"{it[2]:>12.2f}", end="")
            elif i == 2:
                print(f"{int(it[3]):>12}", end="")
            elif i == 3:
                print(f"{it[4]:>12}", end="")
            elif i == 4:
                print(f"{it[5]:>12.2f}", end="")
            elif i == 5:
                print(f"{it[6]:>12}", end="")
            elif i == 6:
                print(f"{it[7]:>12.1f}", end="")
            elif i == 7:
                print(f"{it[8]:>12}", end="")
        print()
    
    print(f"\n=== KESIMPULAN ===")
    print(f"→ Dibutuhkan {n} responden untuk setiap group")
    print(f"→ Total dibutuhkan {n * 2} responden")

# 6. Formative - Problem Discovery - Hal 25-29
def formative_problem_discovery():
    print_header("FORMATIVE SAMPLE SIZE - PROBLEM DISCOVERY", "Materi 9 - Halaman 25-29")
    print_formula("P(x≥1) = 1 - (1-p)ⁿ", "25")
    print_formula("n = ln(1-P(x≥1)) / ln(1-p)", "27")
    
    print("p = probabilitas suatu event terjadi")
    print("n = banyaknya kesempatan event terjadi")
    print("P(x≥1) = probabilitas suatu event terjadi minimal sekali dalam n kesempatan")
    print()
    print("Contoh: Berapa responden untuk menemukan error dengan:")
    print("        80% kemungkinan menemukan setidaknya 1 error")
    print("        15% probabilitas terjadi error")
    print()
    
    print("Pilih mode:")
    print("1. Hitung Sample Size (n) dari P dan p")
    print("2. Hitung Probability P(x≥1) dari p dan n")
    mode = input("Pilihan (1/2): ")
    
    if mode == "2":
        p_input = float(input("\nMasukkan probabilitas event (p), misal 0.15 atau 15: "))
        p = p_input / 100 if p_input > 1 else p_input
        n = int(input("Masukkan sample size (n): "))
        
        P = 1 - (1 - p)**n
        
        print(f"\n{'='*70}")
        print("HASIL PERHITUNGAN:")
        print(f"{'='*70}")
        print(f"\nP(x≥1) = 1 - (1-p)ⁿ")
        print(f"P(x≥1) = 1 - (1-{p})^{n}")
        print(f"P(x≥1) = 1 - {(1-p)**n}")
        print(f"P(x≥1) = {P}")
        
        print(f"\n=== KESIMPULAN ===")
        print(f"→ Dengan {n} user dan probabilitas {p*100}%,")
        print(f"  kemungkinan menemukan minimal 1 event adalah {P*100:.1f}%")
    else:
        P_input = float(input("\nMasukkan target probability P(x≥1), misal 0.80 atau 80: "))
        P = P_input / 100 if P_input > 1 else P_input
        p_input = float(input("Masukkan probabilitas event (p), misal 0.15 atau 15: "))
        p = p_input / 100 if p_input > 1 else p_input
        
        n = math.log(1 - P) / math.log(1 - p)
        
        print(f"\n{'='*70}")
        print("HASIL PERHITUNGAN:")
        print(f"{'='*70}")
        print(f"\nn = ln(1-P) / ln(1-p)")
        print(f"n = ln(1-{P}) / ln(1-{p})")
        print(f"n = ln({1-P}) / ln({1-p})")
        print(f"n = {math.log(1-P)} / {math.log(1-p)}")
        print(f"n = {n}")
        
        print(f"\n=== KESIMPULAN ===")
        print(f"→ Dibutuhkan {math.ceil(n)} user untuk {P*100}% kemungkinan")
        print(f"  menemukan minimal 1 event dengan probabilitas {p*100}%")

def run_materi_9():
    opts = {
        "1": summative_estimating_values,
        "2": summative_no_variability,
        "3": summative_comparing_benchmark,
        "4": summative_within_subjects,
        "5": summative_between_subjects,
        "6": formative_problem_discovery
    }
    while True:
        c = materi_9_menu()
        if c == "0":
            break
        if c in opts:
            try:
                opts[c]()
            except Exception as e:
                print(f"Error: {e}")
            input("\n[Enter untuk lanjut...]")

# ==============================================================================
#                    MATERI 10: PERFORMANCE BASED METRICS
# ==============================================================================

def materi_10_menu():
    print_header("MATERI 10: PERFORMANCE BASED METRICS", "10_Performance_Based_Metrics.pdf")
    print("""
  1. Task Success Rate + CI
  2. Task Time (Geometric Mean)
  3. Error Rate
  4. Efficiency
  5. Lostness

  0. Kembali
""")
    return input("Pilihan: ")

def task_success_ci():
    print_header("Task Success Rate + CI", "Materi 10")
    print_formula("Success Rate = x/n, CI menggunakan adjusted Wald", "Performance PDF")
    x = int(input("Successful tasks: "))
    n = int(input("Total tasks: "))
    p = x/n
    z = 1.96
    n_adj = n + z**2
    p_adj = (x + z**2/2) / n_adj
    se = math.sqrt(p_adj*(1-p_adj)/n_adj)
    print(f"\n=== HASIL ===\nSuccess Rate: {p*100:.1f}%\n95% CI: [{max(0,(p_adj-z*se)*100):.1f}%, {min(100,(p_adj+z*se)*100):.1f}%]")

def geometric_mean_time():
    print_header("Task Time (Geometric Mean)", "Materi 10")
    print_formula("GM = exp(Σlog(xi)/n)", "Performance PDF")
    print("Task times (detik, pisah koma):")
    times = [float(x.strip()) for x in input().split(",")]
    gm = math.exp(sum(math.log(t) for t in times) / len(times))
    am = sum(times)/len(times)
    print(f"\n=== HASIL ===\nArithmetic Mean: {am:.2f}s\nGeometric Mean: {gm:.2f}s")

def error_rate_calc():
    print_header("Error Rate", "Materi 10")
    print_formula("Error Rate = errors / opportunities", "Performance PDF")
    e = int(input("Total errors: "))
    o = int(input("Total opportunities: "))
    print(f"\n=== HASIL ===\nError Rate: {e/o:.4f} ({e/o*100:.2f}%)")

def efficiency_calc():
    print_header("Efficiency", "Materi 10")
    print_formula("Efficiency = Success / Time", "Performance PDF")
    s = float(input("Success rate (0-1): "))
    t = float(input("Avg time (seconds): "))
    print(f"\n=== HASIL ===\nEfficiency: {s/t:.4f}")

def lostness_calc():
    print_header("Lostness", "Materi 10")
    print_formula("L = √((N/S - 1)² + (R/N - 1)²)", "Performance PDF")
    N = int(input("Nodes visited (N): "))
    S = int(input("Optimal path (S): "))
    R = int(input("Unique nodes (R): "))
    L = math.sqrt((N/S - 1)**2 + (R/N - 1)**2)
    print(f"\n=== HASIL ===\nLostness: {L:.4f}\nInterpretasi: {'Baik' if L < 0.4 else 'Sedang' if L < 0.5 else 'Buruk'}")

def run_materi_10():
    opts = {"1": task_success_ci, "2": geometric_mean_time, "3": error_rate_calc, "4": efficiency_calc, "5": lostness_calc}
    while True:
        c = materi_10_menu()
        if c == "0": break
        if c in opts:
            try: opts[c]()
            except Exception as e: print(f"Error: {e}")
            input("\n[Enter untuk lanjut...]")

# ==============================================================================
#                    MATERI 11: ISSUES BASED METRICS
# ==============================================================================

def materi_11_menu():
    print_header("MATERI 11: ISSUES BASED METRICS", "11_Issues_Based_Behavioral...pdf")
    print("""
  1. Problem Discovery Rate
  2. Severity Rating
  3. ROI Calculator
  4. Unique Problems per User

  0. Kembali
""")
    return input("Pilihan: ")

def problem_discovery_rate():
    print_header("Problem Discovery Rate", "Materi 11")
    print_formula("P(discovery) = 1 - (1-p)^n", "Issues PDF")
    p = float(input("Prob per user (p): "))
    n = int(input("Number of users: "))
    P = 1 - (1-p)**n
    print(f"\n=== HASIL ===\nDiscovery Rate: {P:.4f} ({P*100:.1f}%)")

def severity_rating_calc():
    print_header("Severity Rating", "Materi 11")
    print("Frequency (1-4): "); f = int(input())
    print("Impact (1-4): "); i = int(input())
    print("Persistence (1-4): "); p = int(input())
    sev = (f+i+p)/3
    print(f"\n=== HASIL ===\nSeverity: {sev:.2f}\nPriority: {'Low' if sev<2 else 'Medium' if sev<3 else 'High'}")

def roi_calc():
    print_header("ROI Calculator", "Materi 11")
    print_formula("ROI = (Benefits - Costs) / Costs × 100%", "Issues PDF")
    b = float(input("Benefits ($): "))
    c = float(input("Costs ($): "))
    print(f"\n=== HASIL ===\nROI: {((b-c)/c)*100:.1f}%")

def unique_problems():
    print_header("Unique Problems per User", "Materi 11")
    total = int(input("Total unique problems found: "))
    n = int(input("Number of users: "))
    print(f"\n=== HASIL ===\nAvg problems/user: {total/n:.2f}")

def run_materi_11():
    opts = {"1": problem_discovery_rate, "2": severity_rating_calc, "3": roi_calc, "4": unique_problems}
    while True:
        c = materi_11_menu()
        if c == "0": break
        if c in opts:
            try: opts[c]()
            except Exception as e: print(f"Error: {e}")
            input("\n[Enter untuk lanjut...]")

# ==============================================================================
#                    MATERI 12: SELF-REPORTED & COMPARATIVE
# ==============================================================================

def materi_12_menu():
    print_header("MATERI 12: SELF-REPORTED & COMPARATIVE", "12_Self_Reported_Metrics...pdf")
    print("""
SELF-REPORTED:
  1. SUS Score
  2. NPS Score
  3. SEQ Score
  4. UMUX-LITE

COMPARATIVE:
  5. A/B Test (Proportions)
  6. Cohen's d Effect Size

  0. Kembali
""")
    return input("Pilihan: ")

def sus_calc():
    print_header("SUS Score", "Materi 12")
    print_formula("SUS = ((Σodd-5) + (25-Σeven)) × 2.5", "Self-Reported PDF")
    print("Masukkan 10 skor (1-5):")
    scores = [int(input(f"Q{i+1}: ")) for i in range(10)]
    adj = [(s-1) if i%2==0 else (5-s) for i,s in enumerate(scores)]
    sus = sum(adj) * 2.5
    grade = "A" if sus>=80 else "B" if sus>=68 else "C" if sus>=51 else "F"
    print(f"\n=== HASIL ===\nSUS Score: {sus:.1f}\nGrade: {grade}")

def nps_calc():
    print_header("NPS Score", "Materi 12")
    print_formula("NPS = %Promoters - %Detractors", "Self-Reported PDF")
    p = int(input("Promoters (9-10): "))
    pa = int(input("Passives (7-8): "))
    d = int(input("Detractors (0-6): "))
    total = p + pa + d
    nps = ((p - d) / total) * 100
    print(f"\n=== HASIL ===\nNPS: {nps:.1f}")

def seq_calc():
    print_header("SEQ Score", "Materi 12")
    print("SEQ scores (1-7, pisah koma):")
    scores = [float(x.strip()) for x in input().split(",")]
    mean = sum(scores)/len(scores)
    print(f"\n=== HASIL ===\nMean SEQ: {mean:.2f}\nBenchmark: ≥5.5 adalah baik")

def umux_calc():
    print_header("UMUX-LITE", "Materi 12")
    q1 = float(input("Q1 capabilities (1-7): "))
    q2 = float(input("Q2 easy to use (1-7): "))
    umux = ((q1 + q2 - 2) / 12) * 100
    print(f"\n=== HASIL ===\nUMUX-LITE: {umux:.1f}")

def ab_proportions():
    print_header("A/B Test (Proportions)", "Materi 12")
    print_formula("Z = (p1-p2) / SE", "Comparative PDF")
    x1, n1 = int(input("A successes: ")), int(input("A total: "))
    x2, n2 = int(input("B successes: ")), int(input("B total: "))
    p1, p2 = x1/n1, x2/n2
    p_pool = (x1+x2)/(n1+n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    z = (p1-p2)/se
    p_val = 2*(1-stats.norm.cdf(abs(z)))
    print(f"\n=== HASIL ===\np1: {p1:.4f}, p2: {p2:.4f}\nZ: {z:.4f}\nP-value: {p_val:.4f}\nSignifikan: {'Ya' if p_val<0.05 else 'Tidak'}")

def cohens_d_calc():
    print_header("Cohen's d Effect Size", "Materi 12")
    print_formula("d = (M1-M2) / SD_pooled", "Comparative PDF")
    m1, m2 = float(input("Mean 1: ")), float(input("Mean 2: "))
    s1, s2 = float(input("SD 1: ")), float(input("SD 2: "))
    n1, n2 = int(input("n1: ")), int(input("n2: "))
    sp = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
    d = (m1-m2)/sp
    size = "Small" if abs(d)<0.5 else "Medium" if abs(d)<0.8 else "Large"
    print(f"\n=== HASIL ===\nCohen's d: {d:.4f}\nEffect Size: {size}")

def run_materi_12():
    opts = {"1": sus_calc, "2": nps_calc, "3": seq_calc, "4": umux_calc, "5": ab_proportions, "6": cohens_d_calc}
    while True:
        c = materi_12_menu()
        if c == "0": break
        if c in opts:
            try: opts[c]()
            except Exception as e: print(f"Error: {e}")
            input("\n[Enter untuk lanjut...]")

# ==============================================================================
#                                   MAIN
# ==============================================================================

def main():
    print("\n🎓 UX Metrics Calculator")
    print("   Berdasarkan materi kuliah Anda\n")
    
    while True:
        choice = main_menu()
        if choice == "0":
            print("\n✨ Semoga sukses dengan tugas Anda! 🎉\n")
            break
        elif choice == "8": run_materi_8()
        elif choice == "9": run_materi_9()
        elif choice == "10": run_materi_10()
        elif choice == "11": run_materi_11()
        elif choice == "12": run_materi_12()
        else: print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()
