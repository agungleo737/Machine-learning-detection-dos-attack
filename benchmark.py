import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import config

print("Akurasi dan kecepatan")
fitur = json.load(open(config.ARTEFAK_FITUR, encoding="utf-8"))
print(f"Memuat model: {config.ARTEFAK_MODEL}")
booster = lgb.Booster(model_file=config.ARTEFAK_MODEL)
print(f"Memuat Test Set Dedup: {config.PARQUET_TEST}")
df_test = pd.read_parquet(config.PARQUET_TEST, columns=fitur + ["rst_count", "target"])
X_test = df_test[fitur].astype(np.float32).to_numpy()
y_test = df_test["target"].to_numpy()
rst_col = df_test["rst_count"].to_numpy()
n_total = len(y_test)
mask_ben = (y_test == 0)
mask_atk = (y_test == 1)
n_ben = int(mask_ben.sum())
n_atk = int(mask_atk.sum())

print("\nMengukur kecepatan inferensi")
_ = booster.predict(X_test[:1000])
durasi_list = []
for i in range(5):
    t0 = time.perf_counter()
    prob_lgb = booster.predict(X_test, num_iteration=config.NUM_ITERATIONS)
    t1 = time.perf_counter()
    durasi_list.append(t1 - t0)
waktu_median = float(np.median(durasi_list))
throughput = n_total / waktu_median
latensi_per_1k = (waktu_median / n_total) * 1000 * 1000 

pred_lgb = (prob_lgb >= config.THRESHOLD_OPERASIONAL).astype(np.int8)
tp_lgb = int(((pred_lgb == 1) & mask_atk).sum())
fn_lgb = n_atk - tp_lgb
fp_lgb = int(((pred_lgb == 1) & mask_ben).sum())
rec_lgb = (tp_lgb / n_atk) * 100
fpr_lgb = (fp_lgb / n_ben) * 100
pred_base = (rst_col == 0).astype(np.int8)
tp_base = int(((pred_base == 1) & mask_atk).sum())
fn_base = n_atk - tp_base
fp_base = int(((pred_base == 1) & mask_ben).sum())
rec_base = (tp_base / n_atk) * 100
fpr_base = (fp_base / n_ben) * 100

print("hasil benchmark")
print(f"{'Metode / Model':<32} | {'Recall (%)':<12} | {'Serangan Lolos':<15} | {'FP Benign':<12} | {'FPR (%)':<10}")
print(f"{'Baseline Rule (rst_count == 0)':<32} | {rec_base:>10.4f}% | {fn_base:>15,} | {fp_base:>12,} | {fpr_base:>8.4f}%")
print(f"{'LightGBM Final)':<32} | {rec_lgb:>10.4f}% | {fn_lgb:>15,} | {fp_lgb:>12,} | {fpr_lgb:>8.4f}%")

print("hasil benchmark kecepatan reaksi")
print(f"Total Paket Diuji          : {n_total:,} aliran (flows)")
print(f"Waktu Inferensi (Median)   : {waktu_median:.4f} detik")
print(f"Kecepatan (Throughput)     : {throughput:,.0f} aliran / detik")
print(f"Latensi Rata-rata          : {latensi_per_1k:.4f} ms per 1.000 aliran")