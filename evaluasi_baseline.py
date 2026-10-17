import math
import config
import pandas as pd


def hitung_wilson(k, n):
  if n == 0:
    return 0.0, 0.0
  z = 1.959963984540054
  p = k / n
  pembagi = 1.0 + (z**2) / n
  tengah = (p + (z**2) / (2.0 * n)) / pembagi
  rentang = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / pembagi
  return max(0.0, tengah - rentang), min(1.0, tengah + rentang)
def main():
  print(
      f"Membaca {config.PARQUET_TEST} untuk evaluasi baseline rst_count == 0"
  )
  df_test = pd.read_parquet(config.PARQUET_TEST)
  if "rst_count" not in df_test.columns:
    raise KeyError("Kolom rst_count tidak ada di dataset test")
  y_test = df_test["target"].values
  pred = (df_test["rst_count"].values == 0).astype(int)

  tp = int(((pred == 1) & (y_test == 1)).sum())
  fn = int(((pred == 0) & (y_test == 1)).sum())
  fp = int(((pred == 1) & (y_test == 0)).sum())
  tn = int(((pred == 0) & (y_test == 0)).sum())
  total_serangan = tp + fn
  total_benign = fp + tn
  recall = tp / total_serangan if total_serangan > 0 else 0.0
  fpr = fp / total_benign if total_benign > 0 else 0.0
  ci_bawah, ci_atas = hitung_wilson(fp, total_benign)
  print(f"Hasil baseline rst_count == 0 ({len(df_test):,} baris):")
  print(
      f"Recall serangan: {recall * 100:.2f}% ({fn:,} lolos dari"
      f" {total_serangan:,})"
  )
  print(f"FP benign: {fp:,} dari {total_benign:,} ({fpr * 100:.2f}%)")
  print(f"Wilson 95% CI FPR: {ci_bawah * 100:.2f}% sampai {ci_atas * 100:.2f}%")
if __name__ == "__main__":
  main()