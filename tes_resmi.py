import argparse
from datetime import datetime
import json
import math
import os
import sys
import config
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def hitung_wilson(k, n):
  if n == 0:
    return 0.0, 0.0
  z = 1.959963984540054
  p = k / n
  pembagi = 1.0 + (z**2) / n
  tengah = (p + (z**2) / (2.0 * n)) / pembagi
  rentang = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / pembagi
  return max(0.0, tengah - rentang), min(1.0, tengah + rentang)

def ambil_argumen():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default=config.ARTEFAK_MODEL)
  parser.add_argument("--with-subtypes", action="store_true", default=True)
  return parser.parse_args()

def main():
  if os.path.exists(config.ARTEFAK_TES_LOCK):
    with open(config.ARTEFAK_TES_LOCK, "r", encoding="utf-8") as f:
      data = json.load(f)
    print(f"Evaluasi test sudah pernah dijalankan pada {data.get('timestamp')}")
    print(f"Model: {data.get('model_path')} ({data.get('model_sha256')})")
    print(
        f"Baris test: {data['test_rows']:,} (benign: {data['benign_rows']:,},"
        f" serangan: {data['attack_rows']:,})"
    )
    print(
        f"Recall serangan: {data['recall_pct']:.4f}% ({data['fn']:,} lolos dari"
        f" {data['attack_rows']:,})"
    )
    print(
        f"FP benign: {data['fp']:,} dari {data['benign_rows']:,}"
        f" ({data['fpr_pct']:.4f}%)"
    )
    print(
        f"Wilson 95% CI FPR: {data['wilson_low_pct']:.4f}% sampai"
        f" {data['wilson_high_pct']:.4f}%"
    )
    print(f"ROC-AUC: {data['roc_auc']:.7f} | PR-AUC: {data['pr_auc']:.7f}")
    sys.exit(0)

  args = ambil_argumen()
  path_model = args.model
  if not os.path.exists(path_model):
    raise FileNotFoundError(f"File model {path_model} tidak ditemukan.")
  if not os.path.exists(config.PARQUET_TEST):
    raise FileNotFoundError(
        f"File test {config.PARQUET_TEST} tidak ditemukan."
    )
  sha_model = config.hitung_sha256(path_model)
  sha_test = config.hitung_sha256(config.PARQUET_TEST)
  daftar_fitur = config.ambil_fitur()
  print(f"Membaca {config.PARQUET_TEST} dan model {path_model}")
  df_test = pd.read_parquet(config.PARQUET_TEST)
  fitur = config.pilih_kolom(df_test.columns, daftar_fitur)
  model = lgb.Booster(model_file=path_model)
  x_test = df_test[fitur]
  y_test = df_test["target"].values
  prob = model.predict(x_test)
  pred = (prob >= config.THRESHOLD_OPERASIONAL).astype(int)
  tp = int(((pred == 1) & (y_test == 1)).sum())
  fn = int(((pred == 0) & (y_test == 1)).sum())
  fp = int(((pred == 1) & (y_test == 0)).sum())
  tn = int(((pred == 0) & (y_test == 0)).sum())
  total_serangan = tp + fn
  total_benign = fp + tn
  recall = tp / total_serangan if total_serangan > 0 else 0.0
  fpr = fp / total_benign if total_benign > 0 else 0.0
  ci_bawah, ci_atas = hitung_wilson(fp, total_benign)
  roc_auc = float(roc_auc_score(y_test, prob))
  pr_auc = float(average_precision_score(y_test, prob))
  hasil_tes = {
      "timestamp": datetime.now().isoformat(timespec="seconds"),
      "model_path": path_model,
      "model_sha256": sha_model,
      "test_parquet_path": config.PARQUET_TEST,
      "test_parquet_sha256": sha_test,
      "seed": config.SEED,
      "split_mode": config.SPLIT_MODE,
      "num_features": len(fitur),
      "features": fitur,
      "num_iterations": config.NUM_ITERATIONS,
      "threshold": config.THRESHOLD_OPERASIONAL,
      "lgb_params": config.LGB_PARAMS,
      "test_rows": len(df_test),
      "benign_rows": total_benign,
      "attack_rows": total_serangan,
      "tp": tp,
      "fn": fn,
      "fp": fp,
      "tn": tn,
      "recall_pct": recall * 100.0,
      "fpr_pct": fpr * 100.0,
      "wilson_low_pct": ci_bawah * 100.0,
      "wilson_high_pct": ci_atas * 100.0,
      "roc_auc": roc_auc,
      "pr_auc": pr_auc,
  }
  with open(config.ARTEFAK_TES_LOCK, "w", encoding="utf-8") as f:
    json.dump(hasil_tes, f, indent=2)
  print(
      f"Hasil evaluasi test ({len(df_test):,} baris, {len(fitur)} fitur,"
      f" {config.NUM_ITERATIONS} iterasi, seed {config.SEED}):"
  )
  print(f"Model: {path_model} ({sha_model})")
  print(
      f"Recall serangan: {recall * 100:.4f}% ({fn:,} lolos dari"
      f" {total_serangan:,})"
  )
  print(f"FP benign: {fp:,} dari {total_benign:,} ({fpr * 100:.4f}%)")
  print(
      f"Wilson 95% CI FPR: {ci_bawah * 100:.4f}% sampai {ci_atas * 100:.4f}%"
  )
  print(f"ROC-AUC: {roc_auc:.7f} | PR-AUC: {pr_auc:.7f}")

  if "label" in df_test.columns and args.with_subtypes:
    print("\nRecall per sub-tipe di test:")
    df_sub = pd.DataFrame(
        {"label": df_test["label"], "target": y_test, "pred": pred}
    )
    df_att = df_sub[df_sub["target"] == 1]
    rekap = (
        df_att.groupby("label")["pred"]
        .agg(total="count", terdeteksi="sum")
        .assign(
            lolos=lambda d: d["total"] - d["terdeteksi"],
            recall_persen=lambda d: (d["terdeteksi"] / d["total"]) * 100.0,
        )
        .sort_values("recall_persen", ascending=True)
    )
    print(rekap.to_string())

if __name__ == "__main__":
  main()