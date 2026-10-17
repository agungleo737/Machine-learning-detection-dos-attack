import argparse
import config
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


def ambil_argumen():
  parser = argparse.ArgumentParser()
  parser.add_argument("--split", choices=["val", "test"], default="val")
  parser.add_argument(
      "--class-weight", choices=["none", "balanced", "both"], default="both"
  )
  return parser.parse_args()


def jalankan_pohon(depth, bobot, x_train, y_train, x_uji, y_uji, nama_split):
  pohon = DecisionTreeClassifier(
      max_depth=depth, class_weight=bobot, random_state=config.SEED
  )
  pohon.fit(x_train, y_train)
  pred = pohon.predict(x_uji)
  tp = int(((pred == 1) & (y_uji == 1)).sum())
  fn = int(((pred == 0) & (y_uji == 1)).sum())
  fp = int(((pred == 1) & (y_uji == 0)).sum())
  tn = int(((pred == 0) & (y_uji == 0)).sum())
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

  print(
      f"\nDecision Tree depth {depth} (class_weight={bobot}, split={nama_split},"
      f" {len(y_uji):,} baris):"
  )
  print(f"Recall serangan: {recall * 100:.2f}% ({fn:,} lolos dari {tp + fn:,})")
  print(f"FP benign: {fp:,} dari {fp + tn:,} ({fpr * 100:.4f}%)")
  print(export_text(pohon, feature_names=list(x_train.columns)))


def main():
  args = ambil_argumen()
  if args.split == "test" and args.class_weight == "both":
    raise ValueError(
        "Pilih salah satu class_weight saat evaluasi"
        " di test."
    )

  daftar_fitur = config.ambil_fitur()
  path_uji = config.PARQUET_VAL if args.split == "val" else config.PARQUET_TEST
  print(f"Membaca {config.PARQUET_TRAIN} dan {path_uji}")
  df_train = pd.read_parquet(config.PARQUET_TRAIN)
  df_uji = pd.read_parquet(path_uji)
  fitur = config.pilih_kolom(df_train.columns, daftar_fitur)
  x_train, y_train = df_train[fitur], df_train["target"].values
  x_uji, y_uji = df_uji[fitur], df_uji["target"].values
  if args.class_weight == "none":
    opsi_bobot = [None]
  elif args.class_weight == "balanced":
    opsi_bobot = ["balanced"]
  else:
    opsi_bobot = [None, "balanced"]
  for bobot in opsi_bobot:
    for depth in (2, 3):
      jalankan_pohon(depth, bobot, x_train, y_train, x_uji, y_uji, args.split)

if __name__ == "__main__":
  main()