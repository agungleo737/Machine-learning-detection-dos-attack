import config
import lightgbm as lgb
import pandas as pd

DAFTAR_SEED = [42, 100, 2023]
def evaluasi_multi_seed(df_train, df_val, fitur):
  x_train, y_train = df_train[fitur], df_train["target"].values
  x_val, y_val = df_val[fitur], df_val["target"].values
  dtrain = lgb.Dataset(
      x_train, label=y_train, feature_name=fitur, free_raw_data=False
  )
  hasil_seed = []
  for s in DAFTAR_SEED:
    params = dict(config.LGB_PARAMS)
    params["random_state"] = s
    params["bagging_seed"] = s
    params["feature_fraction_seed"] = s
    model = lgb.train(
        params=params, train_set=dtrain, num_boost_round=config.NUM_ITERATIONS
    )
    prob = model.predict(x_val)
    pred = (prob >= config.THRESHOLD_OPERASIONAL).astype(int)
    df_cek = pd.DataFrame({
        "label": df_val["label"].values,
        "target": y_val,
        "pred": pred,
    })
    df_serangan = df_cek[df_cek["target"] == 1]
    per_label = (
        df_serangan.groupby("label")["pred"]
        .agg(total="count", terdeteksi="sum")
        .assign(recall=lambda d: (d["terdeteksi"] / d["total"]) * 100.0)
    )
    per_label["seed"] = s
    hasil_seed.append(per_label.reset_index())
  df_gabung = pd.concat(hasil_seed, ignore_index=True)
  tabel = df_gabung.pivot(index="label", columns="seed", values="recall")
  tabel["min"] = tabel.min(axis=1)
  tabel["max"] = tabel.max(axis=1)
  tabel = tabel.sort_values("min", ascending=True)
  print(f"Recall per sub-tipe di val ({len(df_val):,} baris, 3 seed):")
  print(tabel.to_string(float_format=lambda x: f"{x:.2f}%"))

def evaluasi_isolasi_keluarga(df_train, df_val, fitur):
  skenario = [
      (
          "Tanpa DDoS-ICMP_Fragmentation",
          lambda s: s == "DDoS-ICMP_Fragmentation",
      ),
      (
          "Tanpa keluarga ICMP",
          lambda s: s.str.contains("ICMP", case=False, na=False),
      ),
      ("Tanpa DoS-UDP_Flood", lambda s: s == "DoS-UDP_Flood"),
      (
          "Tanpa keluarga UDP",
          lambda s: s.str.contains("UDP", case=False, na=False),
      ),
  ]

  print("\nUji isolasi sub-tipe dan keluarga di val:")
  x_val = df_val[fitur]
  label_val = df_val["label"].astype(str).str.strip()
  label_train = df_train["label"].astype(str).str.strip()
  for nama, filter_buang in skenario:
    mask_buang = filter_buang(label_train)
    df_sub = df_train.loc[~mask_buang]
    dtrain_sub = lgb.Dataset(
        df_sub[fitur], label=df_sub["target"].values, feature_name=fitur
    )
    model_sub = lgb.train(
        params=config.LGB_PARAMS,
        train_set=dtrain_sub,
        num_boost_round=config.NUM_ITERATIONS,
    )
    pred_val = (model_sub.predict(x_val) >= config.THRESHOLD_OPERASIONAL).astype(
        int
    )
    mask_uji = filter_buang(label_val)
    total_uji = int(mask_uji.sum())
    terdeteksi = int(pred_val[mask_uji].sum()) if total_uji > 0 else 0
    recall = (terdeteksi / total_uji * 100.0) if total_uji > 0 else 0.0
    print(f"{nama}: recall {recall:.2f}% ({terdeteksi:,}/{total_uji:,})")

def main():
  daftar_fitur = config.ambil_fitur()
  print(f"Membaca {config.PARQUET_TRAIN} dan {config.PARQUET_VAL}")
  df_train = pd.read_parquet(config.PARQUET_TRAIN)
  df_val = pd.read_parquet(config.PARQUET_VAL)
  fitur = config.pilih_kolom(df_train.columns, daftar_fitur)
  evaluasi_multi_seed(df_train, df_val, fitur)
  evaluasi_isolasi_keluarga(df_train, df_val, fitur)

if __name__ == "__main__":
  main()