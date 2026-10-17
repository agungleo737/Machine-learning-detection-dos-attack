from datetime import datetime
import os
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import config


def cek_path_model(path_utama):
    if not os.path.exists(path_utama):
        return path_utama
    nama, ekstensi = os.path.splitext(path_utama)
    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_baru = f"{nama}_v_{waktu}{ekstensi}"
    print(f"File {path_utama} sudah ada, model baru disimpan ke {path_baru}")
    return path_baru
def main():
    daftar_fitur = config.ambil_fitur()
    print(f"Membaca {config.PARQUET_TRAIN} dan {config.PARQUET_VAL}")
    df_train = pd.read_parquet(config.PARQUET_TRAIN)
    df_val = pd.read_parquet(config.PARQUET_VAL)
    fitur = config.pilih_kolom(df_train.columns, daftar_fitur)
    print(f"Baris train: {len(df_train):,} | Baris val: {len(df_val):,} | Fitur: {len(fitur)}")
    x_train, y_train = df_train[fitur], df_train["target"].values
    x_val, y_val = df_val[fitur], df_val["target"].values
    dtrain = lgb.Dataset(x_train, label=y_train, feature_name=fitur)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain, feature_name=fitur)

    model = lgb.train(
        params=config.LGB_PARAMS,
        train_set=dtrain,
        num_boost_round=config.NUM_ITERATIONS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(period=15)],
    )

    path_model = cek_path_model(config.ARTEFAK_MODEL)
    model.save_model(path_model)
    sha_model = config.hitung_sha256(path_model)

    print(f"Model disimpan di: {path_model}")
    print(f"SHA-256 model: {sha_model}")
    print(f"Perintah uji test: python tes_resmi.py --model {path_model}")

    prob_val = model.predict(x_val)
    pred_val = (prob_val >= config.THRESHOLD_OPERASIONAL).astype(int)
    tp = int(((pred_val == 1) & (y_val == 1)).sum())
    fn = int(((pred_val == 0) & (y_val == 1)).sum())
    fp = int(((pred_val == 1) & (y_val == 0)).sum())
    tn = int(((pred_val == 0) & (y_val == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    roc_auc = roc_auc_score(y_val, prob_val)
    pr_auc = average_precision_score(y_val, prob_val)
    print(f"Hasil validasi ({len(df_val):,} baris, seed {config.SEED}):")
    print(f"Recall serangan: {recall * 100:.4f}% ({fn:,} lolos dari {tp + fn:,})")
    print(f"FP benign: {fp:,} dari {fp + tn:,} ({fpr * 100:.4f}%)")
    print(f"ROC-AUC: {roc_auc:.7f} | PR-AUC: {pr_auc:.7f}")


if __name__ == "__main__":
    main()