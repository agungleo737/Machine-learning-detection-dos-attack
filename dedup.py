from datetime import datetime
import json
import sys
import numpy as np
import pandas as pd
import config

def buat_hash(df, kolom):
    return pd.util.hash_pandas_object(df[kolom], index=False)
def cek_konflik_split(df, seri_hash, kolom, nama_split):
    df_cek = pd.DataFrame({"hash": seri_hash.values, "target": df["target"].values})
    hitung_target = df_cek.groupby("hash", sort=False)["target"].nunique()
    kandidat = pd.Index(hitung_target.index[hitung_target > 1], dtype="uint64")

    if len(kandidat) == 0:
        print(f"Konflik internal {nama_split}: 0 grup (0 baris)")
        return {"kandidat_hash": 0, "grup_konflik": 0, "baris_konflik": 0}, pd.Index([], dtype="uint64")
    mask = seri_hash.isin(kandidat)
    sub = df.loc[mask, kolom + ["target"]].copy()
    sub["_hash"] = seri_hash.loc[mask].values
    cek_fitur = sub.groupby(kolom, sort=False, dropna=False)["target"].nunique()
    jml_grup = int((cek_fitur > 1).sum())
    if jml_grup > 0:
        df_konflik = cek_fitur[cek_fitur > 1].reset_index()[kolom]
        gabung = pd.merge(sub[["_hash"] + kolom], df_konflik, on=kolom, how="inner")
        hash_konflik = pd.Index(gabung["_hash"].unique(), dtype="uint64")
        jml_baris = int(seri_hash.isin(hash_konflik).sum())
    else:
        hash_konflik = pd.Index([], dtype="uint64")
        jml_baris = 0
    print(f"Konflik internal {nama_split}: {jml_grup:,} grup ({jml_baris:,} baris)")
    return {"kandidat_hash": int(len(kandidat)), "grup_konflik": jml_grup, "baris_konflik": jml_baris}, hash_konflik


def cek_konflik_antar_split(hash_a, target_a, nama_a, hash_b, target_b, nama_b):
    df_a = pd.DataFrame({"hash": hash_a.values, "target": target_a}).drop_duplicates()
    df_b = pd.DataFrame({"hash": hash_b.values, "target": target_b}).drop_duplicates()
    gabung = pd.merge(df_a, df_b, on="hash", suffixes=("_a", "_b"))
    beda = gabung[gabung["target_a"] != gabung["target_b"]]
    hash_konflik = pd.Index(beda["hash"].unique(), dtype="uint64")
    jml_grup = int(len(hash_konflik))
    jml_baris = int(pd.Index(hash_b.values, dtype="uint64").isin(hash_konflik).sum()) if jml_grup > 0 else 0
    print(f"Konflik {nama_a} vs {nama_b}: {jml_grup:,} hash ({jml_baris:,} baris di {nama_b})")
    return {"grup_konflik": jml_grup, "baris_di_split_kedua": jml_baris}, hash_konflik


def main():
    print("Membaca file parquet sebelum dedup")
    df_train = pd.read_parquet(config.RAW_PARQUET_TRAIN)
    df_val = pd.read_parquet(config.RAW_PARQUET_VAL)
    df_test = pd.read_parquet(config.RAW_PARQUET_TEST)
    df_train.columns = config.rapikan_kolom(df_train.columns)
    df_val.columns = config.rapikan_kolom(df_val.columns)
    df_test.columns = config.rapikan_kolom(df_test.columns)
    kolom_hash = config.pilih_kolom(df_train.columns, config.SEMUA_FITUR)
    print(f"Menghitung hash dari {len(kolom_hash)} fitur...")

    hash_train = buat_hash(df_train, kolom_hash)
    hash_val = buat_hash(df_val, kolom_hash)
    hash_test = buat_hash(df_test, kolom_hash)
    k_train, h_train = cek_konflik_split(df_train, hash_train, kolom_hash, "train")
    k_val, h_val = cek_konflik_split(df_val, hash_val, kolom_hash, "val")
    k_test, h_test = cek_konflik_split(df_test, hash_test, kolom_hash, "test")
    k_tr_val, h_tr_val = cek_konflik_antar_split(hash_train, df_train["target"].values, "train", hash_val, df_val["target"].values, "val")
    k_tr_test, h_tr_test = cek_konflik_antar_split(hash_train, df_train["target"].values, "train", hash_test, df_test["target"].values, "test")
    k_val_test, h_val_test = cek_konflik_antar_split(hash_val, df_val["target"].values, "val", hash_test, df_test["target"].values, "test")
    semua_konflik = h_train.union(h_val).union(h_test).union(h_tr_val).union(h_tr_test).union(h_val_test)
    total_konflik = len(semua_konflik)
    if total_konflik > 0:
        if config.CONFLICT_POLICY == "halt":
            log_gagal = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": "stopped_on_conflict",
                "total_conflicting_hashes": int(total_konflik),
                "conflicts": {
                    "train": k_train,
                    "val": k_val,
                    "test": k_test,
                    "train_val": k_tr_val,
                    "train_test": k_tr_test,
                    "val_test": k_val_test,
                },
            }
            with open(config.ARTEFAK_DEDUP_LOG, "w", encoding="utf-8") as f:
                json.dump([log_gagal], f, indent=2)
            print(f"Ditemukan {total_konflik:,} hash konflik. Proses dihentikan sesuai CONFLICT_POLICY='halt'.")
            sys.exit(1)
        elif config.CONFLICT_POLICY == "drop_conflicting_patterns":
            print(f"Membuang {total_konflik:,} hash konflik dari semua split")
            df_train = df_train.loc[~hash_train.isin(semua_konflik)].reset_index(drop=True)
            hash_train = buat_hash(df_train, kolom_hash)
            df_val = df_val.loc[~hash_val.isin(semua_konflik)].reset_index(drop=True)
            hash_val = buat_hash(df_val, kolom_hash)
            df_test = df_test.loc[~hash_test.isin(semua_konflik)].reset_index(drop=True)
            hash_test = buat_hash(df_test, kolom_hash)
        else:
            raise ValueError(f"CONFLICT_POLICY tidak dikenal: {config.CONFLICT_POLICY}")
    awal_train = len(df_train)
    mask_train = ~hash_train.duplicated(keep="first")
    df_train_bersih = df_train.loc[mask_train].reset_index(drop=True)
    index_train = pd.Index(hash_train.loc[mask_train].values, dtype="uint64")
    print(f"Train: {awal_train:,} -> {len(df_train_bersih):,} baris")

    awal_val = len(df_val)
    mask_val = (~hash_val.duplicated(keep="first")) & (~hash_val.isin(index_train))
    df_val_bersih = df_val.loc[mask_val].reset_index(drop=True)
    index_val = pd.Index(hash_val.loc[mask_val].values, dtype="uint64")
    print(f"Val  : {awal_val:,} -> {len(df_val_bersih):,} baris")
    awal_test = len(df_test)
    index_gabung = index_train.union(index_val)
    mask_test = (~hash_test.duplicated(keep="first")) & (~hash_test.isin(index_gabung))
    df_test_bersih = df_test.loc[mask_test].reset_index(drop=True)
    print(f"Test : {awal_test:,} -> {len(df_test_bersih):,} baris")

    df_train_bersih.to_parquet(config.PARQUET_TRAIN, index=False)
    df_val_bersih.to_parquet(config.PARQUET_VAL, index=False)
    df_test_bersih.to_parquet(config.PARQUET_TEST, index=False)
    test_benign = int((df_test_bersih["target"] == 0).sum())
    test_serangan = int((df_test_bersih["target"] == 1).sum())
    catatan = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "split_mode": config.SPLIT_MODE,
        "conflict_policy": config.CONFLICT_POLICY,
        "seed": config.SEED,
        "rows_before": {"train": awal_train, "val": awal_val, "test": awal_test},
        "rows_after": {
            "train": len(df_train_bersih),
            "val": len(df_val_bersih),
            "test": len(df_test_bersih),
            "test_benign": test_benign,
            "test_attack": test_serangan,
        },
        "conflicts": {
            "total_conflicting_hashes": int(total_konflik),
            "train": k_train,
            "val": k_val,
            "test": k_test,
            "train_val": k_tr_val,
            "train_test": k_tr_test,
            "val_test": k_val_test,
        },
    }
    riwayat = []
    if pd.io.common.file_exists(config.ARTEFAK_DEDUP_LOG):
        try:
            with open(config.ARTEFAK_DEDUP_LOG, "r", encoding="utf-8") as f:
                riwayat = json.load(f)
        except Exception:
            riwayat = []
    riwayat.append(catatan)
    with open(config.ARTEFAK_DEDUP_LOG, "w", encoding="utf-8") as f:
        json.dump(riwayat, f, indent=2)
    print(f"Data dedup disimpan: {config.PARQUET_TRAIN}, {config.PARQUET_VAL}, {config.PARQUET_TEST}")
    print(f"Rincian test: {len(df_test_bersih):,} baris (benign: {test_benign:,}, serangan: {test_serangan:,})")
if __name__ == "__main__":
    main()