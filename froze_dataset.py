from datetime import datetime
import glob
import json
import os
import numpy as np
import pandas as pd
import config

def filter_label(seri_label):
    teks = seri_label.astype(str).str.strip()
    return teks.str.startswith("DoS-") | teks.str.startswith("DDoS-") | (teks == "BenignTraffic")
def bagi_file(daftar_csv, mode, seed):
    if mode == "sequential":
        return daftar_csv[:33], daftar_csv[33:40], daftar_csv[40:]
    if mode == "random":
        rng = np.random.default_rng(seed)
        urutan = np.arange(len(daftar_csv))
        rng.shuffle(urutan)
        train_f = [daftar_csv[i] for i in sorted(urutan[:33])]
        val_f = [daftar_csv[i] for i in sorted(urutan[33:40])]
        test_f = [daftar_csv[i] for i in sorted(urutan[40:])]
        return train_f, val_f, test_f
    raise ValueError(f"Mode split tidak sesuai: {mode}")

def main():
    daftar_csv = sorted(glob.glob(config.RAW_GLOB_PATTERN))
    if not daftar_csv:
        raise FileNotFoundError(f"File CSV tidak ditemukan di {config.RAW_GLOB_PATTERN}")
    print(f"Total file CSV: {len(daftar_csv)} (mode split: {config.SPLIT_MODE}, seed: {config.SEED})")
    file_train, file_val, file_test = bagi_file(daftar_csv, config.SPLIT_MODE, config.SEED)
    fitur = config.ambil_fitur()
    print(f"Daftar {len(fitur)} fitur tersimpan di {config.ARTEFAK_FITUR}")
    pembagian = [
        ("train", file_train, config.RAW_PARQUET_TRAIN),
        ("val", file_val, config.RAW_PARQUET_VAL),
        ("test", file_test, config.RAW_PARQUET_TEST),
    ]
    total_mentah = 0
    total_terpakai = 0
    total_nan = 0
    total_inf = 0
    total_baris_kosong = 0
    rekap_split = {}
    for nama_split, list_file, path_out in pembagian:
        kumpulan_df = []
        mentah_split = 0
        nan_split = 0
        inf_split = 0
        baris_kosong_split = 0
        catatan_file = []

        for no, path_csv in enumerate(list_file, start=1):
            df = pd.read_csv(path_csv, low_memory=False)
            df.columns = config.rapikan_kolom(df.columns)
            jml_awal = len(df)
            mentah_split += jml_awal
            df = df.loc[filter_label(df["label"])].copy()
            jml_filter = len(df)
            kolom_fitur = config.pilih_kolom(df.columns, config.SEMUA_FITUR)
            df_angka = df[kolom_fitur].apply(pd.to_numeric, errors="coerce")
            matriks = df_angka.to_numpy(dtype=np.float64, copy=False)
            mask_nan = np.isnan(matriks)
            mask_inf = np.isinf(matriks)
            jml_nan = int(mask_nan.sum())
            jml_inf = int(mask_inf.sum())
            jml_baris_kena = int((mask_nan | mask_inf).any(axis=1).sum())
            nan_split += jml_nan
            inf_split += jml_inf
            baris_kosong_split += jml_baris_kena

            df[kolom_fitur] = df_angka.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
            df["target"] = (df["label"].astype(str).str.strip() != "BenignTraffic").astype(np.int8)
            nama_file = os.path.basename(path_csv)
            df["partisi"] = nama_file
            kumpulan_df.append(df)
            catatan_file.append({
                "file": nama_file,
                "raw_rows": jml_awal,
                "in_scope_rows": jml_filter,
                "nan_cells": jml_nan,
                "inf_cells": jml_inf,
                "affected_rows": jml_baris_kena,
            })
            print(f"[{nama_split} {no}/{len(list_file)}] {nama_file}: {jml_filter:,} baris (NaN: {jml_nan:,}, Inf: {jml_inf:,})")
        df_gabung = pd.concat(kumpulan_df, ignore_index=True)
        df_gabung.to_parquet(path_out, index=False)

        total_mentah += mentah_split
        total_terpakai += len(df_gabung)
        total_nan += nan_split
        total_inf += inf_split
        total_baris_kosong += baris_kosong_split
        jml_benign = int((df_gabung["target"] == 0).sum())
        jml_serangan = int((df_gabung["target"] == 1).sum())
        rekap_split[nama_split] = {
            "num_files": len(list_file),
            "raw_rows": mentah_split,
            "in_scope_rows": len(df_gabung),
            "benign_rows": jml_benign,
            "attack_rows": jml_serangan,
            "nan_cells": nan_split,
            "inf_cells": inf_split,
            "affected_rows": baris_kosong_split,
            "files": catatan_file,
        }
        print(f"Selesai {nama_split}: {len(df_gabung):,} baris disimpan di {path_out} (benign: {jml_benign:,}, serangan: {jml_serangan:,})\n")
    ringkasan = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split_mode": config.SPLIT_MODE,
        "seed": config.SEED,
        "total_files": len(daftar_csv),
        "total_raw_rows": total_mentah,
        "total_out_of_scope_rows": total_mentah - total_terpakai,
        "total_in_scope_rows": total_terpakai,
        "total_nan_cells": total_nan,
        "total_inf_cells": total_inf,
        "total_affected_rows": total_baris_kosong,
        "splits": rekap_split,
    }
    with open(config.ARTEFAK_FROZE_LOG, "w", encoding="utf-8") as f:
        json.dump(ringkasan, f, indent=2)
    print(f"Total baris mentah: {total_mentah:,}")
    print(f"Total baris luar ruang lingkup: {total_mentah - total_terpakai:,}")
    print(f"Total baris terpakai: {total_terpakai:,}")
    print(f"Total NaN/Inf diisi 0: NaN={total_nan:,}, Inf={total_inf:,} ({total_baris_kosong:,} baris)")
if __name__ == "__main__":
    main()