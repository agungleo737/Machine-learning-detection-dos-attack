import argparse
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
import config
from inference.predictor import PrediktorDoS
from firewall.policy import terapkan_kebijakan
from firewall.adapter import jalankan_aksi_firewall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", required=True, help="Path file CSV trafik yang akan diuji")
    parser.add_argument("model", default=config.ARTEFAK_MODEL, help="Path model LightGBM")
    parser.add_argument("limit", type=int, default=10000, help="Batas jumlah baris CSV yang dibaca (0 = semua)")
    parser.add_argument("output", default="hasil_dry_run_firewall.csv", help="File CSV hasil simulasi")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File input tidak ditemukan: {args.input}")
    nrows = args.limit if args.limit and args.limit > 0 else None
    print(f"Membaca data trafik dari: {args.input}")
    df_raw = pd.read_csv(args.input, nrows=nrows, low_memory=False)
    prediktor = PrediktorDoS(model_path=args.model)
    df_pred, info = prediktor.prediksi_df(df_raw)
    df_final = terapkan_kebijakan(df_pred)
    ip_blokir = df_final.loc[df_final["aksi_firewall"] == "BLOCK", "src_ip"].tolist()
    catatan_fw = jalankan_aksi_firewall(ip_blokir, dry_run=True)
    df_final.to_csv(args.output, index=False)
    jml_block = int((df_final["aksi_firewall"] == "BLOCK").sum())
    jml_alert = int((df_final["aksi_firewall"] == "ALERT").sum())
    jml_allow = int((df_final["aksi_firewall"].str.startswith("ALLOW")).sum())
    print(f"Total baris diproses : {info['total_baris']:,} (NaN diisi 0: {info['nan_diisi_nol']:,}, Inf diisi 0: {info['inf_diisi_nol']:,})")
    print(f"Model yang dipakai   : {args.model} (SHA-256: {prediktor.model_sha256[:16]}...)")
    print(f"Keputusan Firewall   : BLOCK={jml_block:,} | ALERT={jml_alert:,} | ALLOW={jml_allow:,}")
    print(f"IP unik disimulasikan: {len(catatan_fw):,} IP")

    if "label" in df_final.columns:
        s = df_final["label"].astype(str).str.strip()
        mask_scope = s.str.startswith("DoS-") | s.str.startswith("DDoS-") | (s == "BenignTraffic")
        if mask_scope.any():
            df_s = df_final.loc[mask_scope]
            y_true = (df_s["label"] != "BenignTraffic").astype(int)
            y_pred = df_s["prediksi"].astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
            print(f"Verifikasi Label CSV : Serangan terdeteksi {tp:,}/{tp + fn:,} ({rec:.2f}%), FP Benign {fp:,}/{fp + tn:,}")
    print(f"Hasil lengkap disimpan di: {args.output}")

if __name__ == "__main__":
    main()