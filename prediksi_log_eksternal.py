import argparse
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import config

def main():
    parser = argparse.ArgumentParser(description="Deteksi DDoS/DoS pada file CSV log trafik eksternal")
    parser.add_argument("--input", type=str, required=True, help="Path ke file CSV log trafik yang ingin diuji")
    parser.add_argument("--output", type=str, default="hasil_deteksi_firewall.csv", help="File output hasil prediksi")
    args = parser.parse_args()
    print("deteksi ddos/dos pada traffict eksternal")
    fitur_wajib = json.load(open(config.ARTEFAK_FITUR, encoding="utf-8"))
    booster = lgb.Booster(model_file=config.ARTEFAK_MODEL)
    print(f"Membaca file log eksternal: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Total baris trafik terbaca: {len(df):,} baris")

    fitur_hilang = [f for f in fitur_wajib if f not in df.columns]
    if fitur_hilang:
        print(f"\nFile CSV tidak memiliki {len(fitur_hilang)} fitur yang dibutuhkan model:")
        print(f"Fitur yang kurang: {fitur_hilang[:10]}")
        print("Pastikan log firewall/PCAP sudah diekstrak menjadi format 44 fitur statistik aliran.")
        return
    X_ext = df[fitur_wajib].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).to_numpy()
    t0 = time.perf_counter()
    prob_serangan = booster.predict(X_ext, num_iteration=config.NUM_ITERATIONS)
    durasi = time.perf_counter() - t0
    pred_label = np.where(prob_serangan >= config.THRESHOLD_OPERASIONAL, "ATTACK (DDoS/DoS)", "BENIGN (Normal)")
    jml_serangan = int((prob_serangan >= config.THRESHOLD_OPERASIONAL).sum())
    jml_normal = len(df) - jml_serangan
    print("ringkasan hasil")
    print(f"Waktu Pemindaian       : {durasi:.4f} detik ({len(df)/max(durasi, 1e-6):,.0f} aliran/detik)")
    print(f"Terdeteksi NORMAL      : {jml_normal:,} aliran ({(jml_normal/len(df))*100:.2f}%)")
    print(f"Terdeteksi SERANGAN    : {jml_serangan:,} aliran ({(jml_serangan/len(df))*100:.2f}%)")
    if "label" in df.columns:
        print("\nKolom 'label' ditemukan Rincian hasil prediksi per jenis trafik asli:")
        df["Prediksi_Model"] = pred_label
        df["Probabilitas_Serangan"] = np.round(prob_serangan, 4)
        ringkasan = df.groupby(["label", "Prediksi_Model"]).size().unstack(fill_value=0)
        print(ringkasan)
    df_out = pd.DataFrame({
        "Probabilitas_Serangan": np.round(prob_serangan, 6),
        "Status_Keputusan": pred_label
    })
    df_out.to_csv(args.output, index=False)
    print(f"\nHasil deteksi per baris disimpan ke: {args.output}")

if __name__ == "__main__":
    main()