import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

WHITELIST_IP = {
    "127.0.0.1",
    "0.0.0.0",
    "localhost",
    "::1",
    "192.168.1.1",
}

BATAS_WASPADA = config.THRESHOLD_OPERASIONAL
BATAS_BLOKIR = 0.85


def tentukan_aksi(prob_serangan, src_ip=None, batas_waspada=BATAS_WASPADA, batas_blokir=BATAS_BLOKIR):
    if src_ip and str(src_ip).strip() in WHITELIST_IP:
        return "ALLOW_WHITELIST"
    skor = float(prob_serangan)
    if skor >= batas_blokir:
        return "BLOCK"
    if skor >= batas_waspada:
        return "ALERT"
    return "ALLOW"


def terapkan_kebijakan(df_prediksi, batas_waspada=BATAS_WASPADA, batas_blokir=BATAS_BLOKIR):
    hasil = df_prediksi.copy()
    daftar_aksi = [
        tentukan_aksi(prob, ip, batas_waspada=batas_waspada, batas_blokir=batas_blokir)
        for prob, ip in zip(hasil["prob_serangan"], hasil["src_ip"])
    ]
    hasil["aksi_firewall"] = daftar_aksi
    return hasil