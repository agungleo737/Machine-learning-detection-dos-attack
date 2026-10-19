import hashlib
import json
import os

RAW_DATA_DIR = "datasetsih"
RAW_GLOB_PATTERN = os.path.join(RAW_DATA_DIR, "*.csv")
SPLIT_MODE = "sequential"
CONFLICT_POLICY = "halt"

RAW_PARQUET_TRAIN = "dataset_frozen_train.parquet"
RAW_PARQUET_VAL = "dataset_frozen_val.parquet"
RAW_PARQUET_TEST = "dataset_frozen_test.parquet"
PARQUET_TRAIN = "dataset_frozen_train_dedup.parquet"
PARQUET_VAL = "dataset_frozen_val_dedup.parquet"
PARQUET_TEST = "dataset_frozen_test_dedup.parquet"
ARTEFAK_FITUR = os.path.join("models", "fitur_terpilih_44.json")
ARTEFAK_MODEL = os.path.join("models", "model_lgb_final_44fitur.txt")
ARTEFAK_FROZE_LOG = "audit_froze_summary.json"
ARTEFAK_DEDUP_LOG = "audit_dedup_summary.json"
ARTEFAK_TES_LOCK = "audit_tes_resmi_locked.json"

SEED = 42
THRESHOLD_OPERASIONAL = 0.5
NUM_ITERATIONS = 105

LGB_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "is_unbalance": True,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": 4,
}
SEMUA_FITUR = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration",
    "Rate", "Srate", "Drate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "DNS",
    "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP",
    "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std",
    "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight"
]
ALIAS_KOLOM = {
    "Protocol_Type": "Protocol Type",
    "Tot_sum": "Tot sum",
    "Tot_size": "Tot size",
}
FITUR = [f for f in SEMUA_FITUR if f not in ("rst_count", "urg_count")]


def rapikan_kolom(kolom_df):
    bersih = [str(c).strip() for c in kolom_df]
    return [ALIAS_KOLOM.get(c, c) for c in bersih]


def pilih_kolom(kolom_df, target_fitur):
    set_kolom = set(kolom_df)
    kebalikan_alias = {v: k for k, v in ALIAS_KOLOM.items()}
    hasil = []
    for f in target_fitur:
        if f in set_kolom:
            hasil.append(f)
        elif f in kebalikan_alias and kebalikan_alias[f] in set_kolom:
            hasil.append(kebalikan_alias[f])
        elif f in ALIAS_KOLOM and ALIAS_KOLOM[f] in set_kolom:
            hasil.append(ALIAS_KOLOM[f])
        else:
            raise KeyError(f"Kolom {f} tidak ditemukan di dataset.")
    return hasil


def ambil_fitur():
    if os.path.exists(ARTEFAK_FITUR):
        with open(ARTEFAK_FITUR, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["features"] if isinstance(data, dict) and "features" in data else data
    os.makedirs(os.path.dirname(ARTEFAK_FITUR), exist_ok=True)
    with open(ARTEFAK_FITUR, "w", encoding="utf-8") as f:
        json.dump(FITUR, f, indent=2)
    return FITUR


def hitung_sha256(path_file):
    h = hashlib.sha256()
    with open(path_file, "rb") as f:
        for blok in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blok)
    return h.hexdigest()