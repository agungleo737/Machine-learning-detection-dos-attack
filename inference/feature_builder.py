import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
import config

def bersihkan_dataframe(df):
    df = df.copy()
    df.columns = config.rapikan_kolom(df.columns)
    fitur_target = config.ambil_fitur()
    kolom_fitur = config.pilih_kolom(df.columns, fitur_target)
    df_num = df[kolom_fitur].apply(pd.to_numeric, errors="coerce")
    arr = df_num.to_numpy(dtype=np.float64, copy=False)
    jml_nan = int(np.isnan(arr).sum())
    jml_inf = int(np.isinf(arr).sum())
    baris_rusak = int((np.isnan(arr) | np.isinf(arr)).any(axis=1).sum())
    X_bersih = df_num.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    X_bersih.columns = fitur_target
    meta = pd.DataFrame(index=df.index)
    if "src_ip" in df.columns:
        meta["src_ip"] = df["src_ip"].astype(str).str.strip()
    elif "Src IP" in df.columns:
        meta["src_ip"] = df["Src IP"].astype(str).str.strip()
    else:
        meta["src_ip"] = [f"flow_{i}" for i in range(len(df))]
    if "label" in df.columns:
        meta["label"] = df["label"].astype(str).str.strip()
    info = {
        "total_baris": len(df),
        "jml_fitur": len(fitur_target),
        "nan_diisi_nol": jml_nan,
        "inf_diisi_nol": jml_inf,
        "baris_terdampak": baris_rusak,
    }
    return X_bersih, meta, info