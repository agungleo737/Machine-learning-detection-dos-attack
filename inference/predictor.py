import os
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
import config
from inference.feature_builder import bersihkan_dataframe


class PrediktorDoS:
    def __init__(self, model_path=None, threshold=None):
        self.model_path = model_path or config.ARTEFAK_MODEL
        self.threshold = config.THRESHOLD_OPERASIONAL if threshold is None else float(threshold)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"File model tidak ditemukan: {self.model_path}")
        self.fitur = config.ambil_fitur()
        self.model = lgb.Booster(model_file=self.model_path)
        self.model_sha256 = config.hitung_sha256(self.model_path)
    def prediksi_df(self, df):
        X_bersih, meta, info = bersihkan_dataframe(df)
        prob = self.model.predict(X_bersih[self.fitur])
        pred = (prob >= self.threshold).astype(np.int8)
        hasil = meta.copy()
        hasil["prob_serangan"] = np.round(prob, 6)
        hasil["prediksi"] = pred
        hasil["status_deteksi"] = np.where(pred == 1, "SERANGAN", "NORMAL")
        return hasil, info