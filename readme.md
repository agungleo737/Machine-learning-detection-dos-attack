# Pipeline Deteksi Serangan DoS/DDoS pada CIC-IoT-2023 (LightGBM)

Repositori ini berisi implementasi pipeline *machine learning* untuk mendeteksi trafik serangan **DoS** dan **DDoS** terhadap trafik normal (**BenignTraffic**) pada dataset **CIC-IoT-2023**. Arsitektur pipeline dirancang untuk mencegah kebocoran data (*data leakage*) melalui pembagian berbasis partisi file, deduplikasi *hash* 64-bit lintas-*split*, audit konflik label, dan penguncian evaluasi *test set* satu kali jalan.

## Ringkasan Dataset & Prapemrosesan

Dataset bersumber dari 47 file partisi CSV CIC-IoT-2023 (`part-00000` hingga `part-00047`, tanpa `part-00022`). Pemfilteran ruang lingkup hanya mempertahankan label `DoS-*`, `DDoS-*`, dan `BenignTraffic`.

| Tahap Pemrosesan | Jumlah Baris | Rincian |
| :--- | ---: | :--- |
| Total baris mentah (47 partisi CSV) | 12.138.459 | Seluruh kelas trafik CIC-IoT-2023 |
| Baris di luar ruang lingkup | 912.897 | Kelas non-DoS/DDoS dibuang |
| Baris dalam ruang lingkup (*in-scope*) | 11.225.562 | 285.989 *benign* (2,55%) dan 10.939.573 serangan (97,45%) |
| Train sebelum dedup (33 partisi) | 7.603.537 | 193.648 *benign*, 7.409.889 serangan |
| Validation sebelum dedup (7 partisi) | 1.889.767 | 48.158 *benign*, 1.841.609 serangan |
| Test sebelum dedup (7 partisi) | 1.732.258 | 44.183 *benign*, 1.688.075 serangan |
| **Train setelah dedup** | **5.418.791** | Bebas duplikat internal (0 konflik label) |
| **Validation setelah dedup** | **1.147.386** | 48.155 *benign*, 1.099.231 serangan (bebas bocor dari Train) |
| **Test setelah dedup** | **1.028.014** | 44.183 *benign*, 983.831 serangan (bebas bocor dari Train & Val) |

Model dilatih menggunakan **44 fitur** (`fitur_terpilih_44.json`), yaitu 46 fitur kanonik CIC-IoT-2023 dikurangi `rst_count` dan `urg_count` untuk menghindari ketergantungan semu pada artefak *flag* TCP tertentu.

## Struktur File Proyek

* `config.py` — Konfigurasi terpusat untuk path dataset, daftar 46/44 fitur, normalisasi nama kolom, dan parameter LightGBM.
* `fitur_terpilih_44.json` — Daftar 44 fitur input yang digunakan saat pelatihan dan evaluasi.
* `froze_dataset.py` — Menyaring kelas `DoS-*`, `DDoS-*`, dan `BenignTraffic`, mengecek nilai `NaN`/`Inf`, serta membagi 47 CSV secara berurutan (33 Train, 7 Val, 7 Test) ke file Parquet.
* `dedup.py` — Menghitung *hash* 64-bit pada 46 fitur, memverifikasi ketiadaan konflik label internal maupun lintas-*split*, dan menghapus duplikat lintas-*split*.
* `train.py` — Melatih model LightGBM (105 iterasi, `learning_rate=0.05`, `num_leaves=31`, `is_unbalance=True`, `seed=42`), menyimpan artefak model beserta SHA-256, dan mencetak evaluasi pada *validation set*.
* `tes_resmi.py` — Menguji model final satu kali pada *test set* beku (`dataset_frozen_test_dedup.parquet`), menghitung Wilson 95% *Confidence Interval* untuk FPR, dan mengunci hasil di `audit_tes_resmi_locked.json`.
* `evaluasi_baseline.py` — Mengevaluasi aturan sederhana `rst_count == 0` sebagai pembanding dasar.
* `baseline_decision.py` — Mengevaluasi pembanding Decision Tree (kedalaman 2 dan 3) dengan pemisahan pemilihan parameter di *validation set*.
* `uji_isolasi_multi_seed.py` — Menguji stabilitas *recall* sub-tipe serangan pada 3 *seed* (`42, 100, 2023`) serta pengujian *Leave-One-Family-Out* (LOFO) untuk keluarga protokol ICMP dan UDP di *validation set*.
* `prediksi_log_eksternal.py` — Menjalankan inferensi model pada file CSV trafik eksternal.

## Hasil Evaluasi Model

### 1. Performa pada Validation Set (1.147.386 Baris)

* **Recall Serangan:** `99,9991%` (10 lolos dari 1.099.231 sampel serangan)
* **False Positive (Benign):** `1` dari 48.155 sampel *benign* (`0,0021%`)
* **ROC-AUC / PR-AUC:** `1,0000000` / `1,0000000`
* **SHA-256 Model (`model_lgb_final_44fitur.txt`):** `023c531753d4a79e24865b00555fdb243f241a67b2c4f18297c7b45a3e38e09c`

### 2. Perbandingan pada Test Set Beku (1.028.014 Baris)

| Model / Metode | Recall Serangan | False Positive (Benign) | FPR (%) | Wilson 95% CI (FPR) | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| Baseline (`rst_count == 0`) | 69,73% | 238 / 44.183 | 0,5387% | 0,47% – 0,61% | - | - |
| Decision Tree (Depth 2) | 97,24% | 414 / 44.183 | 0,9370% | - | - | - |
| Decision Tree (Depth 3) | 98,61% | 257 / 44.183 | 0,5817% | - | - | - |
| **LightGBM (44 Fitur, 105 Iterasi)** | **99,9987%** | **6 / 44.183** | **0,0136%** | **0,0062% – 0,0296%** | **0,9999992** | **0,9999999** |

### 3. Temuan Diagnostik & Batas Kemampuan Model

* **Stabilitas Lintas-Seed (`DoS-UDP_Flood`):** Pengujian pada 3 *seed* (`42, 100, 2023`) menunjukkan seluruh sub-tipe serangan mencapai *recall* `>99%`, kecuali `DoS-UDP_Flood` yang berada di kisaran `84,99% – 85,50%`. Sampel `DoS-UDP_Flood` yang lolos memiliki distribusi fitur yang sangat mirip dengan trafik *benign* (probabilitas prediksi `< 0,10`).
* **Isolasi Keluarga Serangan (LOFO):** Saat sub-tipe `DDoS-ICMP_Fragmentation` dihapus sendiri dari data latih, model tetap mendeteksinya dengan *recall* `99,08%` karena belajar dari varian serangan ICMP lainnya. Namun ketika seluruh keluarga protokol ICMP dihapus dari data latih (LOFO ICMP), kemampuan deteksi terhadap serangan ICMP menurun tajam.

## Cara Menjalankan Pipeline

1. Pasang seluruh dependensi Python:
   ```bash
   pip install -r requirements.txt
   ```
2. Tempatkan 47 file partisi CSV CIC-IoT-2023 di dalam folder `datasetsih/`, lalu jalankan skrip secara berurutan:
   ```bash
   python froze_dataset.py
   python dedup.py
   python baseline_decision.py --split val
   python uji_isolasi_multi_seed.py
   python train.py
   python tes_resmi.py --model model_lgb_final_44fitur.txt
   python evaluasi_baseline.py
   python baseline_decision.py --split test --class-weight none
   ```
