# Catatan skor terbaik DICE FEST 2026: 0,25617

Dicatat pada 15 Oktober 2026.

## 1. Hasil yang sudah diperoleh

Perubahan terakhir adalah mengganti prediksi **listrik** dari baseline v5 dengan blend **`D_1_nnls`** yang sudah dihitung pada run v5. Prediksi diperoleh dengan membaca ulang cache, tanpa training model atau fitting ulang bobot blend.

| Versi | File submission yang terlihat di Kaggle | Public score |
|---|---|---:|
| Baseline v5 | `submission_v5.csv` | 0,25681 |
| V5 dengan listrik NNLS | `submission_v5_electric_nnls_1.csv` | **0,25617** |

**Penurunan public score: 0,00064.** Semakin kecil skor, semakin baik.

Skor 0,25617 dikonfirmasi dari tangkapan layar Kaggle yang dikirim pengguna. Notebook tambahan menghasilkan nama `submission_v5_electric_nnls.csv`; nama yang terlihat pada submission berstatus selesai adalah `submission_v5_electric_nnls_1.csv`.

## 2. Kondisi baseline sebelum perubahan

Sumber: kode dan output tersimpan pada `v5-ensemble.ipynb`.

| Target | Komponen final baseline |
|---|---|
| `target_macro_gas_kwh` | Prediksi EUI dikalikan `bld_dim_06` |
| `target_macro_elec_kwh` | `A_electric`, LightGBM dengan fitur S6 |
| `target_eui_gas_kwh_per_sqm` | `A_eui`, LightGBM dengan fitur S7 dan bobot luas |
| `target_clean_building_ratio` | Aturan clean dan model sakelar dari v5 |
| `target_electric_heating_adoption` | `A_adoption_residual`, ditambah lookup subzone untuk test |

Baseline memakai lima fold `GroupKFold` berdasarkan `subzone_id`. Prediksi test model dasarnya dirata-ratakan dari lima fold. Konfigurasi LightGBM dasarnya memakai seed 42, batas 1.500 pohon, learning rate 0,05, dan early stopping 80.

Output pemilihan akhir tahap D masih memilih `A_electric` dan `A_eui`. Tahap keragaman model C dan Optuna belum dijalankan pada run yang diperiksa: `RUN_C=False` dan `RUN_OPTUNA=False`.

Adoption residual sebelumnya sudah memperbaiki NRMSE OOF adoption dari sekitar 0,136715 menjadi 0,117627. Komponen ini diteruskan pada kandidat terbaik.

## 3. Temuan yang mendasari eksperimen

Tahap D ternyata sudah menghasilkan kandidat listrik yang lebih baik daripada model listrik final, tetapi kandidat tersebut tidak lolos ambang penerimaan otomatis.

| Kandidat | NRMSE listrik OOF | CV gabungan tanpa lookup |
|---|---:|---:|
| Baseline final v5 | 0,326365 | 0,276760 |
| `D_1_mean` | 0,321446 | 0,275776 |
| **`D_1_nnls`** | **0,321378** | **0,275763** |

Untuk `D_1_nnls`, output evaluasi menunjukkan:

- Membaik pada **5 dari 5 fold**.
- Rata-rata perbaikan CV per fold sekitar **0,000998**.
- Simpangan baku perbaikan antar-fold sekitar **0,000064**.
- Status otomatis: **`accepted=False`**.

Aturan v5 meminta minimal empat fold membaik dan rata-rata gain setidaknya `max(5 × simpangan baku, 0,001)`. Kandidat ini sedikit di bawah batas 0,001. Karena itu, kandidat diekspor secara terpisah untuk eksperimen; aturan penerimaan pada notebook asli tidak diubah.

Rata-rata gain per fold dapat sedikit berbeda dari selisih CV OOF seluruh baris karena cara agregasinya berbeda.

## 4. Isi blend listrik yang dipakai

Blend `D_1_nnls` menggabungkan enam prediksi listrik dari run yang sama. Bobot berikut adalah bobot final untuk prediksi test, dibulatkan enam desimal.

| Model | Fitur | Target saat training | Bobot sampel | Bobot blend |
|---|---|---|---|---:|
| `A_electric` | S6 | Listrik total | Seragam | 0,106224 |
| `B_direct_S7_density` | S7_density | Listrik total | Seragam | 0,200482 |
| `B_unit_S6_units` | S6 | Listrik per unit | Jumlah unit | 0,100489 |
| `B_unit_S7_units` | S7_density | Listrik per unit | Jumlah unit | 0,267964 |
| `B_unit_S7_sqrt_units` | S7_density | Listrik per unit | Akar jumlah unit | 0,191417 |
| `B_unit_S7_area` | S7_density | Listrik per unit | Luas | 0,133424 |

Jumlah unit menggunakan `bld_dim_12`; luas menggunakan `bld_dim_06`. Prediksi model per unit sudah dikalikan kembali dengan jumlah unit sebelum disimpan sebagai prediksi listrik total dalam cache.

Bobot blend diperoleh dengan non-negative least squares, lalu dinormalisasi agar jumlahnya satu. Pada evaluasi blend di v5, bobot setiap fold dipelajari dari baris OOF di luar fold yang dinilai. Untuk test, bobot dipelajari dari seluruh OOF. Ini memakai OOF dasar yang sudah ada, bukan evaluasi nested training tambahan.

Kandidat yang dipakai adalah **`D_1_nnls`**, tanpa tambahan artefak v2 yang menggunakan fold zona.

## 5. Implementasi yang dibuat

Notebook pendamping: **`v5-ensemble_cached_blend.ipynb`**. File asli `v5-ensemble.ipynb` tidak diedit.

Notebook pendamping berisi 12 sel, dengan enam sel kode yang semuanya diawali `#tambah`:

| Sel | Isi |
|---:|---|
| 2 | Import, konfigurasi sumber cache, dan fungsi pembacaan |
| 4 | Memuat baseline dan blend dari pasangan artefak yang sesuai |
| 6 | Mencocokkan CSV, urutan baris, label, luas, dan fold |
| 8 | Mengganti listrik dan menerapkan lookup adoption pada test |
| 10 | Menyimpan submission kandidat, prediksi, dan manifest |
| **12** | **RMSE/NRMSE per target, CV gabungan, dan perbandingan fold** |

Alur pemakaian cache:

1. Cari output v5 di `/kaggle/working/outputs/dicefest_v5/full`, atau di bawah `/kaggle/input` bila output sudah ditambahkan sebagai Input.
2. Baca `D_1_nnls_fc2f47200a6e7333.npz` beserta metadata JSON-nya.
3. Temukan pasangan `D_final_*.npz` dan JSON yang cocok melalui signature metadata.
4. Baca fold dan manifest data asli. Hash CSV dicocokkan untuk menjaga kesesuaian data serta urutan baris.
5. Salin prediksi baseline, lalu ganti hanya kolom listrik dengan prediksi NNLS dari cache.
6. Terapkan batas minimum listrik sesuai v5: minimum bagian latih untuk OOF setiap fold dan minimum seluruh train untuk test.
7. Terapkan lookup adoption pada test dan ekspor kandidat.

Empat kolom prediksi dasar lainnya diperiksa agar tetap identik dengan baseline. Hubungan gas = EUI × luas tetap dipertahankan.

File yang diperlukan dari output v5:

- `D_1_nnls_fc2f47200a6e7333.npz` dan `.json`.
- Pasangan `D_final_*.npz` dan `.json` yang cocok.
- `cache/folds.npz`.
- `cache/data_manifest.json`.
- `reproduction.json`.

Data kompetisi yang diperlukan: `train.csv`, `test.csv`, dan `sample_submission.csv`. File model LightGBM tidak diperlukan untuk alur ini. Cache yang hilang menyebabkan proses berhenti dengan pesan, tanpa training otomatis.

## 6. Lookup adoption yang dipertahankan

Untuk subzone train yang nilai adoption-nya konsisten, nilai rata-rata adoption dipetakan ke baris test dengan `subzone_id` yang sama. Konsistensi menggunakan batas selisih maksimum dan minimum sebesar `1e-5`, mengikuti v5.

Cakupannya **169.008 dari 183.987 baris test**, atau sekitar **91,8587%**. Baris test di luar cakupan tetap menggunakan prediksi model adoption residual.

Lookup hanya diterapkan pada test. OOF dinilai tanpa lookup dari seluruh train agar label validasi tidak dipakai untuk mengisi prediksinya sendiri.

## 7. Metrik dan cara membaca hasilnya

Untuk setiap target `j`:

```text
RMSE_j  = sqrt(mean((prediksi_oof_j - label_train_j)^2))
NRMSE_j = RMSE_j / sigma_j
CV      = mean(NRMSE untuk kelima target)
```

| Target | Pembagi sigma | NRMSE baseline | NRMSE kandidat |
|---|---:|---:|---:|
| Gas | 151750,04 | 0,257057 | 0,257057 |
| Listrik | 41725,37 | 0,326365 | **0,321378** |
| EUI | 40,65 | 0,673024 | 0,673024 |
| Clean | 12,49 | 0,009729 | 0,009729 |
| Adoption | 7,18 | 0,117627 | 0,117627 |
| **CV gabungan** | — | **0,276760** | **0,275763** |

Angka OOF di atas berasal dari output tahap D yang tersimpan di `v5-ensemble.ipynb`, dibulatkan enam desimal. Tabel RMSE dari eksekusi notebook pendamping belum dikirim kembali oleh pengguna; angka tersebut tidak diklaim sebagai hasil perhitungan ulang di komputer lokal.

Sel terakhir notebook pendamping menghitung RMSE mentah dan NRMSE dari cache, lalu menampilkan delta kandidat minus baseline. Delta negatif berarti membaik. RMSE mentah kelima target tidak dirata-ratakan karena satuannya berbeda.

**CV tanpa lookup berbeda dari public score Kaggle.** Label test tidak tersedia untuk menghitung RMSE public secara lokal. Perbaikan OOF sekitar 0,001 pada eksperimen ini diikuti perbaikan public 0,00064; hubungan tersebut tidak dianggap sebagai faktor konversi tetap.

## 8. Berkas hasil notebook pendamping

Direktori tujuan: `/kaggle/working/outputs/v5_cached_blend`.

| Berkas | Isi |
|---|---|
| `submission_v5_electric_nnls.csv` | Kandidat listrik NNLS dengan lookup adoption |
| `submission_v5_electric_nnls_nolookup.csv` | Kandidat pembanding tanpa lookup adoption |
| `predictions_cached_blend.npz` | OOF baseline, OOF kandidat, test kandidat dengan/tanpa lookup, ID, fold, dan urutan target |
| `reuse_manifest.json` | Sumber cache, metadata blend, hash, dan konfigurasi pemakaian ulang |
| `column_wise_rmse_nrmse.csv` | RMSE/NRMSE per target serta delta |
| `oof_cv_summary.csv` | CV gabungan baseline dan kandidat |
| `oof_fold_comparison.csv` | Perbandingan CV dan NRMSE listrik pada lima fold |

Notebook pendamping tidak mengganti `/kaggle/working/submission.csv`. Kandidat dipilih sebagai file submission terpisah.

## 9. Status pekerjaan dan catatan lanjutan

- Notebook pendamping sudah lolos pemeriksaan struktur `nbformat`, parsing sintaks Python, dan pemeriksaan nama variabel dengan Pyflakes: nol temuan.
- Sel notebook pendamping tidak dieksekusi secara lokal karena cache berada di sesi Kaggle. Hasil public 0,25617 kemudian dilaporkan pengguna melalui tangkapan layar Kaggle.
- Pengguna kemudian melaporkan output hilang. Pemeriksaan pemulihan dihentikan atas permintaan pengguna; ketersediaan cache saat ini belum diverifikasi.
- Rencana menaikkan batas pohon EUI menjadi 3.000, menjalankan tahap C, dan tuning Optuna belum menjadi bagian dari eksperimen yang menghasilkan 0,25617.

**Baseline terbaik yang dicatat untuk eksperimen berikutnya adalah 0,25617.** Catatan ini menjelaskan perubahan dan hasilnya; prediksi cache tetap diperlukan untuk menghitung ulang metrik tanpa training.
