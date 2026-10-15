# Audit v4 dan rencana perbaikan v5

## Ruang lingkup dan sumber

Audit tanggal 14 Oktober 2026. Notebook dibaca, tidak dieksekusi; tidak ada training atau pencarian hyperparameter lokal.

- Notebook: [dicefest2026_v4.ipynb](dicefest2026_v4.ipynb), 29 sel, 17 sel kode. Semua execution_count null dan outputs kosong.
- SHA256 v4 saat audit: A6261FBE62BE2ED1A4FA7FC9D7A050329CE6C54C89DCBE48D568961288C09D29.
- Pengguna kemudian memberikan ringkasan tahap F dan tabel hasil run Kaggle. Angka di bawah berasal dari pesan tersebut; bukan output tersimpan pada file lokal.
- Log: [hasil_eksplorasi2](<deskripsi dan eksplorasi/hasil_eksplorasi2>), khusus S6-07, S7-09, S5-05, SUB-05/SUB-08; [hasil_eksplorasi.md](<deskripsi dan eksplorasi/hasil_eksplorasi.md>); [panduan1](<deskripsi dan eksplorasi/panduan1>), khusus bagian 8.2 dan 13.1; deskripsi EDA.
- Kode tim features.py/stages.py: **tidak ada** dalam folder yang tersedia. Perbandingan baris demi baris dengan kode tim tidak dapat dilakukan.
- Artefak v2 ditemukan di dicefest_model, bukan dicefest__model. Dibaca tanpa diubah.
- Nomor sel dalam bukti dihitung mulai 1, termasuk Markdown. Nomor baris adalah baris kode di dalam sel, bukan baris JSON notebook.

FAKTA berarti terlihat pada kode, pembacaan file, atau angka yang dilaporkan pengguna. DUGAAN/TEBAKAN berarti hubungan sebab-akibat atau dampak yang belum diukur. Prediksi OOF/test v4, bobot blend, delta per fold, tabel SHAP, dan versi library run Kaggle belum tersedia.

## 1. Lokalisasi: CV notebook v4 vs rujukan tim

Metrik resmi adalah rata-rata lima RMSE/sigma, dengan sigma beku 151750,04; 41725,37; 40,65; 12,49; 7,18. Lebih kecil lebih baik. Referensi clean dan adopsi tim adalah angka pendekatan; selisih juga mengikuti pembulatan referensi.

| Target | S6 tim | S7 tim | v4 A, dilaporkan 6 desimal | v4 final, dari F | Delta final − S7 | Kontribusi delta ke gabungan |
|---|---:|---:|---:|---:|---:|---:|
| Gas | 0,260900 | 0,258700 | 0,257055 | 0,257054542 | −0,001645458 | −0,000329092 |
| Listrik | 0,327900 | 0,327900 | 0,326365 | 0,321342837 | −0,006557163 | −0,001311433 |
| EUI | 0,680100 | 0,676300 | 0,673020 | 0,673019626 | −0,003280374 | −0,000656075 |
| Clean | sekitar 0,010 | sekitar 0,010 | 0,009729 | 0,009728635 | −0,000271365 | −0,000054273 |
| Adopsi tanpa lookup | sekitar 0,118 | sekitar 0,118100 | 0,136715 | 0,136714820 | +0,018614820 | +0,003722964 |
| **Gabungan tanpa lookup** | **0,279400** | **0,278200** | **0,280577** | **0,279572092** | **+0,001372092** | **+0,001372092** |

**FAKTA:** adopsi adalah penyumbang kemunduran CV terhadap S7. Empat target lain memberi keuntungan gabungan sekitar 0,002350872, tetapi adopsi menambah 0,003722964. Data ini tidak mendukung dugaan awal bahwa fitur energi yang hilang adalah penyebab utama.

| Ukuran publik/lookup | Nilai |
|---|---:|
| S7 publik, log tim | 0,25939 |
| S6 publik, log tim | 0,26059 |
| v4 publik, laporan pemilik | 0,26088 |
| v4 − S7 publik | +0,00149 |
| v4 − S6 publik | +0,00029 |
| Perkiraan v4 dari CV − 0,01881 | 0,2607620921 |
| Publik aktual − perkiraan | +0,0001179079 |
| Cakupan lookup test | 0,9185866392734269 = 169.008 / 183.987 |
| Baris test tanpa lookup | 14.979 |

Cakupan lookup sesuai log. Skor adopsi sekitar 0,038 setelah lookup adalah estimasi pada komposisi test, **bukan** OOF GroupKFold subzone. CV adopsi 0,1367 menilai seluruh subzone yang ditahan; perbaikan model hanya mengubah 8,14% test yang tidak tercakup lookup. Karena itu kontribusi CV +0,003723 tidak dapat langsung dianggap sebagai kontribusi publik yang sama.

### Membaca tabel kandidat yang diberikan

- Listrik langsung A sekitar 0,326365 menjadi 0,321343 setelah NNLS: gain gabungan sekitar 0,001004. Kandidat D_1_nnls cocok dengan angka final.
- Rata-rata listrik D_1_mean mempunyai gain sekitar 0,000989 terhadap A, sedikit di bawah batas praktis 0,001. Tabel global tidak menggantikan pemeriksaan gain per fold.
- D_gas_nnls memiliki CV sekitar 0,279240, tetapi tambahan gain terhadap pilihan saat itu sekitar 0,279572 − 0,279240 = **0,000332**, bukan 0,001336. Kolom gain_vs_A memakai acuan A; keputusan gas memakai acuan saat itu. Tidak terpilihnya gas ini konsisten dengan batas 0,001.
- D_2_nnls_v2 menambah gain sekitar 0,000161 terhadap pilihan saat itu. Selain kecil, v2 nonaktif untuk final secara default.
- Skor gas/EUI/clean/adopsi final sama dengan A pada angka yang tersedia. Ini mendukung inferensi bahwa perubahan final berasal dari listrik; bobot dan prediksi v4 tidak tersedia untuk konfirmasi vektor.
- calibrated=false berasal dari A: CV A sekitar 0,280577 meleset 0,002377, dan pemeriksaan sel 8 memakai selisih absolut EUI yang membaik sekitar 0,003280. Flag tersebut berarti “tidak mereproduksi angka acuan dengan toleransi yang dipasang”, bukan bukti leakage atau submission rusak.

## 2. Tinjauan kode A–H

### A. Data dan fold

| Pemeriksaan | Bukti kode v4 | Penilaian / keyakinan |
|---|---|---|
| Ordinal dimensi | Sel 4 baris 39–42: High=0, Medium=1, Low=2, Very Low=3 | Pemetaan eksplisit; kategori asing menghasilkan error. Tinggi. |
| Urban/Rural | Sel 4: Rural=0, Urban=1 | Pemetaan eksplisit, konsisten train/test. Tinggi. |
| Missing dimensi 04/07 | Sel 4 membuat missing_col; NaN tetap dalam fitur tree | Native missing untuk GBDT, bukan imputasi global. MLP mengimputasi median dari fit fold pada sel 6. Tinggi. |
| Kolom dibuang | Sel 4 baris 93–95: nunique <= 1, ditambah env_climate_04 | Pembacaan 11 kolom yang disebut log mengonfirmasi semuanya konstan; 02 dan 04 iklim identik pada train. Tinggi. |
| GroupKFold | Sel 4 baris 64: groups=SUBZONES; baris 67 assert isdisjoint | Tidak ada random row KFold. GroupKFold ini tidak shuffle; seed 42 tidak mengacak fold. Tinggi. |
| Ukuran fold | Algoritme GroupKFold dan N_FOLDS=5 | Jumlah aktual tiap fold tidak disimpan pada pesan pengguna; belum dikonfirmasi. V5 mencetaknya. Sedang. |
| Urutan baris | Sel 4: iloc hanya untuk DEBUG; reset_index setelah posisi dipilih | Mode penuh tidak sort/merge baris. Agregat dipetakan kembali ke urutan asli. Tinggi. |
| ID submission | Sel 4 baris 27: sample_submission["id"].equals(test_raw["id"]) | Ada pemeriksaan urutan, unik, dan tanpa ID null. Tinggi. |

**Perbedaan hitungan fitur:** header aktual berisi 209 kolom, termasuk 5 target dan 3 ID, jadi 201 fitur. Menghapus 11 konstan + 1 duplikat menghasilkan **189**, bukan 190 fitur mentah bersih. Header memuat **43** bld_dim, meskipun salah satu ringkasan log menyebut 42. Angka 190 di log tidak dapat dipenuhi dengan menambah kolom buatan; mungkin ada definisi fitur atau snapshot berbeda, tetapi penyebab persisnya belum diketahui.

### B. Kesetiaan fitur S6/S7

| Keluarga di log | Implementasi v4 | Status |
|---|---|---|
| Rasio ukuran terhadap luas | Sel 4 baris 108–111: seluruh bld_dim numerik / AREA | Ada, termasuk rt_bld_dim_12. |
| Ukuran subzone | Sel 4 baris 112–117: count(train+test grup) | Ada; v4 juga menambah ukuran zona. |
| TE zona S6 | Sel 6: te_zone_eui dan te_zone_elec | Ada; dua kolom untuk listrik S6. |
| 66 selisih log dari 12 kolom | Sel 4 baris 120–127: combinations(POPULAR12, 2), signed-log1p | Ada; nama 12 kolom dan bentuk log persis tidak dapat dipastikan dari log. |
| 24 agregat zona | Sel 4 baris 129–136: mean serta selisih baris dari mean | Ada; pilihan kolom sama dengan POPULAR12 yang diasumsikan. |
| TE zona kedua | Sel 4 baris 156, sel 6 fungsi zone_stats | Ada: EUI berbobot luas, std EUI, listrik/luas. |
| S7 hanya untuk energi gas/EUI | Sel 8: EUI features="S7"; listrik features="S6" | Sesuai konfigurasi S7 dalam log. Listrik S7 baru diuji di B. |
| Rasio kepadatan eksperimen | Sel 4 baris 139–142: 06/12, 07/12, 08/12 | Ada; semantik 07/08 sebagai selubung masih proksi. |

**Tidak ada keluarga fitur wajib yang terbukti hilang.** Yang belum diketahui: daftar tepat 12 fitur, log vs log1p/penanganan nol, rasio mana yang dipilih tim, statistik zona persis, smoothing, jumlah inner fold, ddof std TE, dan parameter model. Semua memerlukan kode tim untuk reproduksi persis.

Perkiraan statis jumlah fitur, berdasarkan 11 konstan yang terkonfirmasi dan dua flag missing yang dilaporkan log:

| Set | Fitur sebelum TE | TE | Total input model |
|---|---:|---:|---:|
| Mentah bersih | 189 | 0 | 189 |
| raw v4: +2 flag +hvac_rest | 192 | 0 | 192 |
| S6: +42 rasio bld_dim numerik +2 ukuran grup | 236 | 2 | 238 |
| S7: +66 pairwise +24 agregat zona | 326 | 5 | 331 |
| S7_density: +3 kepadatan per unit | 329 | 5 | 334 |

Sel 4 baris 167 mencetak panjang FEATURE_SETS sebelum TE; itu kurang 2/5 dari jumlah model sebenarnya. Jumlah akhir run v4 belum dilampirkan; v5 mencetak jumlah termasuk TE.

### C. Leakage dan konsistensi train/test

- **FAKTA, keyakinan tinggi:** TE bersarang. Sel 6 baris 47–49 memanggil zone_stats(fit_idx[inner_fit]); baris inner-valid menerima statistik tanpa subzone-nya. Outer-valid memakai zone_stats(fit_idx). Tidak ditemukan label outer-valid masuk ke fit TE.
- Agregat zona dan count subzone memakai fitur train+test tanpa target (sel 4 baris 114, 129–134). Sesuai skema transduktif tanpa label yang dibolehkan dalam panduan; tidak ada pseudo-label test.
- Test setiap model fold memakai TE outer-train model itu (sel 6 baris 50–52), bukan satu TE seluruh train. Ini **konsisten untuk rata-rata model fold**, bukan bukti bug. Memaksakan TE seluruh train ke model fold mengubah distribusi input. TE seluruh train harus dipasangkan dengan model refit seluruh train.
- Kolom disusun dari daftar yang sama, float32. Zona tak terlihat diisi prior outer-train pada apply_zone_stats. V5 menambah assert kesamaan urutan kolom.
- Early stopping dan pemilihan banyak kandidat tetap memakai outer-validasi/OOF; hasil sedikit optimis. Cross-fitting bobot bukan validasi bersarang penuh untuk seluruh proses pemilihan kandidat.
- Data yang tersedia tidak membuktikan leakage. CV gabungan v4 justru lebih buruk daripada S7 akibat adopsi.

### D. Pipeline target

| Target | Bukti | Kesimpulan |
|---|---|---|
| Gas | Sel 8: A_OOF[:,0] = A_OOF[:,2] * AREA | Baseline benar; sel 19 baris 152–156 masih mengizinkan gas blend langsung mengganti formula. Ini cabang kode, bukan bukti dipakai pada run pengguna. V5 melarangnya untuk final. |
| EUI | Sel 8 baris 7 weight="area"; sel 6 baris 95 weights/weights.mean() | Bobot proporsional luas, rata-rata satu. Hasil v4 lebih baik daripada referensi EUI. |
| Clean | Sel 6 baris 313–329: 015>0; label deficit >=0,5×015; baris 346–349 threshold prediksi 0,5 | Aturan+klasifier+clip tersedia. Skor 0,009729 lebih baik daripada pendekatan 0,010; tidak ada dasar mengganti aturan. |
| Adopsi model | Sel 8 baris 8: spec_for("A_adoption",4,features="raw"), default mode="direct" | Model langsung memakai hvac_rest sebagai fitur, tetapi tidak belajar target residual hvac_rest−adopsi seperti rekomendasi panduan 8.2. Ini perbedaan representasi yang nyata; manfaat memperbaikinya masih hipotesis. |
| Lookup | Sel 26 baris 28: groupby seluruh SUBZONES/Y; hanya grup konsisten | Fallback tetap model FINAL_TEST, bukan nol/rata-rata. Cakupan run dilaporkan tepat 169.008 baris. |
| Persen | Sel 8 serta apply_bounds sel 19 | Clip [0,100] tersedia. |

V5 mempertahankan kontrol adopsi langsung, menambahkan residual adopsi dengan fitur/fold sama, dan hanya memilihnya setelah aturan >=4/5 fold serta mean gain >=max(5×std,0,001). Perbandingan segmen subzone kecil membantu mendiagnosis baris yang tidak tercakup lookup; tidak menggunakan label test atau mencoba menemukan kembali rumus deterministik H8.

### E. Model dan waktu training

- LightGBM lokal v4: 1.500 pohon, learning_rate 0,05, num_leaves 63, min_child_samples 30, subsample/colsample 0,85, L2=1, max_bin=127; seed 42, maksimal 8 thread (sel 2 baris 60–63). Parameter tim tidak tersedia.
- Sel 6 baris 173–176: early stopping 80 iterasi pada outer-validasi. Regresi menggunakan RMSE asli; model listrik per-unit mengalikan galat validasi dengan unit. Kecukupan 1.500 pohon tidak dapat dipastikan tanpa best_iteration; v5 mencetak cap dan indikasi mendekati cap.
- Bobot unit/akar unit/luas adalah pilihan empiris. Bobot unit tidak secara matematis identik dengan RMSE listrik total; padanan loss kuadrat total memerlukan unit². Notebook tidak perlu menambah varian unit² di luar permintaan.
- Prediksi test adalah rata-rata lima model (sel 6 baris 300/367). Tidak ada seed berbeda antar target inti; seed 43/44 hanya keragaman yang disengaja.
- XGBoost hist maksimum 500 pohon; CatBoost CPU 800 iterasi. C default False (sel 2 baris 29), dan run pengguna melaporkan C cache_only 0,040791 detik tanpa baris kandidat C pada tabel yang diberikan. Tidak ada bukti XGBoost menggeser waktu A/B pada run ini.
- Model gas langsung memakan sekitar 866,684520 detik dan tidak menunjukkan perbaikan sendiri atas gas turunan. V5 menjadikannya opsional/nonaktif untuk menghemat waktu.

### F. Blend dan artefak v2

- v2 hanya masuk V2_RESULTS/BLEND_POOL, tidak masuk STATIC_TRAIN atau fold_features. Tidak ditemukan stacking v2 sebagai fitur.
- Sel 19 baris 46–50: NNLS dipasang pada OOF fit_idx dan dinilai pada valid_idx; test menggunakan bobot semua OOF. Bobot nonnegatif dinormalisasi. NNLS kemudian dinormalisasi tidak sama persis dengan optimasi berkendala jumlah bobot satu.
- Korelasi residual >0,95 hanya dicetak di sel 11 baris 43; v4 belum mengeluarkan kandidat tersebut dan belum memiliki filter bobot <0,05. ALLOW_V2_FINAL=False secara default.
- Mean v2 merugikan pada hasil yang dilaporkan: listrik 0,322195 vs mean tanpa v2 0,321421; EUI 0,677303 vs baseline 0,673020. NNLS v2 listrik sama pada enam desimal dengan tanpa v2, tetapi bobot nol tidak dapat dinyatakan terbukti tanpa tabel bobot.
- V5 menyaring korelasi dan bobot pada bagian OOF latih setiap meta-fold, lalu ulang pada seluruh OOF untuk test. Alasan penolakan disimpan. Fold zona v2 tetap tidak sebanding sempurna.
- v4 menerima kandidat sebelum post-processing; efek post-processing baru dilaporkan di akhir. Tabel pengguna menunjukkan D_before_bounds dan D_final sama pada angka yang tersedia, jadi ini bukan penyebab terukur run tersebut. V5 menilai kandidat sesudah batas prediksi dan memeriksa keputusan final terhadap baseline yang dibatasi.
- Seed average dan anggota individual bisa masuk mean bersama pada v4 jika C tersedia. V5 mengeluarkan anggota dari kumpulan blend ketika average-nya dipakai; model individu tetap boleh menjadi pembanding tunggal.

### G. Submission

Sel 26 memeriksa nama/urutan header, 183.987 baris, ID unik dan urutan TEST_IDS, finite, persen [0,100], positif/minimum latih. Kesetaraan TEST_IDS dengan sample_submission sudah diperiksa sel 4. source=paths["lookup"] disalin ke submission.csv; bukan cadangan v2. Submission lama diarsipkan dan nama output baru memakai suffix bila perlu.

Run F yang dilaporkan status complete dengan dua file v4 lookup/nolookup dan main submission.csv. Ini bukti laporan eksekusi, bukan pemeriksaan ulang CSV karena file run tersebut tidak tersedia lokal.

Risiko kecil: pembacaan ulang CSV baru pada sel 26 baris 57 tidak memakai float_precision="round_trip", sedangkan pembacaan cache memakainya. Dengan batas minimum float ketat, pembulatan parser berpotensi memicu assert. Tidak terjadi pada run F complete ini. V5 menyamakan parser dan memeriksa hash salinan main.

### H. Reproduksibilitas dan runtime

| Tahap | Detik dilaporkan | Menit | Status |
|---|---:|---:|---|
| A | 2591,006463 | 43,18 | complete |
| B | 3819,619931 | 63,66 | complete |
| C | 0,040791 | <0,01 | cache_only |
| D | 4,739460 | 0,08 | complete |
| E | 962,980223 | 16,05 | complete |
| F | 3,313017 | 0,06 | complete |
| Total | 7381,699885 | 123,03 | sekitar 2,05 jam |

Nilai seconds=0 pada baris kandidat B di tabel akhir bukan waktu training sebenarnya: sel 19 memanggil record_candidate tanpa waktu dan sel 26 drop_duplicates(...,keep="last") mengganti baris B sebelumnya. V5 mengambil waktu model dari REGISTRY bila evaluasi ulang tidak memberikan waktu.

Cache v4 mempunyai hash CSV/fold/parameter dan pemisahan DEBUG/full. Namun INNER_FOLDS belum masuk signature numerik; mengubah jumlah inner fold tanpa mengganti PIPELINE_REV bisa memakai cache TE lama. V5 memasukkannya bersama konfigurasi metrik dan menyimpan konfigurasi lengkap di reproduction.json. Tidak ada bukti perubahan INNER_FOLDS terjadi pada run v4 yang diberikan.

Tidak ada path lokal/Colab atau import kode tim dalam notebook. Optional import dibungkus try/except; kegagalan LightGBM menghentikan baseline inti, sementara library C/E/v2 boleh dilewati. Timeout kooperatif pada iterasi/batch, bukan penghentian paksa operasi native. SHAP dan tuning bersarang menjadi risiko durasi terbesar v5. Tidak ada pengukuran RAM run pengguna; estimasi RAM bukan hasil profiling.

## 3. Isi aktual artefak v2

Semua array float32. Urutan kolom mengikuti lima target resmi. NPZ tidak memuat ID; kesamaan urutan CSV adalah prasyarat pemilik, bukan sesuatu yang dibuktikan oleh shape.

| Kunci | Shape | NaN per kolom | Rentang kolom valid, dibulatkan |
|---|---|---|---|
| oof_raw | (429303,5) | 0,0,0,0,0 | Gas 6485,46–989621,56; listrik 4966,43–327399,81; EUI 4,96–427,77; clean 11,03–100,63; adopsi −2,05–82,98 |
| oof_log1p | (429303,5) | 0,0,429303,429303,429303 | Gas 22522,98–963341,13; listrik 7616,81–296102,97 |
| test_raw | (183987,5) | 0,0,0,0,0 | Gas 13482,82–976248,13; listrik 5842,46–281259,63; EUI 12,42–440,12; clean 11,49–100,28; adopsi −0,45–78,10 |
| test_log1p | (183987,5) | 0,0,183987,183987,183987 | Gas 20782,24–938999,56; listrik 7792,22–278058,66 |

| Isi pickle | Tipe | Ukuran/kunci |
|---|---|---|
| importance | dict | 2: raw, log1p |
| feature_names | list | 199 nama |
| folds | list | 5 fold |
| active_folds | int | skalar |
| oof_predictions | dict | 5: mean, median, ridge, lgbm_raw, lgbm_log |

v2 raw CV 0,292547 dalam tabel pengguna lebih lemah daripada v4 final. Metadata pickle dipakai untuk inspeksi, bukan sebagai kode tim atau fitur. Fold v2 zona, bobot EUI berbeda, gas langsung; semua perbandingan/blend diberi label itu.

## 4. Tabel temuan dan tindakan

Urutan mengutamakan dampak yang terlihat, kemudian risiko yang belum terjadi. Perkiraan arah tanpa pengukuran ditandai TEBAKAN.

| No | Temuan | Bukti (sel) | Dampak ke skor | Keyakinan | Perbaikan |
|---|---|---|---|---|---|
| 1 | Adopsi v4 lebih buruk; representasi target langsung berbeda dari rekomendasi residual | F pengguna; sel 8: A_adoption mode direct | **Terukur:** +0,003723 gabungan vs referensi adopsi; **TEBAKAN:** residual dapat menguranginya | Tinggi untuk selisih, sedang untuk penyebab | Kontrol langsung vs residual di tahap 1, uji per fold, diagnosis subzone kecil |
| 2 | Gas/listrik/EUI/clean sudah membaik; tidak ada keluarga fitur S7 yang terbukti hilang | F pengguna; sel 4/6/8 | **Terukur:** keuntungan gabungan −0,002351 dari empat target | Tinggi | Pertahankan dasar energi; jangan mengganti 12 fitur secara spekulatif; label aproksimasi |
| 3 | Korelasi/weight v2 belum menjadi filter; mean v2 lebih lemah | Sel 11: hanya print; sel 19; tabel pengguna | **Terukur:** mean v2 memperburuk listrik/EUI kandidat; dampak ke final belum terbukti | Tinggi untuk kandidat | Pruning pada OOF fit setiap fold; simpan bobot/alasan; v2 tetap opsional |
| 4 | Gas langsung bisa masuk final; training pembanding mahal | Sel 19 baris 156; sel 8; waktu pengguna | **TEBAKAN:** risiko mengubah struktur; tidak terlihat pada final ini. Sekitar 14,4 menit untuk pembanding | Tinggi untuk cabang/waktu | Gas final wajib turunan; direct opsional dan diagnostik |
| 5 | Waktu model hilang saat baris hasil dievaluasi ulang | Sel 19 baris 119; sel 26 baris 82 | Tidak mengubah skor; membuat estimasi biaya salah | Tinggi | Pertahankan waktu dari registry dan tabel waktu tahap |
| 6 | Hitungan fitur log tidak konsisten; print jumlah belum termasuk TE | Sel 4 baris 93–95,167; header CSV | **TEBAKAN:** berdampak interpretasi reproduksi, bukan bukti fitur terbuang salah | Tinggi untuk aritmetika | Cetak 189 mentah bersih, fitur tambahan, dan total input model |
| 7 | Cache tidak memasukkan INNER_FOLDS secara eksplisit | Sel 2/4 signature, sel 6 fold_te | **TEBAKAN:** cache salah jika konfigurasi diubah; belum terjadi pada bukti run | Tinggi untuk risiko | Tambahkan konfigurasi TE pada fingerprint |
| 8 | Pemilihan blend dilakukan sebelum batas final; average dapat dihitung bersama anggota | Sel 19 | **TEBAKAN:** keputusan kurang konsisten; post-processing tidak mengubah angka run yang diberikan | Tinggi untuk kode | Nilai setelah batas; pemeriksaan final; deduplikasi anggota average |
| 9 | Flag calibrated berdasarkan selisih absolut baseline, termasuk perbaikan | Sel 8 baris 34–35, F pengguna | Tidak mengubah skor; mudah disalahartikan sebagai bug/leakage | Tinggi | Label reproduksi acuan dan kalibrasi publik dibedakan; tampilkan arah delta |
| 10 | Parser CSV baru/cache berbeda | Sel 26 baris 44/57 | **TEBAKAN:** assert minimum float; tidak terjadi pada run complete | Tinggi untuk kode, rendah untuk kejadian | Gunakan round_trip pada semua pembacaan ulang |
| 11 | Parameter S7 asli, best_iteration, fold aktual run tidak tersedia | Sel 2; keluaran yang diberikan | **TEBAKAN:** peluang regularisasi; bukan bukti perlu mengganti energi yang sudah baik | Sedang | Diagnostik iterasi, tuning opsional hanya energi, validasi bersarang |

## 5. Rekomendasi v5 dan batas kesimpulan

1. Dahulukan tahap 1: residual adopsi dibanding kontrol langsung. Jangan menganggap skor 0,118 akan tercapai sebelum hasil keluar.
2. Pertahankan eksperimen listrik tahap 2 dan NNLS tahap 5; hasil v4 sudah mendukung manfaatnya. Kriteria penerimaan tetap >=4/5 fold serta mean gain >=max(5×std,0,001).
3. Jalankan SHAP tahap 6 dan ekspor dua submission tahap 7. Lookup tetap sama dan diverifikasi cakupannya.
4. Setelah itu tambah tahap 3 (keragaman energi) dan 4 (Optuna, default nonaktif). Optuna memakai 25% outer-train dan tiga inner fold per outer fold, lalu refit penuh lima fold; tidak memilih parameter dari label outer-validasi.
5. Tabel v5 memasukkan CV v4 yang dilaporkan pengguna. Tanpa prediksi/fold v4, delta v5-v4 bersifat deskriptif; uji berpasangan dilakukan di antara kandidat v5 pada fold identik.
6. Perkiraan CV−0,01881 tetap ditampilkan sesuai permintaan, disertai skenario pelonggaran 25% gain adopsi. Keduanya bukan jaminan; perbaikan adopsi hanya berlaku pada baris test tanpa lookup.

**Dugaan penyebab utama:** angka yang diberikan mengarahkan masalah CV ke model adopsi langsung yang lebih lemah, sementara empat target lain sudah melampaui referensi S7. Model residual sesuai panduan merupakan perbaikan yang masuk akal untuk diuji, tetapi belum terbukti menjadi penyebab maupun solusi selisih skor publik. Tidak ada bukti lookup salah, fitur S7 hilang sebagai satu keluarga, atau leakage TE pada kode yang dibaca. Selisih publik +0,00149 dan CV +0,001372 bergerak searah, tetapi tanpa label test, OOF v4, bobot blend, dan kode tim, kontribusi publik per target serta reproduksi S7 persis tidak dapat dipastikan.

