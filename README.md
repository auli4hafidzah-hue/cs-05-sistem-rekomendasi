# Studi Kasus 5 — Sistem Rekomendasi: Content-Based, Item-KNN, dan Matrix Factorization

Membandingkan tiga pendekatan rekomendasi + baseline populeritas pada matriks
interaksi pengguna-item yang sparse, dengan evaluasi **leave-one-out** dan
metrik keragaman.

## Pertanyaan analitis

1. Model mana yang paling akurat merekomendasikan item yang belum pernah
   diinteraksi pengguna?
2. Seberapa besar trade-off antara akurasi dan **keragaman** rekomendasi?
3. Seberapa luas **cakupan** item yang direkomendasikan oleh tiap model?

## Dataset

`interaksi.csv` — **19.420 interaksi** dari 600 pengguna × 240 item
(**kepadatan 13,486%**), rating 1–10 dan jumlah tonton ulang.

Model pembangkitan: 8 segmen pengguna laten, setiap item punya embedding
laten 5 dimensi. Skor afinitas = dot product preferensi segmen × fitur item
+ bias item, diubah jadi probabilitas melalui sigmoid.

**Detail penting:** 15% interaksi diberi skor kecil (probabilitas × 0,15)
sehingga korelasi murni tidak cukup — model harus memanfaatkan pola kolektif
(collaborative filtering), bukan sekadar mengurutkan item terpopuler.

`item_meta.csv` memuat metadata item (genre, durasi, tahun, popularitas,
rata-rata rating) untuk pendekatan berbasis konten.

## Metode

- **Leave-one-out**: interaksi terakhir setiap pengguna dijadikan ground truth,
  sisanya (99,9% data) untuk pelatihan
- **Top-10 rekomendasi**, item yang sudah diinteraksi dikecualikan
- Empat model:
  - **Populeritas** (baseline): item paling banyak diinteraksi
  - **ContentBased**: rata-rata terberat konten item yang disukai pengguna
  - **ItemKNN**: kemiripan kosine antar item, skor = rata-rata terberat item serupa
  - **MatrixFactorization**: SVD 48 komponen pada matriks rating
- Metrik: HitRate@10, Recall@10, NDCG@10, **coverage@10**, **diversity@10**
  (kemiripan kosine rata-rata antar pasangan item dalam satu daftar)

## Hasil terukur

| Model | HitRate@10 | Recall@10 | NDCG@10 | Coverage@10 | Diversity@10 |
|---|---|---|---|---|---|
| **ItemKNN** | **0,0967** | **0,0967** | **0,0404** | 0,4708 | 0,0463 |
| MatrixFactorization | 0,0817 | 0,0817 | 0,0391 | **0,9458** | 0,0232 |
| Populeritas | 0,0750 | 0,0750 | 0,0304 | 0,0417 | **0,0494** |
| ContentBased | 0,0383 | 0,0383 | 0,0173 | **1,0000** | **0,5610** |

## Interpretasi

**ItemKNN menang tipis** (NDCG 0,0404 vs 0,0391 MF), dan ini konsisten dengan
literatur: pada matriks dengan kepadatan sedang (13,5%) dan sinyal segmen yang
kuat, kemiripan antar item langsung lebih efektif daripada dekomposisi matriks.

Trade-off yang jelas terlihat:

- **Akurasi vs cakupan**: Matrix Factorization hanya merekomendasikan 4,17%
  item (kecenderungan ke item populer), sementara ItemKNN mencakup 94,58%.
  Item populer tetap muncul di daftar atas semua orang.
- **Akurasi vs keragaman**: ContentBased punya keragaman tertinggi (0,5610)
  tapi akurasi terendah — model ini hanya mengandalkan genre/metadata, tidak
  menangkap preferensi tersembunyi.
- **Diversity ItemKNN rendah (0,0463)** artinya daftar 10 item yang direkomendasikan
  sangat mirip satu sama lain. Untuk pengguna, ini berarti rekomendasi yang
  monoton — perlu dikombinasikan dengan strategi eksplorasi.

Populeritas sebagai baseline sudah mencapai HitRate 0,0750, cukup dekat dengan
MF (0,0817). Ini peringatan: pada dataset ini, **cukup merekomendasikan yang
populer** — peningkatan model kompleks hanya ~9% relatif.

## Struktur

```
├── src/generate_dataset.py          # interaksi sintetis + metadata item
├── src/materi5_rekomendasi.py       # 4 model, leave-one-out, 5 metrik
├── outputs/
│   ├── item_meta.csv
│   ├── hasil_rekomendasi.csv       # perbandingan lengkap 4 model
│   ├── rekomendasi.png              # kualitas + trade-off akurasi vs keragaman
│   └── metrics.json
```

## Cara menjalankan

```bash
pip install -r requirements.txt
bash run.sh
```

## Output

`outputs/rekomendasi.png` menampilkan dua panel: perbandingan HitRate dan NDCG
antar model, serta scatter trade-off keragaman vs akurasi.
