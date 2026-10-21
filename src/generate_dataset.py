"""Interaksi pengguna-item sintetis untuk evaluasi sistem rekomendasi.

Model pembangkitan: 8 segmen pengguna (preferensi genre laten), 40 item
(masing-masing punya embedding laten 5-d), rating implisit = probabilitas
berinteraksi dari dot product + bias segmen. Matriks interaksi dibuat sparse
seperti nyata (~6% kepadatan), dengan 15% interaksi positif berskala kecil
sehingga model kolarasi murni tidak cukup.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_USERS, N_ITEMS = 600, 240
SEGMENTS = 8
DIM = 5
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

user_seg = rng.integers(0, SEGMENTS, N_USERS)
item_feat = rng.normal(0, 1, (N_ITEMS, DIM))
item_bias = rng.normal(0, 0.35, N_ITEMS)
# preferensi genre laten per segmen: matriks (segmen x item) yang menentukan
# afinitas dasar, sehingga rekomendasi berbasis konten punya sinyal nyata.
seg_pref = rng.normal(0, 1, (SEGMENTS, DIM))

item_meta = pd.DataFrame({
    "item_id": [f"item_{i:03d}" for i in range(N_ITEMS)],
    "genre": rng.choice(["aksi", "komedi", "drama", "horor", "dokumenter",
                         "anak", "thriller", "romansa"], N_ITEMS),
    "durasi_menit": rng.integers(18, 175, N_ITEMS),
    "tahun": rng.integers(2005, 2027, N_ITEMS),
    "popularitas": rng.gamma(2.0, 8.0, N_ITEMS).round(2),
    "rata_rating": np.clip(rng.normal(7.1, 1.1, N_ITEMS), 3.0, 9.8).round(2),
})
item_meta.to_csv(DATA / "item_meta.csv", index=False)

rows = []
for u in range(N_USERS):
    s = user_seg[u]
    # aktivitas dasar pengguna: distribusi log-normal, sebagian pengguna pasif
    n_int = int(np.clip(rng.lognormal(3.4, 0.9), 3, 60))
    # skor afinitas: dot product preferensi segmen x fitur item + bias item
    scores = seg_pref[s] @ item_feat.T + item_bias + rng.normal(0, 0.5, N_ITEMS)
    probs = 1.0 / (1.0 + np.exp(-scores))
    # 15% interaksi berskala kecil -> korelasi murni tidak cukup, butuh CF
    small = rng.random(N_ITEMS) < 0.15
    probs = np.where(small, probs * 0.15, probs)
    chosen = rng.choice(N_ITEMS, size=min(n_int, N_ITEMS), replace=False,
                        p=probs / probs.sum())
    for it in chosen:
        rating = np.clip(round(rng.normal(6.8 + 2.2 * probs[it], 1.0), 1), 1, 10)
        rows.append((f"user_{u:04d}", f"item_{int(it):03d}", int(rating),
                     int(rng.integers(1, 5))))

inter = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "menonton_ulang"])
inter.to_csv(DATA / "interaksi.csv", index=False)
pd.DataFrame({"user_id": [f"user_{u:04d}" for u in range(N_USERS)],
              "segmen": user_seg}).to_csv(DATA / "user_segmen.csv", index=False)

density = len(inter) / (N_USERS * N_ITEMS)
print(f"[generate] interaksi.csv: {len(inter):,} baris | kepadatan {density:.3%}")
print(f"[generate] {N_USERS} pengguna x {N_ITEMS} item | {SEGMENTS} segmen laten")
print(f"[generate] rating rata-rata {inter['rating'].mean():.2f} | "
      f"median interaksi/pengguna {inter.groupby('user_id').size().median():.0f}")
