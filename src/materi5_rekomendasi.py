"""Sistem rekomendasi: membandingkan Content-Based, Item-KNN, dan Matrix Factorization (SVD).

Evaluasi: leave-one-out per pengguna (interaksi terakhir jadi ground truth),
metrik HitRate@10, Recall@10, Precision@10, NDCG@10. Populeritas dipakai
sebagai baseline.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

inter = pd.read_csv(DATA / "interaksi.csv")
meta = pd.read_csv(DATA / "item_meta.csv")

ue = LabelEncoder().fit(inter["user_id"])
ie = LabelEncoder().fit(inter["item_id"])
inter["u"] = ue.transform(inter["user_id"])
inter["i"] = ie.transform(inter["item_id"])
NU, NI = len(ue.classes_), len(ie.classes_)

R = csr_matrix((inter["rating"].astype(float), (inter["u"], inter["i"])),
               shape=(NU, NI))
B = csr_matrix((np.ones(len(inter)), (inter["u"], inter["i"])), shape=(NU, NI))

# ---- leave-one-out: interaksi terakhir (urutan asli paling akhir) jadi test
inter = inter.reset_index(drop=True)
inter["_ord"] = np.arange(len(inter))
last_idx = inter.groupby("u")["_ord"].max()
test_mask = inter.index.isin(last_idx.values)
train, test = inter[~test_mask], inter[test_mask]
Rtr = csr_matrix((train["rating"].astype(float), (train["u"], train["i"])), shape=(NU, NI))
Btr = csr_matrix((np.ones(len(train)), (train["u"], train["i"])), shape=(NU, NI))

# ---- matriks konten (genre one-hot + fitur numerik ternormalisasi)
meta["i"] = ie.transform(meta["item_id"])
genre_oh = pd.get_dummies(meta.set_index("i")["genre"]).reindex(range(NI), fill_value=0)
num = meta.set_index("i")[["durasi_menit", "tahun", "popularitas", "rata_rating"]].reindex(range(NI))
num = (num - num.mean()) / (num.std() + 1e-9)
num = num.fillna(0)
C = np.hstack([genre_oh.values, num.values])
C = normalize(C)

K = 10


def rank_metrics(scores, held):
    """scores: (NU, NI) matriks skor. held: dict user->item index."""
    hits, recalls, ndcgs = [], [], []
    order = np.argsort(-scores, axis=1)
    for u, it in held.items():
        top = order[u, :K]
        hit = int(it in top)
        hits.append(hit)
        recalls.append(hit / 1.0)
        if hit:
            ndcgs.append(1.0 / np.log2(2 + int(np.where(top == it)[0][0])))
        else:
            ndcgs.append(0.0)
    return float(np.mean(hits)), float(np.mean(recalls)), float(np.mean(ndcgs))


held = dict(zip(test["u"].values, test["i"].values))
results = {}

# ---- 1. Populeritas (baseline)
pop = np.asarray(Btr.sum(axis=0)).ravel()
pop_scores = np.tile(pop, (NU, 1))
results["Populeritas"] = rank_metrics(pop_scores, held)

# ---- 2. Content-based: rata-rata konten item yang disukai pengguna
prof = normalize(Btr.astype(float) @ C)
cb = prof @ C.T
cb[Btr.toarray() > 0] = -np.inf
results["ContentBased"] = rank_metrics(cb, held)

# ---- 3. Item-KNN (kolerasi kosine antar item)
item_sim = cosine_similarity(Btr.T.toarray())
np.fill_diagonal(item_sim, 0)
knn = np.asarray(Btr.astype(float).todense()) @ item_sim
knn[Btr.toarray() > 0] = -np.inf
results["ItemKNN"] = rank_metrics(knn, held)

# ---- 4. Matrix Factorization (SVD pada rating)
svd = TruncatedSVD(n_components=48, random_state=SEED)
mf = svd.fit_transform(Rtr)
mf_scores = mf @ svd.components_
mf_scores[Btr.toarray() > 0] = -np.inf
results["MatrixFactorization"] = rank_metrics(mf_scores, held)

# ---- cakupan (coverage) & keragaman (intra-list diversity)
def coverage(scores):
    order = np.argsort(-scores, axis=1)
    rec = order[:, :K].ravel()
    return len(np.unique(rec)) / NI


def diversity(scores):
    order = np.argsort(-scores, axis=1)[:, :K]
    rec_items = C[order]                       # (NU, K, d)
    sims = []
    for u in range(min(NU, 300)):
        m = rec_items[u]
        s = m @ m.T
        n = K * (K - 1) / 2
        sims.append((s.sum() - np.trace(s)) / (2 * n))
    return float(np.mean(sims))


cov = {n: round(coverage(s), 4) for n, s in
       [("Populeritas", pop_scores), ("ContentBased", cb), ("ItemKNN", knn),
        ("MatrixFactorization", mf_scores)]}
div = {n: round(diversity(s), 4) for n, s in
       [("Populeritas", pop_scores), ("ContentBased", cb), ("ItemKNN", knn),
        ("MatrixFactorization", mf_scores)]}

rows = []
for name, (hr, rc, nd) in results.items():
    rows.append({"model": name, "HitRate@10": round(hr, 4), "Recall@10": round(rc, 4),
                 "NDCG@10": round(nd, 4), "coverage@10": cov[name],
                 "diversity@10": div[name]})
res_df = pd.DataFrame(rows).sort_values("NDCG@10", ascending=False)
res_df.to_csv(OUT / "hasil_rekomendasi.csv", index=False)

metrics = {"n_users": int(NU), "n_items": int(NI),
           "n_interaksi": int(len(inter)), "kepadatan": round(len(inter) / (NU * NI), 5),
           "hasil": {r["model"]: r for r in rows}}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(res_df))
axes[0].bar(x - 0.2, res_df["HitRate@10"], 0.4, label="HitRate@10", color="#33502e")
axes[0].bar(x + 0.2, res_df["NDCG@10"], 0.4, label="NDCG@10", color="#2e6bb3")
axes[0].set_xticks(x, res_df["model"], rotation=20, ha="right")
axes[0].set_ylabel("skor")
axes[0].set_title("Kualitas rekomendasi (leave-one-out)")
axes[0].legend()
axes[1].scatter(res_df["diversity@10"], res_df["NDCG@10"], s=90,
                c=["#b3402e", "#33502e", "#2e6bb3", "#8a6d3b"])
for _, r in res_df.iterrows():
    axes[1].annotate(r["model"], (r["diversity@10"], r["NDCG@10"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
axes[1].set_xlabel("keragaman intra-list")
axes[1].set_ylabel("NDCG@10")
axes[1].set_title("Trade-off akurasi vs keragaman")
fig.tight_layout()
fig.savefig(OUT / "rekomendasi.png", dpi=130)
plt.close(fig)

print(f"[analisis] {NU} pengguna x {NI} item | {len(inter):,} interaksi "
      f"(kepadatan {metrics['kepadatan']:.3%})")
print(res_df.to_string(index=False))
