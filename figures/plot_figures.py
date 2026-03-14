"""
Generate all figures for the LLM-Augmented Clustering Engine report.

Usage:
    python figures/plot_figures.py

Figures produced in report/figures/:
  fig1_tsne.png              -- t-SNE 2x2: ground-truth vs K-Means (Bank77)
  fig2_llm_budget.png        -- LLM calls per method (horizontal bar)
  fig3_granularity.png       -- ClusterLLM F-beta curve  [needs make clusterllm]
  fig4_results.png           -- ACC/NMI grouped bars   [needs make run]
  fig5_constraint_graph.png  -- Pairwise constraints overlaid on t-SNE
  fig6_oracle_accuracy.png   -- LLM oracle confusion matrix + P/R/F1
  fig7_normalization_drift.png -- Distance-to-centroid scatter before vs after
  fig8_paraphrase_cloud.png  -- Paraphrase ensemble cloud (PCA-2D)
  fig9_confusion_heatmap.png -- K-Means 77x77 confusion heatmap
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
BANK77_DIR = os.path.join(RESULTS_DIR, "bank77")
OUT_DIR = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Wong (2011) colour-blind palette
PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#999999",
]


# =============================================================================
# Shared helpers
# =============================================================================


def _load_bank77_raw():
    """Return (embeddings_norm, labels_gt)."""
    emb_path = os.path.join(BANK77_DIR, "embeddings.pkl")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(emb_path)
    with open(emb_path, "rb") as f:
        cached = pickle.load(f)
    embeddings = cached["embeddings"]

    from datasets import load_dataset as _hf

    data = _hf("banking77")["test"]
    labels_gt = np.array(data["label"])

    from sklearn.preprocessing import normalize

    return normalize(embeddings, axis=1), labels_gt


def _get_bank77_tsne():
    """Return (emb_norm, labels_gt, tsne_xy), caching the 2-D coordinates."""
    cache_path = os.path.join(BANK77_DIR, "tsne_coords.npy")
    emb_norm, labels_gt = _load_bank77_raw()

    if os.path.exists(cache_path):
        tsne_xy = np.load(cache_path)
        print("  [t-SNE] Loaded cached coordinates.")
        return emb_norm, labels_gt, tsne_xy

    print("  [t-SNE] Running PCA(50) -> t-SNE (~30 s)...")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca_feats = PCA(n_components=50, random_state=42).fit_transform(emb_norm)
    tsne_xy = TSNE(
        n_components=2, perplexity=40, max_iter=1000, random_state=42, n_jobs=-1
    ).fit_transform(pca_feats)
    np.save(cache_path, tsne_xy)
    print(f"  [t-SNE] Saved -> {cache_path}")
    return emb_norm, labels_gt, tsne_xy


# =============================================================================
# Figure 1 -- t-SNE ground-truth vs K-Means
# =============================================================================


def plot_tsne():
    print("[fig1] t-SNE ground-truth vs K-Means")
    try:
        emb_norm, labels_gt, tsne_xy = _get_bank77_tsne()
    except Exception as e:
        print(f"  Skipping: {e}")
        return

    from sklearn.cluster import KMeans

    n_classes = len(np.unique(labels_gt))
    labels_km = KMeans(
        n_clusters=n_classes, random_state=42, n_init="auto"
    ).fit_predict(emb_norm)

    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, labs, title in zip(
        axes, [labels_gt, labels_km], ["Ground-truth labels", "K-Means assignments"]
    ):
        colors = [cmap(int(lbl) % 20) for lbl in labs]
        ax.scatter(
            tsne_xy[:, 0],
            tsne_xy[:, 1],
            c=colors,
            s=2.5,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(title, pad=5, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    fig.suptitle(
        "t-SNE of Bank77 Embeddings (all-mpnet-base-v2)",
        y=1.01,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_tsne.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
# Figure 2 -- LLM call budget
# =============================================================================


def plot_llm_budget():
    print("[fig2] LLM call budget")
    # Actual call counts: triplets(1024) + pairwise(400) per dataset for ClusterLLM
    data = [
        ("Keyphrase",      3_080, 4_500, 2_472),
        ("Normalization",  3_080, 4_500, 2_472),
        ("Paraphrase",     3_080, 4_500, 2_472),
        ("PCKMeans",       2_000, 2_000, 2_000),
        ("ClusterLLM",     1_424, 1_424, 1_424),
        ("LLM Correction",   251,   361,   250),
    ]
    methods = [d[0] for d in data]
    bank77 = [d[1] for d in data]
    clinc  = [d[2] for d in data]
    tweet  = [d[3] for d in data]
    y, h = np.arange(len(methods)), 0.25

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.barh(y + h,   bank77, h, label="Bank77", color=PALETTE[0], alpha=0.85)
    ax.barh(y,       clinc,  h, label="CLINC",  color=PALETTE[1], alpha=0.85)
    ax.barh(y - h,   tweet,  h, label="Tweet",  color=PALETTE[2], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel("LLM generation calls")
    ax.set_title("LLM Call Budget per Method")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(bank77 + clinc + tweet) * 1.18)

    out = os.path.join(OUT_DIR, "fig2_llm_budget.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
# Figure 3 -- ClusterLLM F-beta granularity curve
# =============================================================================


def plot_granularity():
    csv_path = os.path.join(BANK77_DIR, "clusterllm_fbeta_scores.csv")
    if not os.path.exists(csv_path):
        print(
            "[fig3] Not yet available -- run 'make clusterllm DATASETS=bank77' first."
        )
        return
    print("[fig3] ClusterLLM granularity curve")
    df = pd.read_csv(csv_path)
    k_star = int(df.loc[df["fbeta"].idxmax(), "k"])
    k_true = 77

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.plot(df["k"], df["fbeta"], color=PALETTE[0], lw=1.5, label="F-beta score")
    ax.axvline(k_star, color=PALETTE[2], ls="--", lw=1.2, label=f"k* = {k_star}")
    ax.axvline(k_true, color=PALETTE[3], ls=":", lw=1.2, label=f"True k = {k_true}")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("F-beta (beta=5)")
    ax.set_title("ClusterLLM Granularity Selection (Bank77)")
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    out = os.path.join(OUT_DIR, "fig3_granularity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
# Figure 4 -- Results grouped bar chart
# =============================================================================


def plot_results():
    csv_candidates = [
        os.path.join(RESULTS_DIR, "experiment_results_table.csv"),
        os.path.join(ROOT, "experiment_results_table.csv"),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if csv_path is None:
        print("[fig4] experiment_results_table.csv not found -- run 'make run' first.")
        return
    print(f"[fig4] Results bars from {csv_path}")
    df = pd.read_csv(csv_path)
    if "Accuracy" in df.columns:
        df = df.rename(columns={"Accuracy": "ACC"})
    if not {"Dataset", "Method", "ACC", "NMI"}.issubset(df.columns):
        print("[fig4] Unexpected columns -- skipping.")
        return

    df = df.groupby(["Dataset", "Method"])[["ACC", "NMI"]].mean().reset_index()
    datasets = [
        d
        for d in ["bank77", "clinc", "tweet"]
        if d in df["Dataset"].str.lower().unique()
    ]
    if not datasets:
        print("[fig4] No data -- skipping.")
        return

    order = [
        "KMeans",
        "JoSE + Spherical KMeans",
        "Pairwise Constraints",
        "LLM Correction",
        "Keyphrase (concatenated)",
        "LLM Normalization",
        "LLM Paraphrase Ensemble",
        "ClusterLLM",
    ]
    SHORT = {
        "KMeans": "K-Means",
        "JoSE + Spherical KMeans": "JoSE",
        "Pairwise Constraints": "PCKMeans",
        "LLM Correction": "Correction",
        "Keyphrase (concatenated)": "Keyphrase",
        "LLM Normalization": "Norm.",
        "LLM Paraphrase Ensemble": "Paraph.",
        "ClusterLLM": "ClusterLLM",
    }
    methods_in = df["Method"].unique().tolist()
    methods = [m for m in order if m in methods_in]
    methods += [m for m in methods_in if m not in methods]
    labels = [SHORT.get(m, m) for m in methods]

    y, w = np.arange(len(methods)), 0.30
    n_ds = len(datasets)
    # Vertical stacked layout: one row per dataset, methods on y-axis
    fig, axes = plt.subplots(n_ds, 1, figsize=(3.5, 1.5 + 0.40 * len(methods) * n_ds / n_ds), sharex=True)
    if n_ds == 1:
        axes = [axes]

    acc_patch = mpatches.Patch(color=PALETTE[0], label="ACC")
    nmi_patch = mpatches.Patch(color=PALETTE[1], label="NMI")

    for idx, (ax, ds) in enumerate(zip(axes, datasets)):
        sub = df[df["Dataset"].str.lower() == ds]

        def _v(col, m):
            r = sub[sub["Method"] == m]
            return float(r[col].values[0]) if len(r) else 0.0

        acc_vals = [_v("ACC", m) for m in methods]
        nmi_vals = [_v("NMI", m) for m in methods]

        ax.barh(y + w / 2, acc_vals, w, color=PALETTE[0], alpha=0.85)
        ax.barh(y - w / 2, nmi_vals, w, color=PALETTE[1], alpha=0.85)
        ax.set_title(ds.capitalize(), pad=3, fontsize=9, loc="left")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
        if idx == n_ds - 1:
            ax.set_xlabel("Score", fontsize=8)

    axes[0].legend(
        handles=[acc_patch, nmi_patch], loc="lower right", fontsize=8, framealpha=0.85
    )
    fig.tight_layout(pad=0.4, h_pad=0.8)

    out = os.path.join(OUT_DIR, "fig4_results.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
# Figure 5 -- Pairwise constraint graph overlaid on t-SNE
# =============================================================================


def plot_constraint_graph():
    cache_path = os.path.join(BANK77_DIR, "pairwise_cache.pkl")
    if not os.path.exists(cache_path):
        print("[fig5] pairwise_cache.pkl not found -- skipping.")
        return
    print("[fig5] Pairwise constraint graph on t-SNE")
    try:
        emb_norm, labels_gt, tsne_xy = _get_bank77_tsne()
    except Exception as e:
        print(f"  Skipping: {e}")
        return

    with open(cache_path, "rb") as f:
        pair_labels = pickle.load(f)

    yes_pairs = [(i, j) for (i, j), v in pair_labels.items() if v == "YES"]
    no_pairs = [(i, j) for (i, j), v in pair_labels.items() if v == "NO"]

    # Sample far fewer NO edges so the graph isn't a web
    rng = np.random.default_rng(42)
    n_no_show = min(60, len(no_pairs))
    no_sample = [
        no_pairs[k] for k in rng.choice(len(no_pairs), size=n_no_show, replace=False)
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    # Background: dots coloured by ground-truth class
    cmap = plt.get_cmap("tab20")
    colors_gt = [cmap(int(lbl) % 20) for lbl in labels_gt]
    ax.scatter(
        tsne_xy[:, 0],
        tsne_xy[:, 1],
        c=colors_gt,
        s=3.5,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # Cannot-link edges (very sparse, very faint)
    for i, j in no_sample:
        ax.plot(
            [tsne_xy[i, 0], tsne_xy[j, 0]],
            [tsne_xy[i, 1], tsne_xy[j, 1]],
            color="#D55E00",
            alpha=0.18,
            lw=0.7,
            zorder=2,
        )

    # Must-link edges (all, prominent)
    for i, j in yes_pairs:
        ax.plot(
            [tsne_xy[i, 0], tsne_xy[j, 0]],
            [tsne_xy[i, 1], tsne_xy[j, 1]],
            color="#009E73",
            alpha=0.70,
            lw=1.0,
            zorder=3,
        )

    legend_elems = [
        Line2D(
            [0],
            [0],
            color="#009E73",
            lw=1.8,
            label=f"Must-link / YES  (n={len(yes_pairs)})",
        ),
        Line2D(
            [0],
            [0],
            color="#D55E00",
            lw=1.8,
            label=f"Cannot-link / NO  (sample {n_no_show} of {len(no_pairs)})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#aaaaaa",
            markersize=5,
            label="Documents (coloured by class)",
        ),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "LLM Pairwise Constraints Overlaid on Bank77 t-SNE\n"
        "(green = must-link, red = cannot-link)",
        pad=7,
    )

    out = os.path.join(OUT_DIR, "fig5_constraint_graph.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
# Figure 6 -- LLM oracle accuracy
# =============================================================================


def plot_oracle_accuracy():
    cache_path = os.path.join(BANK77_DIR, "pairwise_cache.pkl")
    if not os.path.exists(cache_path):
        print("[fig6] pairwise_cache.pkl not found -- skipping.")
        return
    print("[fig6] LLM oracle accuracy")
    try:
        _, labels_gt = _load_bank77_raw()
    except Exception as e:
        print(f"  Skipping: {e}")
        return

    with open(cache_path, "rb") as f:
        pair_labels = pickle.load(f)

    tp = fp = tn = fn = 0
    for (i, j), label in pair_labels.items():
        same = labels_gt[i] == labels_gt[j]
        if label == "YES":
            tp += int(same)
            fp += int(not same)
        elif label == "NO":
            tn += int(not same)
            fn += int(same)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.5, 5.2)
    )
    fig.subplots_adjust(top=0.88, hspace=0.55)

    # ── Left: 2x2 confusion matrix ───────────────────────────────────────────
    mat = np.array([[tp, fp], [fn, tn]], dtype=float)
    mat_norm = mat / mat.sum(axis=1, keepdims=True)
    cell_labels = [["TP", "FP"], ["FN", "TN"]]
    colors_map = [["#009E73", "#D55E00"], ["#D55E00", "#0072B2"]]
    labels_rows = ["LLM: YES\n(must-link)", "LLM: NO\n(cannot-link)"]
    labels_cols = ["Same class", "Diff. class"]

    for r in range(2):
        for c in range(2):
            alpha = 0.25 + 0.65 * mat_norm[r, c]
            ax1.add_patch(
                plt.Rectangle((c, 1 - r), 1, 1, color=colors_map[r][c], alpha=alpha)
            )
            ax1.text(
                c + 0.5,
                1.5 - r,
                f"{cell_labels[r][c]}\n{int(mat[r,c])}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 2)
    ax1.set_xticks([0.5, 1.5])
    ax1.set_xticklabels(labels_cols, fontsize=8.5)
    ax1.set_yticks([0.5, 1.5])
    ax1.set_yticklabels(labels_rows[::-1], fontsize=8.5)
    ax1.tick_params(length=0)
    ax1.set_title("Confusion Matrix\n(ground-truth columns)", pad=5, fontsize=9)

    # ── Right: metrics bar ───────────────────────────────────────────────────
    metrics = ["Precision", "Recall", "F1", "Specificity"]
    values = [precision, recall, f1, specificity]
    clrs = [PALETTE[0], PALETTE[2], PALETTE[3], PALETTE[1]]
    bars = ax2.barh(metrics, values, color=clrs, alpha=0.87, height=0.55)
    for bar, v in zip(bars, values):
        ax2.text(
            v + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}",
            va="center",
            fontsize=9.5,
        )
    ax2.set_xlim(0, 1.18)
    ax2.set_xlabel("Score")
    ax2.set_title("Oracle Quality", pad=5, fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "LLM Oracle Accuracy on Bank77 Pairwise Queries" f"  (n = {tp+fp+tn+fn:,})",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )

    out = os.path.join(OUT_DIR, "fig6_oracle_accuracy.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")
    print(
        f"    Precision={precision:.3f}  Recall={recall:.3f}  "
        f"F1={f1:.3f}  Specificity={specificity:.3f}"
    )


# =============================================================================
# Figure 7 -- Normalisation: distance-to-centroid scatter (before vs after)
# =============================================================================


def plot_normalization_drift():
    norm_cache = os.path.join(BANK77_DIR, "normalization_cache.pkl")
    if not os.path.exists(norm_cache):
        print("[fig7] normalization_cache.pkl not found -- skipping.")
        return
    print("[fig7] Normalisation distance-to-centroid scatter")

    with open(norm_cache, "rb") as f:
        norm_map = pickle.load(f)
    norm_texts = [norm_map.get(i, "") for i in range(3080)]

    with open(os.path.join(BANK77_DIR, "embeddings.pkl"), "rb") as f:
        orig_emb = pickle.load(f)["embeddings"]

    norm_emb_cache = os.path.join(BANK77_DIR, "normalization_embeddings.npy")
    if os.path.exists(norm_emb_cache):
        norm_emb = np.load(norm_emb_cache)
        print("  Loaded cached normalized embeddings.")
    else:
        print("  Embedding normalized texts via sentence-transformers...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-mpnet-base-v2")
        norm_emb = model.encode(
            norm_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )
        np.save(norm_emb_cache, norm_emb)

    from sklearn.preprocessing import normalize

    try:
        _, labels_gt = _load_bank77_raw()
    except Exception as e:
        print(f"  Skipping: {e}")
        return

    orig_norm = normalize(orig_emb, axis=1)
    norm_norm = normalize(norm_emb, axis=1)

    # Distance to ground-truth class centroid: before and after
    n_classes = int(labels_gt.max()) + 1
    dist_before = np.zeros(len(labels_gt))
    dist_after = np.zeros(len(labels_gt))
    for c in range(n_classes):
        idx = np.where(labels_gt == c)[0]
        centroid_orig = orig_norm[idx].mean(axis=0)
        centroid_norm = norm_norm[idx].mean(axis=0)
        centroid_orig /= np.linalg.norm(centroid_orig) + 1e-9
        centroid_norm /= np.linalg.norm(centroid_norm) + 1e-9
        # cosine distance = 1 - dot product (for unit vectors)
        dist_before[idx] = 1.0 - orig_norm[idx] @ centroid_orig
        dist_after[idx] = 1.0 - norm_norm[idx] @ centroid_norm

    improved = dist_after < dist_before
    pct = 100.0 * improved.mean()

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    lim = max(dist_before.max(), dist_after.max()) * 1.05

    # Background: all points, coloured by improvement
    ax.scatter(
        dist_before[~improved],
        dist_after[~improved],
        s=3,
        alpha=0.25,
        color="#D55E00",
        linewidths=0,
        rasterized=True,
        label=f"Farther after norm. ({100-pct:.0f}%)",
    )
    ax.scatter(
        dist_before[improved],
        dist_after[improved],
        s=3,
        alpha=0.25,
        color="#0072B2",
        linewidths=0,
        rasterized=True,
        label=f"Closer after norm. ({pct:.0f}%)",
    )

    # Diagonal y = x
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="No change")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Distance to class centroid (original)")
    ax.set_ylabel("Distance to class centroid (normalised)")
    ax.set_title(
        "Normalisation Effect on Embedding Space\n"
        "Points below the diagonal moved closer to their class centroid",
        pad=6,
    )
    ax.legend(fontsize=8, framealpha=0.9, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate mean delta
    mean_delta = (dist_before - dist_after).mean()
    ax.text(
        0.97,
        0.03,
        f"Mean dist. reduction: {mean_delta:+.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, pad=2),
    )

    out = os.path.join(OUT_DIR, "fig7_normalization_drift.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}  ({pct:.1f}% of docs moved closer to centroid)")


# =============================================================================
# Figure 8 -- Paraphrase ensemble cloud (PCA-2D, clean)
# =============================================================================


def plot_paraphrase_cloud():
    para_cache = os.path.join(BANK77_DIR, "paraphrase_cache.pkl")
    if not os.path.exists(para_cache):
        print("[fig8] paraphrase_cache.pkl not found -- skipping.")
        return
    print("[fig8] Paraphrase embedding cloud")

    with open(para_cache, "rb") as f:
        para_map = pickle.load(f)

    try:
        emb_norm, labels_gt, tsne_xy = _get_bank77_tsne()
    except Exception as e:
        print(f"  Skipping: {e}")
        return


    # Pick 4 well-separated classes via greedy farthest-point on t-SNE centroids
    n_classes = int(labels_gt.max()) + 1
    class_centroids_tsne = np.array(
        [tsne_xy[labels_gt == c].mean(axis=0) for c in range(n_classes)]
    )
    chosen = [0]
    for _ in range(3):
        dists = np.array(
            [
                min(
                    np.linalg.norm(class_centroids_tsne[c] - class_centroids_tsne[ch])
                    for ch in chosen
                )
                for c in range(n_classes)
            ]
        )
        dists[chosen] = -1
        chosen.append(int(np.argmax(dists)))

    # 2 focal docs per class (closest to their class centroid in embedding space)
    focal_indices = []
    for c in chosen:
        idxs = np.where(labels_gt == c)[0]
        centroid = emb_norm[idxs].mean(axis=0)
        dists = np.linalg.norm(emb_norm[idxs] - centroid, axis=1)
        focal_indices.extend(idxs[np.argsort(dists)[:2]].tolist())
    focal_indices = focal_indices[:8]

    # Load or generate paraphrase embeddings
    para_emb_cache = os.path.join(BANK77_DIR, "paraphrase_focal_embeddings.npy")
    para_idx_cache = os.path.join(BANK77_DIR, "paraphrase_focal_indices.npy")
    cached_ok = False
    if os.path.exists(para_emb_cache) and os.path.exists(para_idx_cache):
        if np.load(para_idx_cache).tolist() == focal_indices:
            para_embs_flat = np.load(para_emb_cache)
            cached_ok = True
            print("  Loaded cached paraphrase embeddings.")

    if not cached_ok:
        print("  Embedding paraphrase texts via sentence-transformers...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-mpnet-base-v2")
        para_texts = []
        for idx in focal_indices:
            para_texts.extend(para_map.get(idx, [])[:3])
        para_embs_flat = model.encode(
            para_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        np.save(para_emb_cache, para_embs_flat)
        np.save(para_idx_cache, np.array(focal_indices))

    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA

    para_embs_flat = normalize(para_embs_flat, axis=1)

    focal_orig = emb_norm[focal_indices]  # (8, 768)
    para_blocks = para_embs_flat.reshape(8, 3, 768)  # (8, 3, 768)
    stack = np.concatenate([focal_orig[:, None, :], para_blocks], axis=1)  # (8, 4, 768)
    means_raw = stack.mean(axis=1)  # (8, 768)
    means = normalize(means_raw, axis=1)

    # PCA-2D on only the focal + paraphrase + mean points (40 total)
    all_pts = np.vstack(
        [
            focal_orig,  # 8
            para_embs_flat,  # 24
            means,  # 8
        ]
    )  # 40 x 768
    pca2 = PCA(n_components=2, random_state=42)
    pts2d = pca2.fit_transform(all_pts)

    xy_orig = pts2d[:8]
    xy_paras = pts2d[8:32].reshape(8, 3, 2)
    xy_means = pts2d[32:]

    # Assign a distinct colour per class (not per focal doc)
    class_colors = {}
    cmap4 = [PALETTE[i] for i in [0, 1, 2, 3]]
    for k_idx, (k, c_id) in enumerate(
        zip(range(8), [labels_gt[i] for i in focal_indices])
    ):
        class_idx = chosen.index(int(c_id))
        class_colors[k_idx] = cmap4[class_idx]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    for k in range(8):
        col = class_colors[k]
        # Paraphrases -> mean connectors
        for p in range(3):
            xp, yp = xy_paras[k, p]
            ax.plot(
                [xp, xy_means[k, 0]],
                [yp, xy_means[k, 1]],
                color=col,
                lw=0.9,
                alpha=0.45,
                ls="--",
                zorder=2,
            )
            ax.scatter(
                xp,
                yp,
                s=45,
                color=col,
                alpha=0.70,
                marker="o",
                edgecolors="none",
                zorder=3,
            )

        # Original -> mean connector
        ax.plot(
            [xy_orig[k, 0], xy_means[k, 0]],
            [xy_orig[k, 1], xy_means[k, 1]],
            color=col,
            lw=0.9,
            alpha=0.45,
            ls="--",
            zorder=2,
        )

        # Original (diamond)
        ax.scatter(
            *xy_orig[k],
            s=90,
            color=col,
            alpha=0.95,
            marker="D",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

        # Ensemble mean (star)
        ax.scatter(
            *xy_means[k],
            s=200,
            color=col,
            alpha=1.0,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    # Class legend (one entry per class, not per focal doc)
    class_patches = [
        mpatches.Patch(color=cmap4[i], label=f"Intent class {chosen[i]}")
        for i in range(4)
    ]
    symbol_elems = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="gray",
            markersize=7,
            label="Original document",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=6,
            label="LLM paraphrase",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gray",
            markersize=11,
            label="Ensemble mean",
        ),
    ]
    ax.legend(
        handles=class_patches + symbol_elems,
        loc="lower right",
        fontsize=7.5,
        framealpha=0.9,
        ncol=1,
        borderpad=0.5,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PCA dimension 1", labelpad=3)
    ax.set_ylabel("PCA dimension 2", labelpad=3)
    ax.set_title(
        "Paraphrase Ensemble Cloud (PCA-2D, 8 focal docs)\n"
        "Stars = ensemble mean; diamonds = original; circles = paraphrases",
        pad=7,
    )

    out = os.path.join(OUT_DIR, "fig8_paraphrase_cloud.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
# Figure 9 -- K-Means confusion heatmap
# =============================================================================


def plot_confusion_heatmap():
    print("[fig9] K-Means confusion heatmap")
    try:
        emb_norm, labels_gt = _load_bank77_raw()
    except Exception as e:
        print(f"  Skipping: {e}")
        return

    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment

    n_classes = int(labels_gt.max()) + 1
    pred = KMeans(n_clusters=n_classes, random_state=42, n_init="auto").fit_predict(
        emb_norm
    )

    contingency = np.zeros((n_classes, n_classes))
    for p, t in zip(pred, labels_gt):
        contingency[p, t] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = contingency[row_ind][:, col_ind[np.argsort(row_ind)]]

    row_sums = matched.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_norm = matched / row_sums

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    order_h = leaves_list(linkage(pdist(conf_norm), method="average"))
    conf_sorted = conf_norm[order_h][:, order_h]

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    sns.heatmap(
        conf_sorted,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={
            "label": "Recall (fraction of class docs\n" "assigned to matched cluster)",
            "shrink": 0.6,
        },
    )
    ax.set_xlabel("Predicted cluster (matched to ground-truth intent)")
    ax.set_ylabel("Ground-truth intent class")
    ax.set_title(
        "K-Means Confusion Heatmap on Bank77\n"
        "(diagonal = per-class recall; hierarchically sorted)"
    )

    mean_recall = conf_norm.diagonal().mean()
    ax.text(
        0.02,
        0.02,
        f"Mean recall = {mean_recall:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        color="white",
        bbox=dict(facecolor="#333333", alpha=0.65, pad=2),
    )

    out = os.path.join(OUT_DIR, "fig9_confusion_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}  (mean recall = {mean_recall:.3f})")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    for fn in [
        plot_tsne,
        plot_llm_budget,
        plot_granularity,
        plot_results,
        plot_constraint_graph,
        plot_oracle_accuracy,
        plot_normalization_drift,
        plot_paraphrase_cloud,
        plot_confusion_heatmap,
    ]:
        try:
            fn()
        except Exception as exc:
            import traceback

            print(f"  ERROR in {fn.__name__}: {exc}")
            traceback.print_exc()
    print("\nDone. Check report/figures/ for output PDFs.")
