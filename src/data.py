import numpy as np
import os
import pickle
import json
import glob
import csv
from typing import List, Tuple, Any, Optional, Callable
from langchain_core.embeddings import Embeddings
from datasets import load_dataset as load_dataset_hf


def _balance_classes(
    docs: List[str], labels: List[int], raw_data: List[Any], max_samples: int
) -> Tuple[List[str], List[int], List[Any]]:
    """Balance classes by limiting samples per class."""
    if not max_samples or not labels:
        return docs, labels, raw_data

    labels_np = np.array(labels)
    indices = []
    for lbl in np.unique(labels_np):
        class_idx = np.where(labels_np == lbl)[0]
        indices.extend(
            np.random.choice(class_idx, min(len(class_idx), max_samples), replace=False)
        )
    indices.sort()

    balanced_docs = [docs[i] for i in indices]
    balanced_labels = [labels[i] for i in indices]
    balanced_raw = [raw_data[i] for i in indices] if raw_data else balanced_docs

    print(f"Balanced to {len(balanced_labels)} samples ({max_samples}/class)")
    return balanced_docs, balanced_labels, balanced_raw


def _get_embeddings(
    docs: List[str],
    cache_path: Optional[str],
    embedding_model: Embeddings,
    dataset_name: str,
) -> np.ndarray:
    """Get embeddings from cache or generate and cache them.

    Cache location: {cache_path}/{dataset_name}/embeddings.pkl
    The model class name is stored inside the pickle so stale caches
    (e.g. switching from OpenAI to sentence-transformers) are detected
    and regenerated automatically.
    """
    if not docs or not embedding_model:
        return np.array([])

    cache_file = None
    model_id = type(embedding_model).__name__
    if cache_path:
        dataset_dir = os.path.join(cache_path, dataset_name)
        cache_file = os.path.join(dataset_dir, "embeddings.pkl")

        try:
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                # Validate: cache stores {"model": str, "embeddings": np.ndarray}
                if isinstance(cached, dict) and cached.get("model") == model_id:
                    print(f"  Loaded embeddings from cache: {cache_file}")
                    return cached["embeddings"]
                else:
                    print(
                        f"  Cache model mismatch ({cached.get('model')} vs {model_id}), regenerating."
                    )
        except Exception as e:
            print(f"  Cache load failed: {e}")

    print(f"  Generating embeddings for {dataset_name} ({len(docs)} docs)...")
    try:
        embeddings = np.array(embedding_model.embed_documents(docs))
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump({"model": model_id, "embeddings": embeddings}, f)
            print(f"  Embeddings cached to: {cache_file}")
        return embeddings
    except Exception as e:
        print(f"  Embedding failed: {e}")
        return np.array([])


def _load_tweet_yin_wang(path: str) -> Tuple[List[str], List[int], List[str]]:
    """Load the Yin & Wang (2016) 89-class tweet intent dataset.

    Supports JSONL (each line: {"text": ..., "label": ...}),
    TSV (text<tab>label), and CSV (text,label) formats.
    Reads all files matching *.jsonl / *.tsv / *.csv under path/.
    """
    texts, labels_str = [], []

    # Try JSONL first (few-shot-clustering repo format)
    for fpath in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                text = entry.get("text") or entry.get("query") or ""
                label = str(entry.get("label") or entry.get("cluster_id") or "")
                if text and label:
                    texts.append(text)
                    labels_str.append(label)

    # Try TSV if nothing found — use pandas to handle header row correctly
    if not texts:
        import pandas as pd

        for fpath in sorted(glob.glob(os.path.join(path, "*.tsv"))):
            df = pd.read_csv(fpath, sep="\t")
            # Support both (text, label) and (label, text) column orders
            if "text" in df.columns and "label" in df.columns:
                texts.extend(df["text"].astype(str).tolist())
                labels_str.extend(df["label"].astype(str).tolist())
            elif len(df.columns) >= 2:
                texts.extend(df.iloc[:, 0].astype(str).tolist())
                labels_str.extend(df.iloc[:, 1].astype(str).tolist())

    # Try CSV if still nothing
    if not texts:
        for fpath in sorted(glob.glob(os.path.join(path, "*.csv"))):
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        texts.append(row[0])
                        labels_str.append(row[1])

    if not texts:
        raise FileNotFoundError(f"No valid tweet data files found at {path}")

    label_map = {v: i for i, v in enumerate(sorted(set(labels_str)))}
    labels = [label_map[lbl] for lbl in labels_str]
    print(
        f"Loaded Tweet (Yin & Wang): {len(texts)} samples, {len(label_map)} categories"
    )
    return texts, labels, texts


def _load_clinc() -> Tuple[List[str], List[int], List[str]]:
    """Load CLINC dataset."""
    try:
        dataset = load_dataset_hf("clinc_oos", "small")["test"]
        texts = dataset["text"]
        intents = dataset["intent"]

        # Filter out intent 42 (out-of-scope) and remap remaining intents
        filtered_data = [
            (text, intent) for text, intent in zip(texts, intents) if intent != 42
        ]
        filtered_texts, filtered_intents = zip(*filtered_data)

        intent_mapping = {
            intent: idx for idx, intent in enumerate(sorted(set(filtered_intents)))
        }
        remapped_intents = [intent_mapping[intent] for intent in filtered_intents]

        print(
            f"Loaded CLINC: {len(filtered_texts)} samples, {len(intent_mapping)} intents"
        )
        return list(filtered_texts), remapped_intents, list(filtered_texts)
    except Exception as e:
        print(f"Error loading CLINC dataset: {e}")
        return [], [], []


def _load_hf_dataset(
    dataset: str,
    text_field: str,
    label_field: str,
    filter_func: Optional[Callable] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """Load HuggingFace dataset with optional filtering."""
    try:
        data = load_dataset_hf(*dataset.split("/"))["test"]
        texts, labels = data[text_field], data[label_field]

        if filter_func:
            texts, labels = zip(*filter(filter_func, zip(texts, labels)))

        if isinstance(labels[0], str):
            label_map = {v: i for i, v in enumerate(set(labels))}
            labels = [label_map[lbl] for lbl in labels]

        return list(texts), list(labels), list(texts)
    except Exception as e:
        print(f"Error loading {dataset}: {e}")
        return [], [], []


def load_dataset(
    dataset_name: str,
    cache_path: Optional[str] = None,
    embedding_model: Optional[Embeddings] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load one of the supported datasets: bank77, clinc, tweet."""
    if dataset_name == "clinc":
        docs, labels, raw_data = _load_clinc()
    elif dataset_name == "tweet":
        local_path = os.path.join(
            os.path.dirname(__file__), "..", "datasets", "tweet_yin_wang"
        )
        local_path = os.path.normpath(local_path)
        if os.path.exists(local_path):
            docs, labels, raw_data = _load_tweet_yin_wang(local_path)
        else:
            print(
                f"WARNING: Yin & Wang (2016) 89-class tweet dataset not found at {local_path}"
            )
            print("  Falling back to cardiffnlp/tweet_eval/topic (20 classes).")
            print("  To get the correct dataset, run:")
            print(
                "    git clone --depth=1 https://github.com/viswavi/few-shot-clusteringLLM /tmp/fsc"
            )
            print("    cp -r /tmp/fsc/data/tweet datasets/tweet_yin_wang")
            docs, labels, raw_data = _load_hf_dataset(
                dataset="cardiffnlp/tweet_eval/topic",
                text_field="text",
                label_field="label",
            )
    elif dataset_name == "bank77":
        docs, labels, raw_data = _load_hf_dataset(
            dataset="banking77", text_field="text", label_field="label"
        )
    else:
        print(f"Unknown dataset: {dataset_name}. Supported: bank77, clinc, tweet")
        return np.array([]), np.array([]), []

    docs, labels, raw_data = _balance_classes(docs, labels, raw_data, max_samples)

    features = _get_embeddings(docs, cache_path, embedding_model, dataset_name)

    print(f"Loaded {dataset_name}: {len(docs)} samples, {len(set(labels))} classes")
    return features, np.array(labels), docs
