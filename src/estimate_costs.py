"""
Estimate LLM token usage and cost from existing output CSVs.

Uses tiktoken (gpt-4o tokenizer, close enough for gpt-4.1-nano) to count
input and output tokens from the saved prompt/response columns, then applies
published gpt-4.1-nano pricing.

Usage:
    python -m src.estimate_costs
    python -m src.estimate_costs --datasets bank77,clinc,tweet
"""

import os
import argparse
import pandas as pd
import tiktoken
from src.config import RESULTS_ROOT

# GPT-4.1-nano pricing (USD per token)
_PRICE_INPUT = 0.100 / 1_000_000
_PRICE_CACHED = 0.025 / 1_000_000
_PRICE_OUTPUT = 0.400 / 1_000_000

_ENC = tiktoken.encoding_for_model("gpt-4o")  # same BPE family as gpt-4.1


def _count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Per-method extractors
# ---------------------------------------------------------------------------


def _pairwise(df: pd.DataFrame) -> tuple[int, int]:
    inp = df["full_prompt"].apply(_count).sum()
    out = (
        df["raw_llm_response"].apply(_count).sum()
        if "raw_llm_response" in df.columns
        else df["constraint_type"].apply(_count).sum()
    )
    return int(inp), int(out)


def _normalization(df: pd.DataFrame) -> tuple[int, int]:
    # No prompt column saved — estimate from text length: input ≈ prompt template + original
    # Prompt ~50 tokens overhead + original tokens; output = normalized tokens
    orig_tokens = df["original"].apply(_count).sum()
    norm_tokens = df["normalized"].apply(_count).sum()
    prompt_overhead = 50 * len(df)
    return int(orig_tokens + prompt_overhead), int(norm_tokens)


def _paraphrase(df: pd.DataFrame) -> tuple[int, int]:
    orig_tokens = df["original"].apply(_count).sum()
    para_tokens = df["paraphrases"].apply(_count).sum()
    prompt_overhead = 60 * len(df)
    return int(orig_tokens + prompt_overhead), int(para_tokens)


def _keyphrase(df: pd.DataFrame) -> tuple[int, int]:
    doc_tokens = df["document_text"].apply(_count).sum()
    kp_tokens = df["generated_keyphrases"].apply(_count).sum()
    prompt_overhead = 50 * len(df)
    return int(doc_tokens + prompt_overhead), int(kp_tokens)


def _correction(df: pd.DataFrame) -> tuple[int, int]:
    inp = df["full_prompt"].apply(_count).sum() if "full_prompt" in df.columns else 0
    out = df["llm_answer"].apply(_count).sum() if "llm_answer" in df.columns else 0
    return int(inp), int(out)


def _clusterllm(df: pd.DataFrame) -> tuple[int, int]:
    # anchor/option_a/option_b/llm_response columns
    inp = 0
    out = 0
    for col in ["anchor", "option_a", "option_b"]:
        if col in df.columns:
            inp += df[col].apply(_count).sum()
    if "llm_response" in df.columns:
        out = df["llm_response"].apply(_count).sum()
    # add prompt template overhead ~80 tokens/call
    inp += 80 * len(df)
    return int(inp), int(out)


_EXTRACTORS = {
    "pairwise_queries_output.csv": ("PCKMeans", _pairwise),
    "correction_queries_output.csv": ("LLM Correction", _correction),
    "keyphrase_expansions_output.csv": ("Keyphrase", _keyphrase),
    "normalization_output.csv": ("LLM Normalization", _normalization),
    "paraphrase_output.csv": ("LLM Paraphrase", _paraphrase),
    "clusterllm_output.csv": ("ClusterLLM", _clusterllm),
}


def estimate_dataset(dataset: str) -> list[dict]:
    dataset_dir = os.path.join(RESULTS_ROOT, dataset)
    rows = []
    for filename, (method, extractor) in _EXTRACTORS.items():
        fpath = os.path.join(dataset_dir, filename)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath)
            inp, out = extractor(df)
            calls = len(df)
            cost = inp * _PRICE_INPUT + out * _PRICE_OUTPUT
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "calls": calls,
                    "input_tokens": inp,
                    "cached_tokens": 0,
                    "output_tokens": out,
                    "cost_usd": round(cost, 6),
                }
            )
        except Exception as e:
            print(f"  Warning: could not process {fpath}: {e}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="bank77,clinc,tweet")
    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_rows = []
    for dataset in datasets:
        all_rows.extend(estimate_dataset(dataset))

    if not all_rows:
        print("No data found.")
        return

    summary_path = os.path.join(RESULTS_ROOT, "llm_costs_summary.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(summary_path, index=False)
    print(f"Cost summary saved to {summary_path}\n")

    print(
        f"  {'Method':<22} {'Dataset':<8} {'Calls':>6}  {'Input Tok':>10}  {'Output Tok':>10}  {'Cost $':>8}"
    )
    print(f"  {'-'*22} {'-'*8} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
    total_cost = 0.0
    for _, row in df.iterrows():
        print(
            f"  {row['method']:<22} {row['dataset']:<8} {row['calls']:>6}  "
            f"{row['input_tokens']:>10,}  {row['output_tokens']:>10,}  {row['cost_usd']:>8.4f}"
        )
        total_cost += row["cost_usd"]
    print(f"\n  Total estimated cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
