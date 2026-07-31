"""Transcript-level and sentence-level financial language features."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def extract_structural_text(structured_list) -> tuple[str, str]:
    """Separate an earnings call into prepared remarks and Q&A text."""
    presentation: list[str] = []
    qa: list[str] = []
    in_qa = False
    if not isinstance(structured_list, list):
        return "", ""
    for turn in structured_list:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip().lower()
        text = str(turn.get("text", "")).strip()
        if "question-and-answer" in speaker or ("operator" in speaker and "question" in text.lower()):
            in_qa = True
            continue
        (qa if in_qa else presentation).append(text)
    return " ".join(presentation), " ".join(qa)


def split_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split transcript text without requiring a heavyweight NLP package."""
    if not isinstance(text, str):
        return []
    sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    return [sentence for sentence in sentences if len(sentence) >= min_chars]


def _safe_slope(values: np.ndarray) -> float:
    if len(values) < 2 or np.allclose(values, values[0]):
        return 0.0
    x = np.linspace(-1.0, 1.0, len(values))
    return float(np.polyfit(x, values, 1)[0])


def aggregate_sentiment_features(
    probabilities: np.ndarray,
    prefix: str,
    positive_index: int = 0,
    negative_index: int = 1,
    neutral_index: int = 2,
) -> dict[str, float]:
    """Aggregate per-sentence FinBERT probabilities into stable features.

    A sentence is considered positive/negative when its corresponding class
    probability is greater than the other directional class probability.
    Entropy is averaged over sentences and the slope uses sentence position.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 3 or len(probabilities) == 0:
        return {f"{prefix}_{name}": np.nan for name in _feature_names()}
    probabilities = np.clip(probabilities[:, :3], 1e-8, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    positive = probabilities[:, positive_index]
    negative = probabilities[:, negative_index]
    neutral = probabilities[:, neutral_index]
    net = positive - negative
    entropy = -(probabilities * np.log(probabilities)).sum(axis=1)
    thirds = np.array_split(net, 3)

    # Very short sections can produce an empty middle or end group.  Keep the
    # feature unavailable for that group without emitting noisy NumPy warnings.
    def _group_mean(group: np.ndarray) -> float:
        return float(np.mean(group)) if len(group) else np.nan

    result = {
        "sent_mean": float(net.mean()),
        "sent_std": float(net.std(ddof=0)),
        "sent_p10": float(np.quantile(net, 0.10)),
        "sent_p90": float(np.quantile(net, 0.90)),
        "pos_mean": float(positive.mean()),
        "neg_mean": float(negative.mean()),
        "neutral_mean": float(neutral.mean()),
        "pos_frac": float((positive > negative).mean()),
        "neg_frac": float((negative > positive).mean()),
        "entropy": float(entropy.mean()),
        "begin_mean": _group_mean(thirds[0]),
        "middle_mean": _group_mean(thirds[1]),
        "end_mean": _group_mean(thirds[2]),
        "slope": _safe_slope(net),
        "n_sentences": float(len(net)),
    }
    return {f"{prefix}_{key}": value for key, value in result.items()}


def _feature_names() -> tuple[str, ...]:
    return (
        "sent_mean", "sent_std", "sent_p10", "sent_p90", "pos_mean", "neg_mean",
        "neutral_mean", "pos_frac", "neg_frac", "entropy", "begin_mean",
        "middle_mean", "end_mean", "slope", "n_sentences",
    )


def build_sentence_feature_row(
    presentation_text: str,
    qa_text: str,
    scorer: Callable[[Sequence[str]], np.ndarray],
) -> dict[str, float]:
    """Create presentation, Q&A, and divergence features for one call."""
    sections = {
        "pres": split_sentences(presentation_text),
        "qa": split_sentences(qa_text),
    }
    output: dict[str, float] = {}
    for prefix, sentences in sections.items():
        probabilities = scorer(sentences) if sentences else np.empty((0, 3))
        output.update(aggregate_sentiment_features(probabilities, prefix))

    for key in ("sent_mean", "sent_std", "entropy", "neg_frac", "pos_frac", "slope"):
        output[f"qa_minus_pres_{key}"] = output[f"qa_{key}"] - output[f"pres_{key}"]
    return output


def make_finbert_sentence_scorer(tokenizer, model, device: str = "cpu", batch_size: int = 32, max_length: int = 128):
    """Return a batched scorer compatible with ``build_sentence_feature_row``."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in Colab
        raise ImportError("Install torch and transformers to use FinBERT sentence scoring.") from exc

    model.eval()

    def score(sentences: Sequence[str]) -> np.ndarray:
        if not sentences:
            return np.empty((0, 3))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(sentences), batch_size):
                batch = list(sentences[start : start + batch_size])
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                probabilities = torch.softmax(model(**encoded).logits, dim=-1)
                outputs.append(probabilities.detach().cpu().numpy())
        return np.vstack(outputs)

    return score


def build_sentence_feature_frame(
    frame: pd.DataFrame,
    scorer: Callable[[Sequence[str]], np.ndarray],
    presentation_col: str = "pres_clean",
    qa_col: str = "qa_clean",
    row_batch_size: int | None = None,
) -> pd.DataFrame:
    """Add sentence-level features using one continuous batched inference pass.

    Scoring each presentation and Q&A separately creates hundreds of small
    tokenizer/model calls.  Flattening the sentences first keeps the same
    features while allowing the GPU to process a continuous stream of batches.
    """
    if row_batch_size is not None and row_batch_size > 0 and len(frame) > row_batch_size:
        chunks = [
            build_sentence_feature_frame(
                frame.iloc[start : start + row_batch_size],
                scorer,
                presentation_col=presentation_col,
                qa_col=qa_col,
                row_batch_size=None,
            )
            for start in range(0, len(frame), row_batch_size)
        ]
        return pd.concat(chunks, axis=0)

    section_sentences = []
    all_sentences: list[str] = []
    for _, row in frame.iterrows():
        pres = split_sentences(row.get(presentation_col, ""))
        qa = split_sentences(row.get(qa_col, ""))
        section_sentences.append((pres, qa))
        all_sentences.extend(pres)
        all_sentences.extend(qa)

    all_probabilities = scorer(all_sentences) if all_sentences else np.empty((0, 3))
    cursor = 0
    rows = []
    for pres, qa in section_sentences:
        pres_probabilities = all_probabilities[cursor : cursor + len(pres)]
        cursor += len(pres)
        qa_probabilities = all_probabilities[cursor : cursor + len(qa)]
        cursor += len(qa)

        output: dict[str, float] = {}
        output.update(aggregate_sentiment_features(pres_probabilities, "pres"))
        output.update(aggregate_sentiment_features(qa_probabilities, "qa"))
        for key in ("sent_mean", "sent_std", "entropy", "neg_frac", "pos_frac", "slope"):
            output[f"qa_minus_pres_{key}"] = output[f"qa_{key}"] - output[f"pres_{key}"]
        rows.append(output)

    features = pd.DataFrame(rows, index=frame.index)
    return pd.concat([frame.copy(), features], axis=1)


def load_lm_lexicons(path: str | Path) -> dict[str, set[str]]:
    """Load Loughran–McDonald category columns from a local CSV.

    The CSV should contain a ``Word`` column and any subset of category
    columns such as ``Uncertainty``, ``Litigious``, ``Constraining``,
    ``Weak_Modal`` and ``Strong_Modal``. Keeping the lexicon as a local input
    makes the feature build reproducible without downloading data at runtime.
    """
    table = pd.read_csv(path)
    word_col = next((column for column in table.columns if column.lower() == "word"), None)
    if word_col is None:
        raise ValueError("The financial lexicon CSV must contain a Word column.")
    # The LM file also contains metadata columns such as Seq_num, Word Count,
    # and Source.  Only word-category indicator columns should become lexicons.
    category_names = {
        "negative", "positive", "uncertainty", "litigious",
        "strongmodal", "weakmodal", "constraining", "complexity",
    }
    lexicons = {}
    for column in table.columns:
        if column == word_col:
            continue
        normalized = re.sub(r"[^a-z]", "", str(column).lower())
        if normalized not in category_names:
            continue
        values = pd.to_numeric(table[column], errors="coerce").fillna(0)
        words = table.loc[values > 0, word_col].astype(str).str.lower()
        lexicons[normalized] = set(words)
    if not lexicons:
        raise ValueError("The financial lexicon CSV contains no recognized LM category columns.")
    return lexicons


def dictionary_features(text: str, lexicons: Mapping[str, set[str]], prefix: str = "all") -> dict[str, float]:
    """Compute normalized financial-language category counts."""
    tokens = [token.lower() for token in _TOKEN_RE.findall(text or "")]
    denominator = max(1, len(tokens))
    return {
        f"{prefix}_lm_{category}_rate": float(sum(token in words for token in tokens) / denominator)
        for category, words in lexicons.items()
    } | {f"{prefix}_lm_token_count": float(len(tokens))}


def add_dictionary_features(
    frame: pd.DataFrame,
    lexicons: Mapping[str, set[str]],
    presentation_col: str = "pres_clean",
    qa_col: str = "qa_clean",
) -> pd.DataFrame:
    """Add presentation, Q&A, and difference dictionary features."""
    rows = []
    for _, row in frame.iterrows():
        pres = dictionary_features(row.get(presentation_col, ""), lexicons, "pres")
        qa = dictionary_features(row.get(qa_col, ""), lexicons, "qa")
        combined = {**pres, **qa}
        for key in lexicons:
            combined[f"qa_minus_pres_lm_{key}_rate"] = qa[f"qa_lm_{key}_rate"] - pres[f"pres_lm_{key}_rate"]
        rows.append(combined)
    return pd.concat([frame.copy(), pd.DataFrame(rows, index=frame.index)], axis=1)


def earnings_language_features(
    presentation_text: str,
    qa_text: str,
) -> dict[str, float]:
    """Create transcript-derived earnings and guidance proxy features.

    These are not EPS surprises. They measure whether the call discusses
    per-share results, beats/misses, expectations, or guidance direction.
    All text is available at call time, so these features are causal for the
    post-call target.
    """
    pres = str(presentation_text or "").lower()
    qa = str(qa_text or "").lower()
    combined = f"{pres} {qa}"

    def count(pattern: str, text: str = combined) -> float:
        return float(len(re.findall(pattern, text, flags=re.IGNORECASE)))

    token_count = max(1, len(_TOKEN_RE.findall(combined)))
    return {
        "eps_mention_count": count(r"\b(?:eps|e\.p\.s\.|earnings per share|per-share)\b"),
        "eps_mention_rate": count(r"\b(?:eps|e\.p\.s\.|earnings per share|per-share)\b") / token_count,
        "beat_language_count": count(r"\b(?:beat|beats|beating|exceeded|outperformed)\b"),
        "miss_language_count": count(r"\b(?:miss|missed|missing|fell short|shortfall)\b"),
        "above_expectations_count": count(r"\b(?:above|ahead of|better than) (?:our |the )?expectations?\b"),
        "below_expectations_count": count(r"\b(?:below|behind|worse than) (?:our |the )?expectations?\b"),
        "guidance_up_count": count(r"\b(?:(?:raise|raised|raising|increased|increase|upward)\b[^.]{0,80}\b(?:guidance|outlook|forecast)|(?:guidance|outlook|forecast)\b[^.]{0,80}\b(?:raise|raised|raising|increased|increase|upward))\b"),
        "guidance_down_count": count(r"\b(?:(?:lower|lowered|lowering|reduced|reduce|downward)\b[^.]{0,80}\b(?:guidance|outlook|forecast)|(?:guidance|outlook|forecast)\b[^.]{0,80}\b(?:lower|lowered|lowering|reduced|reduce|downward))\b"),
        "guidance_maintained_count": count(r"\b(?:(?:reaffirm|reaffirmed|maintain|maintained|unchanged)\b[^.]{0,80}\b(?:guidance|outlook|forecast)|(?:guidance|outlook|forecast)\b[^.]{0,80}\b(?:reaffirm|reaffirmed|maintain|maintained|unchanged))\b"),
        "forward_language_count": count(r"\b(?:expect|expects|expecting|anticipate|anticipated|outlook|guidance|forecast)\b"),
    }


def add_earnings_language_features(
    frame: pd.DataFrame,
    presentation_col: str = "pres_clean",
    qa_col: str = "qa_clean",
) -> pd.DataFrame:
    """Add transcript-derived earnings and guidance proxy columns."""
    rows = [
        earnings_language_features(row.get(presentation_col, ""), row.get(qa_col, ""))
        for _, row in frame.iterrows()
    ]
    return pd.concat([frame.copy(), pd.DataFrame(rows, index=frame.index)], axis=1)
