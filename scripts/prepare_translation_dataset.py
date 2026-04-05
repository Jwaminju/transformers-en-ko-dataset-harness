from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TranslationPair:
    relative_path: str
    chunk_id: int
    text: str
    target: str
    source_chars: int
    target_chars: int

    def as_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RejectedPair:
    relative_path: str
    chunk_id: int
    reason: str
    github_url: str
    text: str
    target: str
    source_chars: int
    target_chars: int

    def as_json_dict(self) -> dict[str, object]:
        return asdict(self)


CODE_LIKE_RE = re.compile(
    r"`[^`]+`|(?:^|\s)--[\w-]+|(?:^|\s)[A-Z][A-Z0-9_]{2,}|[A-Za-z_][\w./-]*\.[A-Za-z0-9_./-]+|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+"
)
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
HANGUL_RE = re.compile(r"[가-힣]")
LONG_ENGLISH_RUN_RE = re.compile(r"(?:\b[A-Za-z]+(?:'[A-Za-z]+)?\b[\s,;()]*){5,}")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_blocks(text: str, min_block_chars: int) -> list[str]:
    blocks = []
    for raw_block in text.split("\n\n"):
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        if len(lines) > 1:
            for line in lines:
                if len(line) >= min_block_chars:
                    blocks.append(line)
            continue
        block = "\n".join(lines).strip()
        if len(block) >= min_block_chars:
            blocks.append(block)
    return blocks


def alignment_cost(source_group: list[str], target_group: list[str]) -> float:
    source_text = "\n\n".join(source_group)
    target_text = "\n\n".join(target_group)
    source_chars = len(source_text)
    target_chars = len(target_text)
    ratio_penalty = abs(math.log((source_chars + 1) / (target_chars + 1)))
    group_penalty = abs(len(source_group) - len(target_group)) * 0.2
    return ratio_penalty + group_penalty


def align_blocks(
    source_blocks: list[str],
    target_blocks: list[str],
    max_group_size: int = 3,
    max_step_cost: float = 1.35,
) -> list[tuple[str, str]]:
    if not source_blocks or not target_blocks:
        return []

    source_len = len(source_blocks)
    target_len = len(target_blocks)
    inf = float("inf")
    dp = [[inf] * (target_len + 1) for _ in range(source_len + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (target_len + 1) for _ in range(source_len + 1)]
    dp[0][0] = 0.0

    for i in range(source_len + 1):
        for j in range(target_len + 1):
            if dp[i][j] == inf:
                continue
            for source_step in range(1, max_group_size + 1):
                for target_step in range(1, max_group_size + 1):
                    next_i = i + source_step
                    next_j = j + target_step
                    if next_i > source_len or next_j > target_len:
                        continue
                    cost = alignment_cost(source_blocks[i:next_i], target_blocks[j:next_j])
                    if cost > max_step_cost:
                        continue
                    new_cost = dp[i][j] + cost
                    if new_cost < dp[next_i][next_j]:
                        dp[next_i][next_j] = new_cost
                        back[next_i][next_j] = (source_step, target_step)

    if back[source_len][target_len] is None:
        return []

    aligned: list[tuple[str, str]] = []
    i = source_len
    j = target_len
    while i > 0 and j > 0:
        steps = back[i][j]
        if steps is None:
            return []
        source_step, target_step = steps
        prev_i = i - source_step
        prev_j = j - target_step
        aligned.append(
            (
                "\n\n".join(source_blocks[prev_i:i]),
                "\n\n".join(target_blocks[prev_j:j]),
            )
        )
        i = prev_i
        j = prev_j

    if i != 0 or j != 0:
        return []

    aligned.reverse()
    return aligned


def mask_code_like_spans(text: str) -> str:
    return CODE_LIKE_RE.sub(" ", text)


def build_github_url(relative_path: str) -> str:
    return f"https://github.com/huggingface/transformers/blob/main/docs/source/ko/{relative_path}"


def detect_target_noise(target_text: str) -> str | None:
    masked = mask_code_like_spans(target_text)
    english_words = ENGLISH_WORD_RE.findall(masked)
    hangul_count = len(HANGUL_RE.findall(masked))
    ascii_letter_count = sum(1 for char in masked if char.isascii() and char.isalpha())

    if hangul_count < 20 and len(english_words) >= 5:
        return "mostly_english_target"
    if ascii_letter_count > hangul_count * 0.6 and len(english_words) >= 6:
        return "high_english_ratio"
    if LONG_ENGLISH_RUN_RE.search(masked) and hangul_count < max(len(english_words) * 2, 24):
        return "long_english_run"
    return None


def sentence_count(text: str) -> int:
    normalized = re.sub(r"([.!?])([\"'])?\s+", r"\1\n", text)
    parts = [part.strip() for part in normalized.splitlines() if part.strip()]
    return max(len(parts), 1)


def detect_structural_mismatch(
    aligned_chunks: list[tuple[str, str]],
    sentence_ratio_threshold: float = 2.2,
    first_chunk_length_ratio_threshold: float = 1.9,
) -> str | None:
    if not aligned_chunks:
        return "alignment_failed"

    source_first, target_first = aligned_chunks[0]
    source_sentences = sentence_count(source_first)
    target_sentences = sentence_count(target_first)
    sentence_ratio = max(source_sentences, target_sentences) / max(min(source_sentences, target_sentences), 1)
    length_ratio = max(len(source_first), len(target_first)) / max(min(len(source_first), len(target_first)), 1)

    if sentence_ratio >= sentence_ratio_threshold:
        return "structural_mismatch"
    if length_ratio >= first_chunk_length_ratio_threshold and sentence_ratio >= 1.4:
        return "structural_mismatch"
    return None


def build_translation_pairs(
    rows: list[dict[str, object]],
    min_block_chars: int = 30,
    min_chunk_chars: int = 80,
    max_length_ratio: float = 3.0,
    max_group_size: int = 3,
    max_step_cost: float = 1.35,
    sentence_ratio_threshold: float = 2.2,
    first_chunk_length_ratio_threshold: float = 1.9,
) -> tuple[list[TranslationPair], list[RejectedPair]]:
    pairs: list[TranslationPair] = []
    rejected_pairs: list[RejectedPair] = []

    for row in rows:
        relative_path = str(row["relative_path"])
        source_text = str(row["source_text"])
        target_text = str(row["target_text"])

        source_blocks = split_blocks(source_text, min_block_chars=min_block_chars)
        target_blocks = split_blocks(target_text, min_block_chars=min_block_chars)
        if not source_blocks or not target_blocks:
            continue

        aligned_chunks = align_blocks(
            source_blocks,
            target_blocks,
            max_group_size=max_group_size,
            max_step_cost=max_step_cost,
        )
        if not aligned_chunks:
            rejected_pairs.append(
                RejectedPair(
                    relative_path=relative_path,
                    chunk_id=0,
                    reason="alignment_failed",
                    github_url=build_github_url(relative_path),
                    text=source_text,
                    target=target_text,
                    source_chars=len(source_text),
                    target_chars=len(target_text),
                )
            )
            continue

        mismatch_reason = detect_structural_mismatch(
            aligned_chunks,
            sentence_ratio_threshold=sentence_ratio_threshold,
            first_chunk_length_ratio_threshold=first_chunk_length_ratio_threshold,
        )
        if mismatch_reason:
            rejected_pairs.append(
                RejectedPair(
                    relative_path=relative_path,
                    chunk_id=0,
                    reason=mismatch_reason,
                    github_url=build_github_url(relative_path),
                    text=aligned_chunks[0][0],
                    target=aligned_chunks[0][1],
                    source_chars=len(aligned_chunks[0][0]),
                    target_chars=len(aligned_chunks[0][1]),
                )
            )
            continue

        for chunk_id, (source_chunk, target_chunk) in enumerate(aligned_chunks, start=1):
            if len(source_chunk) < min_chunk_chars or len(target_chunk) < min_chunk_chars:
                continue
            length_ratio = len(source_chunk) / max(len(target_chunk), 1)
            if length_ratio > max_length_ratio or length_ratio < 1 / max_length_ratio:
                rejected_pairs.append(
                    RejectedPair(
                        relative_path=relative_path,
                        chunk_id=chunk_id,
                        reason="length_ratio",
                        github_url=build_github_url(relative_path),
                        text=source_chunk,
                        target=target_chunk,
                        source_chars=len(source_chunk),
                        target_chars=len(target_chunk),
                    )
                )
                continue
            noise_reason = detect_target_noise(target_chunk)
            if noise_reason:
                rejected_pairs.append(
                    RejectedPair(
                        relative_path=relative_path,
                        chunk_id=chunk_id,
                        reason=noise_reason,
                        github_url=build_github_url(relative_path),
                        text=source_chunk,
                        target=target_chunk,
                        source_chars=len(source_chunk),
                        target_chars=len(target_chunk),
                    )
                )
                continue
            pairs.append(
                TranslationPair(
                    relative_path=relative_path,
                    chunk_id=chunk_id,
                    text=source_chunk,
                    target=target_chunk,
                    source_chars=len(source_chunk),
                    target_chars=len(target_chunk),
                )
            )

    return pairs, rejected_pairs


def write_jsonl(rows: list[TranslationPair] | list[RejectedPair], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_json_dict(), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert aligned document pairs into chunked translation training examples."
    )
    parser.add_argument(
        "--input",
        default=str(Path("data") / "transformers_en_ko_docs.jsonl"),
        help="Document-level JSONL produced by build_translation_corpus.py",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "transformers_en_ko_pairs.jsonl"),
        help="Chunked training pairs in JSONL format",
    )
    parser.add_argument(
        "--rejected-output",
        default=str(Path("data") / "transformers_en_ko_rejected_pairs.jsonl"),
        help="Rejected chunk pairs with reasons and GitHub links",
    )
    parser.add_argument("--min-block-chars", type=int, default=30)
    parser.add_argument("--min-chunk-chars", type=int, default=80)
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    parser.add_argument("--max-group-size", type=int, default=3)
    parser.add_argument("--max-step-cost", type=float, default=1.35)
    parser.add_argument("--sentence-ratio-threshold", type=float, default=2.2)
    parser.add_argument("--first-chunk-length-ratio-threshold", type=float, default=1.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rejected_output_path = Path(args.rejected_output)

    rows = read_jsonl(input_path)
    pairs, rejected_pairs = build_translation_pairs(
        rows,
        min_block_chars=args.min_block_chars,
        min_chunk_chars=args.min_chunk_chars,
        max_length_ratio=args.max_length_ratio,
        max_group_size=args.max_group_size,
        max_step_cost=args.max_step_cost,
        sentence_ratio_threshold=args.sentence_ratio_threshold,
        first_chunk_length_ratio_threshold=args.first_chunk_length_ratio_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(pairs, output_path)
    write_jsonl(rejected_pairs, rejected_output_path)
    print(f"Wrote {len(pairs)} training pairs to {output_path}")
    print(f"Wrote {len(rejected_pairs)} rejected pairs to {rejected_output_path}")


if __name__ == "__main__":
    main()
