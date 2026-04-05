from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import frontmatter

from scripts.repo_sync import sync_repo

DOC_EXTENSIONS = {".md", ".mdx"}


@dataclass
class ParallelDocRecord:
    relative_path: str
    source_path: str
    target_path: str
    source_text: str
    target_text: str
    source_chars: int
    target_chars: int

    def as_json_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_docs_root(repo_dir: Path, docs_root: str | None = None) -> Path:
    if docs_root:
        root = Path(docs_root)
    else:
        root = repo_dir / "docs" / "source"
    if not root.exists():
        raise FileNotFoundError(f"Docs root not found: {root}")
    return root


def iter_locale_files(locale_root: Path) -> Iterable[Path]:
    for path in locale_root.rglob("*"):
        if not path.is_file() or path.suffix not in DOC_EXTENSIONS:
            continue
        rel = path.relative_to(locale_root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        yield rel


def build_locale_index(docs_root: Path, locale: str) -> dict[str, Path]:
    locale_root = docs_root / locale
    if not locale_root.exists():
        raise FileNotFoundError(f"Locale directory not found: {locale_root}")
    return {str(rel): locale_root / rel for rel in iter_locale_files(locale_root)}


def strip_markdown(text: str) -> str:
    body = frontmatter.loads(text).content
    body = re.sub(
        r"^\s*\*?This model was released on \d{4}-\d{2}-\d{2} and added to Hugging Face Transformers on \d{4}-\d{2}-\d{2}\.\*?\s*$",
        " ",
        body,
        flags=re.MULTILINE,
    )
    body = re.sub(r"^\s{0,3}#{1,6}\s+.*$", " ", body, flags=re.MULTILINE)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    body = re.sub(r"<--.*?-->", " ", body, flags=re.DOTALL)
    body = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
    body = re.sub(r"~~~.*?~~~", " ", body, flags=re.DOTALL)
    body = re.sub(r"</?[\w:-]+(?:\s+[^>]*?)?>", " ", body)
    body = re.sub(r"\[\[[^\]]+\]\]", " ", body)
    body = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
    body = re.sub(r"`([^`]+)`", r"\1", body)
    body = re.sub(r"^\s{0,3}[-*+]\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"^\s{0,3}\d+\.\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"[*_]{1,3}", "", body)
    body = "\n".join(line.strip() for line in body.splitlines())
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"[ \t]+", " ", body)
    return body.strip()


def build_parallel_records(
    docs_root: Path,
    source_locale: str = "en",
    target_locale: str = "ko",
    min_chars: int = 100,
) -> tuple[list[ParallelDocRecord], list[str], list[str]]:
    source_index = build_locale_index(docs_root, source_locale)
    target_index = build_locale_index(docs_root, target_locale)

    shared_paths = sorted(set(source_index) & set(target_index))
    source_only = sorted(set(source_index) - set(target_index))
    target_only = sorted(set(target_index) - set(source_index))

    records: list[ParallelDocRecord] = []
    for relative_path in shared_paths:
        source_text = strip_markdown(source_index[relative_path].read_text())
        target_text = strip_markdown(target_index[relative_path].read_text())
        if len(source_text) < min_chars or len(target_text) < min_chars:
            continue
        records.append(
            ParallelDocRecord(
                relative_path=relative_path,
                source_path=str(source_index[relative_path]),
                target_path=str(target_index[relative_path]),
                source_text=source_text,
                target_text=target_text,
                source_chars=len(source_text),
                target_chars=len(target_text),
            )
        )

    return records, source_only, target_only


def write_jsonl(records: list[ParallelDocRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_json_dict(), ensure_ascii=False) + "\n")


def write_csv(records: list[ParallelDocRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "source_path",
                "target_path",
                "source_text",
                "target_text",
                "source_chars",
                "target_chars",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_json_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a parallel translation corpus from localized Markdown docs."
    )
    parser.add_argument(
        "--repo-url",
        default="https://github.com/huggingface/transformers.git",
        help="Repository URL to clone or update.",
    )
    parser.add_argument(
        "--repo-dir",
        default=str(Path(".cache") / "huggingface-transformers"),
        help="Local repository directory.",
    )
    parser.add_argument(
        "--docs-root",
        default=None,
        help="Override docs root. Defaults to <repo-dir>/docs/source.",
    )
    parser.add_argument("--source-locale", default="en")
    parser.add_argument("--target-locale", default="ko")
    parser.add_argument(
        "--output",
        default=str(Path("data") / "transformers_en_ko_docs.jsonl"),
        help="Output path. Supported extensions: .jsonl, .csv",
    )
    parser.add_argument("--min-chars", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    output_path = Path(args.output)

    print(f"Syncing {args.repo_url} into {repo_dir}...")
    sync_repo(args.repo_url, repo_dir)

    docs_root = resolve_docs_root(repo_dir, args.docs_root)
    print(f"Building parallel corpus from {docs_root}...")
    records, source_only, target_only = build_parallel_records(
        docs_root,
        source_locale=args.source_locale,
        target_locale=args.target_locale,
        min_chars=args.min_chars,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        write_csv(records, output_path)
    else:
        write_jsonl(records, output_path)

    print(f"Wrote {len(records)} aligned documents to {output_path}")
    print(f"Unmatched {args.source_locale}-only files: {len(source_only)}")
    print(f"Unmatched {args.target_locale}-only files: {len(target_only)}")


if __name__ == "__main__":
    main()
