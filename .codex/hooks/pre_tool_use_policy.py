from __future__ import annotations

import json

from common import fail, guess_command, load_event, repo_root, warn


DESTRUCTIVE_PATTERNS = [
    "git reset --hard",
    "git checkout --",
    "rm -rf",
    "rm -fr",
]


def main() -> None:
    event = load_event()
    command = guess_command(event)
    lowered = command.lower()

    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern in lowered:
            fail(f"Blocked by repo policy: destructive command detected: {pattern}")

    if "scripts/build_hf_dataset_repo.py" in command:
        review_paths = [
            repo_root() / "data/review/train_clean_review.md",
            repo_root() / "data/review/eval_contaminated_review.md",
            repo_root() / "data/review/clean_suspect_review.md",
        ]
        missing_reviews = [str(path) for path in review_paths if not path.exists()]
        if missing_reviews:
            warn(
                "Dataset packaging is running without all review artifacts present:\n- "
                + "\n- ".join(missing_reviews)
            )

    if "scripts/upload_hf_dataset.py" not in command:
        return

    root = repo_root()
    status_path = root / "hf_dataset_repo/metadata/build_status.json"
    if not status_path.exists():
        fail(f"Blocked by repo policy: missing build status file: {status_path}")

    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get("status") != "passed":
        fail(
            "Blocked by repo policy: dataset package status is not 'passed'. "
            f"Current status: {status.get('status', 'unknown')}"
        )

    required_paths = [
        root / "hf_dataset_repo/data/train.jsonl",
        root / "hf_dataset_repo/data/validation_clean.jsonl",
        root / "hf_dataset_repo/data/stress_test_contaminated.jsonl",
        root / "hf_dataset_repo/docs/translation_dataset_harness.md",
        root / "hf_dataset_repo/metadata/blacklist.txt",
        root / "data/review/train_clean_review.md",
        root / "data/review/eval_contaminated_review.md",
        root / "data/review/clean_suspect_review.md",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        fail("Blocked by repo policy: missing package artifacts:\n- " + "\n- ".join(missing))


if __name__ == "__main__":
    main()
