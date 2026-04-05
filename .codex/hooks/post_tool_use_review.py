from __future__ import annotations

from common import fail, guess_command, load_event, repo_root


def ensure_paths_exist(paths: list[str]) -> None:
    root = repo_root()
    missing = [str(root / rel_path) for rel_path in paths if not (root / rel_path).exists()]
    if missing:
        fail("Expected artifacts are missing:\n- " + "\n- ".join(missing))


def main() -> None:
    event = load_event()
    command = guess_command(event)

    checks = {
        "scripts/split_train_validation.py": [
            "data/transformers_en_ko_train_split.jsonl",
            "data/transformers_en_ko_validation_clean.jsonl",
        ],
        "scripts/build_hf_dataset_repo.py": [
            "hf_dataset_repo/README.md",
            "hf_dataset_repo/AGENTS.md",
            "hf_dataset_repo/data/train.jsonl",
            "hf_dataset_repo/data/validation_clean.jsonl",
            "hf_dataset_repo/data/stress_test_contaminated.jsonl",
            "hf_dataset_repo/metadata/build_status.json",
            "hf_dataset_repo/docs/translation_dataset_harness.md",
        ],
    }

    for script_name, paths in checks.items():
        if script_name in command:
            ensure_paths_exist(paths)


if __name__ == "__main__":
    main()
