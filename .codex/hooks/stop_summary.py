from __future__ import annotations

import json

from common import repo_root


def main() -> None:
    status_path = repo_root() / "hf_dataset_repo/metadata/build_status.json"
    if not status_path.exists():
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    print(
        "Dataset package status: "
        f"{status.get('status', 'unknown')} "
        f"(train={status.get('train_rows', '?')}, "
        f"validation={status.get('validation_rows', '?')}, "
        f"stress={status.get('stress_test_rows', '?')})"
    )


if __name__ == "__main__":
    main()
