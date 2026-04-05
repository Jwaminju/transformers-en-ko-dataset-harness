from __future__ import annotations

from common import repo_root


def main() -> None:
    root = repo_root()
    print(
        "\n".join(
            [
                "Translation dataset harness loaded.",
                f"- Read: {root / 'AGENTS.md'}",
                f"- Read: {root / 'docs/translation_dataset_harness.md'}",
                f"- Review artifacts: {root / 'data/review'}",
                f"- Guardrails: {root / '.codex/hooks.json'}",
            ]
        )
    )


if __name__ == "__main__":
    main()
