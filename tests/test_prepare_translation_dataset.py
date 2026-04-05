from scripts.prepare_translation_dataset import (
    align_blocks,
    build_translation_pairs,
    detect_structural_mismatch,
    detect_target_noise,
    split_blocks,
)


def test_split_blocks_drops_short_sections():
    text = "short\n\nThis is a long enough block.\n\nAnother usable block."
    assert split_blocks(text, min_block_chars=20) == [
        "This is a long enough block.",
        "Another usable block.",
    ]
def test_align_blocks_preserves_order():
    source_blocks = ["one", "two", "three", "four"]
    target_blocks = ["하나", "둘", "셋", "넷"]
    assert align_blocks(source_blocks, target_blocks, max_group_size=1) == [
        ("one", "하나"),
        ("two", "둘"),
        ("three", "셋"),
        ("four", "넷"),
    ]


def test_build_translation_pairs_creates_chunked_examples():
    rows = [
        {
            "relative_path": "guide/intro.md",
            "source_text": "First English paragraph is long enough.\n\nSecond English paragraph is also long enough.",
            "target_text": "첫 번째 한국어 문단은 충분히 깁니다.\n\n두 번째 한국어 문단도 충분히 깁니다.",
        }
    ]

    pairs, rejected = build_translation_pairs(
        rows,
        min_block_chars=10,
        min_chunk_chars=10,
        max_group_size=1,
    )

    assert len(pairs) == 2
    assert rejected == []
    assert pairs[0].relative_path == "guide/intro.md"
    assert pairs[0].chunk_id == 1
    assert "First English" in pairs[0].text
    assert "첫 번째 한국어" in pairs[0].target


def test_build_translation_pairs_skips_bad_length_ratio():
    rows = [
        {
            "relative_path": "guide/noisy.md",
            "source_text": "A very long English paragraph that keeps going for alignment filtering.\n\nAnother long English paragraph.",
            "target_text": "매우 긴 한국어 문단이지만 길이 차이를 충분히 만들기 위해 짧게 유지합니다.",
        }
    ]

    pairs, rejected = build_translation_pairs(
        rows,
        min_block_chars=3,
        min_chunk_chars=3,
        max_length_ratio=2.0,
    )

    assert pairs == []
    assert rejected[0].reason in {"length_ratio", "alignment_failed", "structural_mismatch"}


def test_detect_target_noise_ignores_code_like_english():
    assert detect_target_noise("명령어는 `accelerate launch --num_processes 2 train.py` 를 사용하세요.") is None


def test_detect_target_noise_flags_untranslated_english_sentence():
    reason = detect_target_noise("Then launch your training with additional arguments if needed.")
    assert reason in {"mostly_english_target", "high_english_ratio", "long_english_run"}


def test_build_translation_pairs_rejects_failed_alignment():
    rows = [
        {
            "relative_path": "guide/bad.md",
            "source_text": "A short source block.\n\nAnother short source block.",
            "target_text": "이 문단은 매우 길고 다른 내용을 포함합니다.\n\n추가 문단도 있어 문단 경계가 맞지 않도록 구성합니다.\n\n또 다른 문단까지 추가합니다.",
        }
    ]

    pairs, rejected = build_translation_pairs(
        rows,
        min_block_chars=3,
        min_chunk_chars=3,
        max_group_size=1,
        max_step_cost=0.05,
    )

    assert pairs == []
    assert rejected[0].reason == "alignment_failed"


def test_detect_structural_mismatch_flags_rewritten_intro():
    aligned_chunks = [
        (
            "Chat models are conversational models you can send a message to and receive a response. Most language models from mid-2023 onwards are chat models.",
            "이 글을 보고 있다면 채팅 모델에 대해 어느 정도 알고 계실 것입니다. 채팅 모델이란 메세지를 주고받을 수 있는 대화형 인공지능입니다. 대표적으로 ChatGPT가 있고, 이와 비슷하거나 더 뛰어난 오픈소스 채팅 모델이 많이 존재합니다. 이러한 모델들은 무료 다운로드할 수 있으며, 로컬에서 실행할 수 있습니다. 이 가이드는 채팅 모델을 처음 사용하는 분들에게 유용할 것입니다.",
        )
    ]
    assert detect_structural_mismatch(aligned_chunks) == "structural_mismatch"
