# Transformers EN-KO Dataset Harness

다국어 기술 문서에서 보수적인 영한 번역 학습 데이터셋을 만들기 위한 Codex 중심 소스 코드 저장소입니다.

English guide: [README.md](README.md)

현재 기본 대상 코퍼스는 `huggingface/transformers/docs/source/en` 과 `docs/source/ko` 이지만, 이 하네스는 더 일반적인 원칙을 전제로 설계되어 있습니다.

- 같은 상대 경로라고 번역쌍이라고 가정하지 않음
- 정렬은 보수적으로 수행함
- 재작성되거나 현지화된 문서는 억지로 맞추지 않고 제외함

전체 운영 원칙은 [docs/translation_dataset_harness.md](docs/translation_dataset_harness.md) 를 보면 됩니다.

배포된 데이터셋 산출물은 Hugging Face에서 볼 수 있습니다.

- [jmj-minju/transformers-en-ko-aligned-docs](https://huggingface.co/datasets/jmj-minju/transformers-en-ko-aligned-docs)

## 저장소 구조

- `docs/`: 하네스 문서와 방법론
- `.codex/`: runtime hook과 guardrail
- `.agents/skills/`: repo 전용 build/review workflow
- `scripts/`: 데이터셋 생성, 검토, 패키징, 업로드 스크립트
- `tests/`: 번역 하네스 테스트
- `data/`: 중간 산출물, 최종 산출물, 리뷰 산출물
- `hf_dataset_repo/`: Hugging Face 업로드용 패키지 폴더

## 빠른 시작

의존성 설치:

```bash
uv sync --extra translation --extra dev
```

또는 작업 타깃 확인:

```bash
make help
```

문서 단위 코퍼스 생성:

```bash
uv run python scripts/build_translation_corpus.py \
  --repo-url https://github.com/huggingface/transformers.git \
  --repo-dir .cache/huggingface-transformers \
  --output data/transformers_en_ko_docs.jsonl
```

정렬된 chunk pair 생성:

```bash
uv run python scripts/prepare_translation_dataset.py \
  --input data/transformers_en_ko_docs.jsonl \
  --output data/transformers_en_ko_pairs.jsonl
```

clean / contaminated 문서 분리:

```bash
uv run python scripts/split_translation_dataset.py \
  --pairs-input data/transformers_en_ko_pairs.jsonl \
  --rejected-input data/transformers_en_ko_rejected_pairs.jsonl \
  --train-output data/transformers_en_ko_train_clean.jsonl \
  --eval-output data/transformers_en_ko_eval_contaminated.jsonl \
  --docs-output data/transformers_en_ko_doc_split.csv
```

업로드용 패키지 생성:

```bash
uv run python scripts/build_hf_dataset_repo.py \
  --train-input data/transformers_en_ko_train_split.jsonl \
  --validation-input data/transformers_en_ko_validation_clean.jsonl \
  --eval-input data/transformers_en_ko_eval_contaminated.jsonl \
  --blacklist-input data/blacklist.txt \
  --doc-split-input data/transformers_en_ko_doc_split.csv \
  --rejected-docs-input data/transformers_en_ko_rejected_docs.csv \
  --source-commit ce7efd8f19 \
  --dataset-name "Transformers EN-KO Aligned Docs" \
  --output-dir hf_dataset_repo
```

같은 흐름을 Makefile로도 실행할 수 있습니다.

```bash
make review
make final-train
make train-validation
make package
```

데이터셋 패키지 업로드:

```bash
uv run python scripts/upload_hf_dataset.py \
  --repo-id jmj-minju/transformers-en-ko-aligned-docs \
  --folder hf_dataset_repo \
  --private
```

또는:

```bash
make upload-private DATASET_REPO_ID=jmj-minju/transformers-en-ko-aligned-docs
```

## 파인튜닝

clean split으로 학습하고 clean validation으로 모델 선택을 합니다.

```bash
uv run python scripts/train_translation_model.py \
  --train-file data/transformers_en_ko_train_split.jsonl \
  --validation-file data/transformers_en_ko_validation_clean.jsonl \
  --model-name-or-path Helsinki-NLP/opus-mt-tc-big-en-ko \
  --output-dir outputs/opus-en-ko-transformers
```

`data/transformers_en_ko_eval_contaminated.jsonl` 은 정식 validation이 아니라 stress-test 용도로만 사용하는 것이 맞습니다.
