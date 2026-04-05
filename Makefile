SHELL := /bin/sh

UV := uv
PYTHON := $(UV) run python
PYTEST := $(UV) run pytest -q

REPO_URL ?= https://github.com/huggingface/transformers.git
REPO_DIR ?= .cache/huggingface-transformers
SOURCE_COMMIT ?= ce7efd8f19
DATASET_NAME ?= Transformers EN-KO Aligned Docs
DATASET_REPO_ID ?=
MODEL_NAME ?= Helsinki-NLP/opus-mt-tc-big-en-ko
MODEL_OUTPUT_DIR ?= outputs/opus-en-ko-transformers

.PHONY: help sync test corpus pairs rejected-report split-docs clean-suspects \
	review-clean review-contaminated review-suspects review final-train \
	train-validation package upload-private upload-public train all

help:
	@printf "%s\n" \
	"Available targets:" \
	"  make sync               Install standalone dependencies" \
	"  make test               Run translation harness tests" \
	"  make corpus             Build doc-level EN/KO corpus" \
	"  make pairs              Build aligned chunk pairs and rejected pairs" \
	"  make rejected-report    Build rejected-doc summaries" \
	"  make split-docs         Split clean vs contaminated docs" \
	"  make clean-suspects     Rank suspicious clean docs" \
	"  make review             Build all review markdown files" \
	"  make final-train        Apply blacklist to clean train pairs" \
	"  make train-validation   Split final clean data into train/validation" \
	"  make package            Build hf_dataset_repo/" \
	"  make upload-private DATASET_REPO_ID=user/name" \
	"  make upload-public DATASET_REPO_ID=user/name" \
	"  make train              Fine-tune the seq2seq model on clean train/validation" \
	"  make all                review + final splits + package"

sync:
	$(UV) sync --extra translation --extra dev

test:
	$(PYTEST)

corpus:
	$(PYTHON) scripts/build_translation_corpus.py \
		--repo-url "$(REPO_URL)" \
		--repo-dir "$(REPO_DIR)" \
		--output data/transformers_en_ko_docs.jsonl

pairs:
	$(PYTHON) scripts/prepare_translation_dataset.py \
		--input data/transformers_en_ko_docs.jsonl \
		--output data/transformers_en_ko_pairs.jsonl \
		--rejected-output data/transformers_en_ko_rejected_pairs.jsonl

rejected-report:
	$(PYTHON) scripts/report_rejected_pairs.py \
		--input data/transformers_en_ko_rejected_pairs.jsonl \
		--output-csv data/transformers_en_ko_rejected_docs.csv \
		--output-json data/transformers_en_ko_rejected_docs.json

split-docs:
	$(PYTHON) scripts/split_translation_dataset.py \
		--pairs-input data/transformers_en_ko_pairs.jsonl \
		--rejected-input data/transformers_en_ko_rejected_pairs.jsonl \
		--train-output data/transformers_en_ko_train_clean.jsonl \
		--eval-output data/transformers_en_ko_eval_contaminated.jsonl \
		--docs-output data/transformers_en_ko_doc_split.csv

clean-suspects:
	$(PYTHON) scripts/build_clean_suspects.py \
		--input data/transformers_en_ko_train_clean.jsonl \
		--output data/transformers_en_ko_clean_suspects.csv

review-clean:
	$(PYTHON) scripts/build_alignment_review.py \
		--input data/transformers_en_ko_train_clean.jsonl \
		--output data/review/train_clean_review.md \
		--title "Train Alignment Review"

review-contaminated:
	$(PYTHON) scripts/build_alignment_review.py \
		--input data/transformers_en_ko_eval_contaminated.jsonl \
		--output data/review/eval_contaminated_review.md \
		--title "Eval Alignment Review"

review-suspects:
	$(PYTHON) scripts/build_suspect_review.py \
		--pairs-input data/transformers_en_ko_train_clean.jsonl \
		--suspects-input data/transformers_en_ko_clean_suspects.csv \
		--output data/review/clean_suspect_review.md

review: split-docs clean-suspects review-clean review-contaminated review-suspects

final-train:
	$(PYTHON) scripts/filter_blacklist_pairs.py \
		--input data/transformers_en_ko_train_clean.jsonl \
		--blacklist data/blacklist.txt \
		--output data/transformers_en_ko_train_final.jsonl

train-validation:
	$(PYTHON) scripts/split_train_validation.py \
		--input data/transformers_en_ko_train_final.jsonl \
		--train-output data/transformers_en_ko_train_split.jsonl \
		--validation-output data/transformers_en_ko_validation_clean.jsonl \
		--validation-ratio 0.1 \
		--seed 42

package: final-train train-validation
	$(PYTHON) scripts/build_hf_dataset_repo.py \
		--train-input data/transformers_en_ko_train_split.jsonl \
		--validation-input data/transformers_en_ko_validation_clean.jsonl \
		--eval-input data/transformers_en_ko_eval_contaminated.jsonl \
		--blacklist-input data/blacklist.txt \
		--doc-split-input data/transformers_en_ko_doc_split.csv \
		--rejected-docs-input data/transformers_en_ko_rejected_docs.csv \
		--source-commit "$(SOURCE_COMMIT)" \
		--dataset-name "$(DATASET_NAME)" \
		--output-dir hf_dataset_repo

upload-private: package
	@test -n "$(DATASET_REPO_ID)" || (echo "DATASET_REPO_ID is required"; exit 1)
	$(PYTHON) scripts/upload_hf_dataset.py \
		--repo-id "$(DATASET_REPO_ID)" \
		--folder hf_dataset_repo \
		--private

upload-public: package
	@test -n "$(DATASET_REPO_ID)" || (echo "DATASET_REPO_ID is required"; exit 1)
	$(PYTHON) scripts/upload_hf_dataset.py \
		--repo-id "$(DATASET_REPO_ID)" \
		--folder hf_dataset_repo

train:
	$(PYTHON) scripts/train_translation_model.py \
		--train-file data/transformers_en_ko_train_split.jsonl \
		--validation-file data/transformers_en_ko_validation_clean.jsonl \
		--model-name-or-path "$(MODEL_NAME)" \
		--output-dir "$(MODEL_OUTPUT_DIR)"

all: review final-train train-validation package
