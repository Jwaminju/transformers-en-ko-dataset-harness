from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face seq2seq model for translation.")
    parser.add_argument("--train-file", required=True, help="JSONL or CSV file with `text` and `target` columns.")
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--model-name-or-path", default="Helsinki-NLP/opus-mt-tc-big-en-ko")
    parser.add_argument("--output-dir", default="outputs/translation-model")
    parser.add_argument("--source-prefix", default="")
    parser.add_argument("--source-lang-code", default=None)
    parser.add_argument("--target-lang-code", default=None)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=768)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-strategy", default="epoch", choices=["steps", "epoch"])
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--predict-with-generate", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import evaluate
    import numpy as np
    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file

    file_extension = Path(args.train_file).suffix.lstrip(".")
    if file_extension not in {"json", "jsonl", "csv"}:
        raise ValueError("Only .jsonl, .json, and .csv are supported.")

    dataset_loader = "json" if file_extension in {"json", "jsonl"} else "csv"
    raw_datasets = load_dataset(dataset_loader, data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    if args.source_lang_code and hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = args.source_lang_code
    if args.target_lang_code:
        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = args.target_lang_code
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.target_lang_code)
            if forced_bos_token_id is not None and forced_bos_token_id >= 0:
                model.config.forced_bos_token_id = forced_bos_token_id

    prefix = args.source_prefix

    def preprocess_function(examples):
        inputs = [prefix + text for text in examples["text"]]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_predictions = [prediction.strip() for prediction in decoded_predictions]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        metrics = sacrebleu.compute(predictions=decoded_predictions, references=decoded_labels)
        prediction_lengths = [
            np.count_nonzero(prediction != tokenizer.pad_token_id) for prediction in predictions
        ]
        return {
            "bleu": round(metrics["score"], 4),
            "gen_len": round(float(np.mean(prediction_lengths)), 2),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=args.predict_with_generate,
        eval_strategy=args.eval_strategy if "validation" in tokenized_datasets else "no",
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        push_to_hub=args.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if "validation" in tokenized_datasets else None,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    if "validation" in tokenized_datasets:
        metrics = trainer.evaluate(max_length=args.max_target_length)
        print(metrics)


if __name__ == "__main__":
    main()
