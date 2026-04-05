# Translation Dataset Agent Harness for Codex

This document specifies how to automate the creation of fine-tuning data from multilingual documentation repositories, based on the work we did on `huggingface/transformers/docs/source/en` and `docs/source/ko`.

It is written as an operational harness document for future agents, not as a retrospective note.
It is designed first for Codex-based workflows across the app, CLI, web, and future App Server or SDK automation.

## Current State vs Next Revision

This document mixes two things on purpose, but they should be read separately:

- current implemented pipeline: the scripts and artifacts that already exist in this repository
- next harness revision: the design corrections we want future Codex runs to follow

Current implemented pipeline:

- conservative path matching and chunk alignment
- document-level contamination split
- manual blacklist layer
- final train, validation, and stress-test packaging

Next harness revision:

- preserve headings and code blocks as alignment-time structure
- classify rewritten documents before alignment
- add explicit confidence routing and stronger agent review
- treat the current build as a baseline, not the final architecture

When this document says "should", read it as next-revision guidance unless a script in `scripts/` already implements it.

## Goal

Build a conservative English-to-Korean translation dataset from documentation sources with:

- maximum automatic filtering
- minimal but targeted human review
- reproducible intermediate artifacts
- a clean separation between:
  - alignment-time preprocessing
  - training-time preprocessing
  - evaluation-time review

The key lesson is simple:

- Same file path does not imply parallel translation.
- Automated alignment alone is not enough.
- The harness must be designed to reject aggressively, score uncertainty, and route only uncertain cases to review.

## Codex-Specific Operating Model

Recent Codex guidance changes how this harness should be structured.

- Keep `AGENTS.md` short and high-signal. It should be an index into the repository, not the whole policy.
- Keep repository-local files as the system of record. If a keep/drop decision is not written into the repo, future Codex runs should treat it as non-existent.
- Prefer progressive disclosure. Put stable entry points in `AGENTS.md`, detailed policy in this document, and execution details in scripts and generated artifacts.
- Design for agent review, not only human review. Codex should pre-triage high-risk documents and chunks before a human sees them.
- Treat approvals and sandbox policy as part of the harness design. Interactive review runs and non-interactive rebuilds should not use the same autonomy settings.

For this repository, that means:

- root `AGENTS.md` points to this document and the canonical scripts
- `docs/translation_dataset_harness.md` is the durable operating spec
- `.codex/hooks.json` is the runtime guardrail layer
- `.agents/skills/` contains repo-scoped reusable workflows
- review markdown in `data/review/` is the primary review surface
- `data/blacklist.txt` is the durable human override layer
- packaging and upload happen only after harness gates pass

## Codex Execution Modes

The same harness should support three execution modes.

### 1. Interactive Codex run

Use this when rebuilding the dataset locally with review.

- sandbox: `workspace-write`
- approvals: `on-request`
- required outputs: all intermediate artifacts plus review markdown
- stopping condition: if uncertain cases remain, stop and wait for review instead of auto-publishing

### 2. Non-interactive Codex run

Use this in CI or scheduled rebuilds.

- run to completion with fixed inputs and thresholds
- never publish if warning thresholds trip
- emit machine-readable summaries and fail loudly on missing artifacts
- prefer a single reproducible entrypoint such as `codex exec` plus repository scripts

### 3. Codex App Server or SDK orchestration

Use this only when the pipeline needs durable threads, resumability, or UI integration.

- map one dataset build to one thread
- map each major stage to one or more turns
- persist review artifacts as outputs of the thread, not as transient terminal text
- use approval pauses for destructive or publishing actions

This matches the current Codex harness direction, where threads, turns, and typed items are the durable units of work and review.

## Codex Repository Layout

Codex works better when the repository has a small entry point and discoverable deeper docs.

Recommended layout:

- `AGENTS.md`: short map for future Codex runs
- `docs/translation_dataset_harness.md`: full dataset-building harness
- `.codex/`: runtime hooks and repo-level Codex automation settings
- `.agents/skills/`: repo-scoped skills for build and review workflows
- `scripts/`: executable pipeline stages
- `data/`: generated artifacts, blacklist, and split outputs
- `data/review/`: human- and agent-readable review surfaces
- `hf_dataset_repo/`: packaging target only after gates pass

Do not put all of the policy into `AGENTS.md`.
Do not rely on chat memory, Slack, or ad hoc shell history.

## Latest Harness Principles

These design choices are aligned with recent public guidance on harness and eval design:

- OpenAI: use eval-driven development, define clear objectives, collect datasets, define metrics, compare runs, and continuously evaluate.
  - Source: [Evaluation best practices](https://platform.openai.com/docs/guides/evaluation-best-practices)
- OpenAI: for agent systems, prefer reproducible datasets, trace grading, and workflow-level evaluation.
  - Source: [Agent evals](https://platform.openai.com/docs/guides/agent-evals)
- OpenAI: treat the repository as the system of record, prefer progressive disclosure over giant instructions, and push constraints into tooling.
  - Source: [Harness engineering](https://openai.com/index/harness-engineering/)
- LangSmith: separate offline evaluation from online evaluation, use human review, code rules, and LLM-as-judge together, and establish a feedback loop from failing cases back into datasets.
  - Sources: [LangSmith Evaluation](https://docs.langchain.com/langsmith/evaluation), [Evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts)

For this project, that translates into:

- keep the harness repository-local
- keep every intermediate file versioned
- score failures at the document level and chunk level
- use agent review to triage uncertainty
- reserve human review for the small residual set

Also note the current Codex product guidance:

- Codex docs now expose first-class guidance around `AGENTS.md`, `Skills`, `Subagents`, approvals, non-interactive mode, and App Server integration.
- The Codex App Server article describes the harness in terms of durable threads, turns, and typed items, which is useful when designing a long-running dataset build or review workflow.
- Codex security guidance treats sandbox mode and approval policy as first-class controls, not afterthoughts.

## Failure Modes We Actually Observed

These are not hypothetical. They happened in this project.

### 1. Rewritten docs

Some Korean docs were not direct translations of the English docs. They were localized or re-authored.

Example:

- English: `docs/source/en/conversations.md`
- Korean: `docs/source/ko/conversations.md`

Symptoms:

- title changed substantially
- first paragraph was rewritten
- section order diverged
- examples differed

Conclusion:

- These files are not parallel translation data.
- No chunking strategy can reliably rescue them.

### 2. Source-only metadata

Some English files included release metadata lines like:

- `This model was released on ... and added to Hugging Face Transformers on ...`

The Korean docs often omitted them.

If not removed before alignment, this shifted the first chunk and corrupted subsequent matching.

### 3. Header mismatch

English and Korean files often used different section heading styles or translated headings loosely.

If headings were treated as ordinary text, they could force false alignments.

### 4. Code-anchor loss

We removed code blocks early, and that may have hurt alignment in some documents.

In technical docs, code blocks are often alignment anchors rather than noise.

### 5. Paragraph boundary mismatch

English docs often used shorter paragraph segmentation.
Korean docs often merged several ideas into larger prose blocks.

If chunking assumes identical paragraph boundaries, alignment drifts.

### 6. Partial translation / untranslated English

Some Korean docs contained untranslated explanatory English in the middle of otherwise Korean text.

This is valid as a contamination signal, but not every English token is bad.

Code-like English must often be preserved.

## What Worked

The following strategy produced usable training data:

1. Match files by relative path.
2. Remove obvious non-content artifacts.
3. Split text into smaller alignment blocks.
4. Align conservatively with small group sizes.
5. Reject documents with structural mismatch.
6. Reject chunks with contamination or severe length drift.
7. Review suspicious clean docs.
8. Maintain a manual blacklist.
9. Produce:
   - clean train
   - clean validation
   - contaminated stress test

This yielded a small but conservative dataset.

## Critical Review of Current Preprocessing

This section is intentionally critical. The previous pipeline is usable, but not ideal.

### Did removing headings help or hurt?

Both.

It helped on rewritten files where headings were not semantically aligned.
It hurt on structured docs where headings were the best alignment anchors.

Conclusion:

- Removing headings globally at alignment time was too aggressive.
- Headings should not be dropped blindly.
- Better approach:
  - preserve headings as structure markers
  - do not necessarily include the raw heading text in training output
  - use them as alignment hints, not as mandatory lexical matches

Why this likely hurt us:

- a heading is often the cleanest section boundary in technical docs
- removing it can cause the first prose paragraph of one section to fuse with the previous section's tail
- when the target omits or rewrites introductory prose, the missing heading makes first-chunk drift harder to detect
- this likely contributed to cases where the source chunk looked too long or already off by one paragraph

Recommended alignment-time representation:

- `## Overview` -> `SECTION_HEADER`
- translated heading text stored as metadata
- section boundary retained even if actual heading string differs

### Did removing code blocks help or hurt?

Also both.

It helped reduce training noise.
It likely hurt alignment because code blocks and nearby instruction prose are often mirrored across languages.

Conclusion:

- Code blocks should not be removed before alignment.
- They can be normalized instead.

Why this likely hurt us:

- code fences often separate instruction prose from result prose
- removing them can glue together paragraphs that were only related because they surrounded an example
- if both languages share the same command or API snippet, the code block is an anchor that helps alignment rather than noise
- this likely made some technical how-to docs look structurally flatter than they really were

Recommended alignment-time representation:

- replace each code block with a placeholder such as `CODE_BLOCK`
- keep code-block count and relative position as alignment features
- optionally keep a short hash of block contents for rough matching

Then, at training export time:

- either remove code blocks
- or keep them if the target use case includes code-oriented technical translation

### Core design correction

Alignment preprocessing and training preprocessing must be different stages.

Bad pattern:

- remove headings and code blocks before alignment

Better pattern:

- alignment stage: preserve structure, normalize noisy tokens
- export stage: remove or simplify content depending on the training objective

This is the single biggest design correction if we want future Codex runs to be more automatic and less review-heavy.

## Harness Architecture

The harness should have the following stages.

### Stage 0. Repo sync

Inputs:

- upstream repo URL
- commit SHA or branch
- source locale path
- target locale path

Outputs:

- local checked-out repo
- recorded source commit SHA

Requirements:

- pin commit used for dataset build
- record it in every artifact

### Stage 1. Document inventory

Match files by relative path and build a manifest with:

- `relative_path`
- `source_path`
- `target_path`
- extension
- file size
- commit SHA

Outputs:

- `manifest.csv`

### Stage 2. Alignment-time preprocessing

At this stage, do not optimize for training text cleanliness. Optimize for alignment reliability.

Rules:

- remove front matter
- remove known one-sided metadata lines
- normalize links to text
- preserve section boundaries
- preserve code blocks as placeholders
- preserve admonition boundaries
- preserve lists as list items
- preserve line boundaries when they encode semantics

Suggested representation:

- headings -> section markers with optional text metadata
- code blocks -> placeholders
- html/doc-builder tags -> normalized placeholders or dropped if non-semantic

Outputs:

- document-level normalized source/target text
- section index
- block list with block type

### Stage 3. Document-type classification

Before alignment, classify whether a document is likely:

- `parallel`
- `localized`
- `rewritten`
- `unknown`

Heuristics:

- first 1-2 section similarity
- heading count similarity
- block count ratio
- code-block count pattern
- sentence count ratio in opening section
- presence of rewritten introductory prose

Routing:

- `parallel` -> continue
- `localized` or `rewritten` -> reject document
- `unknown` -> send to agent review

This stage should catch files like `conversations.md`.

Outputs:

- `document_classification.jsonl`

### Stage 4. Block extraction

Blocks should be finer-grained than “blank line paragraph”.

Recommended block hierarchy:

- section
- prose paragraph
- single sentence if paragraph is very long
- list item
- code placeholder
- admonition

Goal:

- give the aligner small units
- avoid one large English paragraph being matched to a merged Korean block

### Stage 5. Candidate alignment

Use dynamic programming over small grouped blocks.

Allowed groupings:

- 1:1
- 1:2
- 2:1
- 2:2
- optionally 1:3 and 3:1 only if confidence remains high

Features for alignment score:

- length ratio
- block type compatibility
- code placeholder alignment
- list structure alignment
- section boundary consistency
- sentence count ratio
- anchor-token overlap for code-like identifiers

Outputs:

- aligned block pairs with per-pair score

### Stage 6. Chunk-level reject rules

Reject chunks when:

- source/target length ratio exceeds threshold
- target contains non-code explanatory English
- source and target sentence counts diverge sharply
- code or list structure mismatches badly

Important:

- code-like English should be protected
- explanatory English should be penalized

Outputs:

- `accepted_pairs.jsonl`
- `rejected_pairs.jsonl`

Each rejected row should include:

- `relative_path`
- `chunk_id`
- `reason`
- `github_url`
- `text`
- `target`
- `source_chars`
- `target_chars`
- `confidence`

### Stage 7. Document-level reject rules

Reject entire documents when:

- structural mismatch is detected
- alignment fails globally
- any chunk reject indicates clear document drift
- opening sections are rewritten

This is intentionally conservative.

Outputs:

- `doc_split.csv`
  - `clean`
  - `contaminated`

### Stage 8. Agent review

Use an agent review pass before human review.

The agent should receive:

- source snippet
- target snippet
- relative path
- local file link
- GitHub link
- reason scores
- confidence scores

The agent should output one of:

- `keep`
- `drop`
- `needs_human_review`

And one rationale:

- `parallel_translation`
- `localized_rewrite`
- `partial_translation`
- `misaligned_chunk`
- `acceptable_paraphrase`

Agent review should focus on:

- first chunk of each document
- highest-risk clean documents
- documents with large ratio variance
- documents with mixed keep/drop evidence
- documents where heading or code-block structure diverges unexpectedly

This reduces manual review volume significantly.

Recommended Codex reviewer split:

- subagent 1: document classifier
- subagent 2: chunk auditor
- subagent 3: packaging checker

Do not use one giant reviewer prompt for all of these jobs.

### Stage 9. Human review

Human review should only touch:

- `needs_human_review`
- top suspect clean docs
- any document the agent labels as inconsistent across chunks

Human review outputs:

- `blacklist.txt`
- optional `allowlist.txt`

Human review should be the exception path, not the default path.

### Stage 10. Final packaging

Produce three final sets:

- `train_split.jsonl`
- `validation_clean.jsonl`
- `stress_test_contaminated.jsonl`

Do not use contaminated data as the main validation set.

## Review Artifacts Required

The harness must always generate these artifacts:

- `rejected_pairs.jsonl`
- `rejected_docs.csv`
- `doc_split.csv`
- `clean_suspects.csv`
- `review/train_clean_review.md`
- `review/eval_contaminated_review.md`
- `review/clean_suspect_review.md`
- `blacklist.txt`
- final `train/validation/stress-test` files

If any artifact is missing, the run is incomplete.

## Codex Automation Contract

For a future Codex automation, the run should be considered successful only if all of the following are true:

- the upstream commit SHA is pinned and recorded
- every expected artifact exists
- warning thresholds do not fire
- any `needs_human_review` cases are either resolved or explicitly deferred
- final `train`, `validation`, and `stress-test` sets are regenerated from the same run
- publish steps are gated behind an explicit approval or an explicit non-interactive publish flag

Suggested machine-readable status file:

```json
{
  "source_commit": "string",
  "matched_docs": 0,
  "clean_docs": 0,
  "contaminated_docs": 0,
  "accepted_pairs": 0,
  "rejected_pairs": 0,
  "needs_human_review_docs": 0,
  "blacklist_docs": 0,
  "status": "passed | review_required | failed"
}
```

## Metrics and Thresholds

The harness should track:

- total matched docs
- docs rejected before alignment
- docs rejected after alignment
- clean doc count
- contaminated doc count
- accepted pair count
- rejected pair count
- manual blacklist count
- validation row count
- train row count

Recommended warning thresholds:

- if clean docs < 20% of matched docs, raise warning
- if first-chunk structural mismatch rate > 25%, raise warning
- if accepted pairs fall sharply after a code change, require review
- if blacklist grows run-over-run, inspect upstream drift

## Agent-Evaluable Rubric

Use this rubric for agent review.

Prompt template:

1. Are source and target describing the same idea unit?
2. Is the target a direct translation/paraphrase of the source, not a rewritten replacement?
3. Are there missing sentences that shift the meaning materially?
4. Is explanatory English present in target outside code-like spans?
5. Should this chunk be kept for translation fine-tuning?

Expected output schema:

```json
{
  "decision": "keep | drop | needs_human_review",
  "reason": "parallel_translation | localized_rewrite | partial_translation | misaligned_chunk | acceptable_paraphrase",
  "confidence": 0.0,
  "notes": "short justification"
}
```

Preferred grading mode:

- pairwise or pass/fail over free-form scoring
- strongest available judge model
- human calibration over a seed set

If using Codex as reviewer, keep the review task narrow and structured:

- prefer JSON output
- prefer one decision per document or chunk
- avoid asking the reviewer to fix data and judge it in the same turn
- keep the reviewer read-only whenever possible

## How to Minimize Human Review

Human review can be minimized, but not fully eliminated.

The best improvements are:

### 1. Document rewrite detection before alignment

This likely saves more review time than any later-stage tweak.

### 2. Two-pass preprocessing

- alignment-time: preserve structure
- training-time: clean text aggressively

### 3. Agent review on uncertain cases only

Do not ask humans to inspect all clean docs.
Ask the agent first.

### 4. Confidence-based routing

Each document should get a confidence score.

Example routing:

- confidence >= 0.9 -> auto keep
- 0.6 <= confidence < 0.9 -> agent review
- confidence < 0.6 -> auto reject or human review depending on doc importance

### 5. Golden document set

Maintain a small reviewed set of:

- definitely parallel docs
- definitely rewritten docs
- borderline acceptable paraphrase docs

Use it to regression-test harness changes.

## Fine-Tuning Recommendation

For this project, use:

- `train_split.jsonl` for training
- `validation_clean.jsonl` for model selection
- `stress_test_contaminated.jsonl` for stress testing only

Do not use contaminated docs as the main validation signal.

Recommended model families:

- seq2seq translation models first
- LoRA/PEFT only if moving to a larger general LLM

This dataset is small and conservative, so precision matters more than scale.

## Proposed Next Revision of the Harness

If we rebuild this pipeline, the next version should:

1. classify documents before alignment
2. preserve headings and code placeholders during alignment
3. align on typed blocks, not plain paragraphs
4. score uncertainty explicitly
5. use agent review before human review
6. export training text only after alignment is finalized
7. expose a Codex-friendly status artifact and publish gate
8. keep `AGENTS.md` short and use it as the index into the harness
9. support both interactive local review and non-interactive CI rebuilds

## Repository-Local Operating Principle

All harness knowledge should live in the repository.

Do not rely on:

- Slack decisions
- unstored human memory
- ad hoc shell history
- one-off manual judgments that are not captured in blacklist or docs

The harness is only reproducible if the next agent can:

- read this document
- read `AGENTS.md`
- inspect the scripts
- regenerate the artifacts
- understand why a document was kept or dropped

## Sources

- OpenAI, [Evaluation best practices](https://platform.openai.com/docs/guides/evaluation-best-practices)
- OpenAI, [Agent evals](https://platform.openai.com/docs/guides/agent-evals)
- OpenAI, [Harness engineering](https://openai.com/index/harness-engineering/)
- OpenAI, [Unlocking the Codex harness: how we built the App Server](https://openai.com/index/unlocking-the-codex-harness/)
- OpenAI, [Introducing Codex](https://openai.com/index/introducing-codex/)
- LangSmith, [Evaluation](https://docs.langchain.com/langsmith/evaluation)
- LangSmith, [Evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts)
