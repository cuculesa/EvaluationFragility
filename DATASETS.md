# Dataset provenance

EvalFrag does not redistribute benchmark files in the repository. `evalfrag prepare-data` retrieves them from their public source repositories and records a local SHA-256 manifest.

## GSM8K

- Source repository: `openai/grade-school-math`
- Pinned revision: `b0bb162`
- File: `grade_school_math/data/test.jsonl`
- Expected structure: JSON Lines with `question` and `answer`; the final numeric answer follows `####`.
- Repository license: MIT.
- Note: the source repository was archived in April 2026. The downloader uses a pinned repository revision; the local SHA-256 manifest remains the run-time integrity authority.

## BIG-Bench Hard: date understanding

- Source repository: `suzgunmirac/BIG-Bench-Hard`
- Pinned revision: `9ee07bd`
- Files: `bbh/date_understanding.json` and `cot-prompts/date_understanding.txt`
- Repository license: MIT.
- The official task-specific CoT prompt is required for the `fewshot_tagged` condition. The prototype incorrectly labeled a zero-example instruction as “few-shot”; production code rejects that condition if the official prompt file is missing.
