# Changelog

## 1.0.0 — 2026-07-30

Production hardening of the original evaluation-fragility prototype:

- replaced implicit synthetic execution with explicit live/synthetic modes;
- added strict configuration and result schemas;
- added stable item IDs, dataset checksums, pinned source revisions, and provenance artifacts;
- switched orchestration to retryable Inspect eval sets with checkpoints and bounded concurrency;
- retained sample-level parser outcomes for audit and paired inference;
- added multi-seed aggregation, Wilson intervals, paired item bootstrap, exact parser contrasts, and multiplicity correction;
- corrected the BBH few-shot condition to use the official task prompt;
- removed causal “reasoning loss” language from the dashboard;
- made the dashboard self-contained and hardened embedded JSON;
- added tests, CI, container packaging, a wheel build, security guidance, and an evaluation card.
- added strict experiment-grid completeness checks and terminal run-state tracking;
- added staged dataset acquisition and an allowlisted manifest to prevent silent substitution;
- added deployment identity fields, a publication gate, and whole-run artifact verification;
- added deeper secret redaction, safe run-name validation, CSP-protected dashboards, and typed-package metadata;
- retained the original prototype snapshot and a production-hardening audit for traceability.
