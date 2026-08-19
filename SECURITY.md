# Security and privacy

## Secrets

- Supply API credentials through provider-specific environment variables, a secret manager, or workload identity.
- Never add credentials to TOML, command history, source control, or model names.
- Resolved configuration artifacts redact common credential keys, authorization headers, URL user information, and URL query strings. Redaction is defense in depth, not a substitute for keeping credentials out of configuration.
- EvalFrag sets `log_model_api=False`; Inspect can still record error context. Review logs before sharing.

## Sensitive artifacts

Inspect logs contain prompts, completions, scores, and metadata. Store run directories in access-controlled storage. `records.jsonl` omits completion text by default and stores a SHA-256 hash instead, but it is still evaluation data and should follow the same retention policy.

`evalfrag verify-artifacts` rejects modified, missing, extra, or symbolic-link artifacts relative to the completed run manifest. Write release reports outside the immutable run directory.

## Dashboard safety

The dashboard is generated from a strict Pydantic schema, uses no third-party scripts or fonts, escapes script-termination and JavaScript line-separator sequences, and includes a restrictive content-security policy. Do not serve arbitrary legacy JSON without migration and validation.

## Dataset supply chain

Benchmark files are downloaded from pinned public revisions into a staging directory, structurally checked, and moved into place only after a complete manifest passes verification. Manifests accept only the files and source metadata allowlisted by this release. For stronger guarantees, mirror approved files internally and pin immutable object versions.

## Filesystem safety

Experiment names are restricted to safe slugs. Generated files are written atomically with restrictive default permissions. Artifact verification rejects path traversal and symbolic links.

## Reporting vulnerabilities

Do not open a public issue containing credentials, private model outputs, endpoint details, or unreleased evaluation results. Use your organization’s private security reporting channel.
