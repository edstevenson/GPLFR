# GPLFR public package notes

Future public repository for the GPLFR methods-paper reference implementation.

## Owns

- Flat public GPLFR package rooted at `gplfr/*.py`.
- Public surface `from gplfr import GPLFR, create_synthetic_data`.
- The quickstart notebook and runnable examples under `gplfr/demos/`.

## Guardrails

- Keep this repository lean and method-focused.
- Do not add private data, machine-local paths, generated experiment outputs, or credentials.
- Keep benchmark-specific data pipelines and large-scale orchestration out of the public package.
- Prefer small synthetic examples that run on a normal development machine.
