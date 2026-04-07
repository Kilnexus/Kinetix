# Monorepo Foundation

This repository is the integration root for the Kinetix unified inference framework.

## Phase 1

- Establish clean top-level ownership for shared runtime code.
- Remove the retired `legacy/` worktree after migration is complete.
- Move local model and dataset assets into shared top-level homes.

## Current Layout

- `sdk/`: package surface and execution/session entrypoints
- `engine/`: shared runtime core, memory, tensor, scheduler, artifact loading
- `engine/runtime/<modality>/`: internal runtime code grouped by concern instead of flat files
- `adapters/`: modality-specific runtimes built on top of `engine/`
- `apps/`: CLI and service entrypoints
- `docs/`: migration and architecture notes
- `models/`: local model artifacts organized by modality (`text/`, `ocr/`, `vision/`)
- `datasets/`: local datasets and evaluation assets organized by modality

## Extraction Order

1. Scheduler and runtime pooling
2. Memory reuse and tensor lifecycle management
3. Artifact loading and backend abstraction
4. Model family and adapter registry
5. Shared operator surface
