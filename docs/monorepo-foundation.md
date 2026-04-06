# Monorepo Foundation

This repository is the integration root for the Kinetix unified inference framework.

## Phase 1

- Import the full git history of `Axionyx`, `SwiftOCR`, and `Zinfer` into `legacy/`.
- Establish clean top-level ownership for shared runtime code.
- Move local model and dataset assets into shared top-level homes outside the legacy history trees.

## Target Layout

- `legacy/`: history-preserved imported repositories
- `engine/`: shared runtime core, memory, tensor, scheduler, artifact loading
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
