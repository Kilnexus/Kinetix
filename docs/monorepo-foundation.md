# Monorepo Foundation

This repository is the integration root for the Kinetix unified inference framework.

## Phase 1

- Import the full git history of `Axionyx`, `SwiftOCR`, and `Zinfer` into `legacy/`.
- Keep the original repositories untouched on disk during the migration.
- Establish clean top-level ownership for shared runtime code.

## Target Layout

- `legacy/`: history-preserved imported repositories
- `engine/`: shared runtime core, memory, tensor, scheduler, artifact loading
- `adapters/`: modality-specific runtimes built on top of `engine/`
- `apps/`: CLI and service entrypoints
- `docs/`: migration and architecture notes

## Extraction Order

1. Scheduler and runtime pooling
2. Memory reuse and tensor lifecycle management
3. Artifact loading and backend abstraction
4. Model family and adapter registry
5. Shared operator surface
