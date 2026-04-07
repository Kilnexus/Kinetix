# Kinetix

Unified inference monorepo for the NeoZedAtlas runtime stack.

Current layout:

- Build new shared runtime layers under `engine/`.
- Move modality-specific execution code into `adapters/`.
- Keep public package and orchestration entrypoints under `sdk/`.
- Keep executable entrypoints under `apps/cli` and `apps/services`.
- Keep local model artifacts under `models/`:
  `models/text`, `models/ocr`, `models/vision`
- Keep local datasets and evaluation assets under `datasets/`:
  `datasets/vision`
- Converge toward one scheduler and one reusable inference core for vision, OCR, text, video, TTS, and future AI runtimes.

Current directory intent:

- `sdk/`: package surface and shared execution/session entrypoints
- `engine/`: reusable runtime internals grouped by domain (`core`, `artifacts`, `runtime`, `scheduler`)
- `engine/runtime/vision/`: layered by concern (`io`, `analysis`, `memory`, `modules`, `nn`)
- `adapters/`: modality adapters (`text`, `vision`, `ocr`)
- `apps/`: executable surfaces only
