# Kinetix

Unified inference monorepo for the NeoZedAtlas runtime stack.

Current layout:

- Preserve the full git history of the existing `Axionyx`, `SwiftOCR`, and `Zinfer` repositories under `legacy/`.
- Build new shared runtime layers under `engine/`.
- Move modality-specific execution code into `adapters/`.
- Keep local model artifacts under `models/`:
  `models/text`, `models/ocr`, `models/vision`
- Keep local datasets and evaluation assets under `datasets/`:
  `datasets/vision`
- Converge toward one scheduler and one reusable inference core for vision, OCR, text, video, TTS, and future AI runtimes.
