# Kinetix

Unified inference monorepo for the NeoZedAtlas runtime stack.

Current migration strategy:

- Preserve the full git history of the existing `Axionyx`, `SwiftOCR`, and `Zinfer` repositories under `legacy/`.
- Build new shared runtime layers under `engine/`.
- Move modality-specific execution code into `adapters/`.
- Converge toward one scheduler and one reusable inference core for vision, OCR, text, video, TTS, and future AI runtimes.
