# Legacy Extraction Map

This document maps imported repositories to the new shared Kinetix engine surface.

## Axionyx

- Candidate source for `engine/core/memory`: reuse allocator and tensor lifecycle reuse
- Candidate source for `engine/artifacts/graph`: graph schema and execution-plan loading
- Candidate source for `adapters/vision`: YOLO detection runtime and preprocessing

## SwiftOCR

- Candidate source for `adapters/ocr`: OCR pipeline shape and task-level entrypoints
- Candidate source for `engine/core/tensor`: shape-generic tensor helpers
- Candidate source for `engine/artifacts/graph`: lightweight DAG execution model

## Zinfer

- Candidate source for `engine/scheduler`: runtime pool, request scheduling, batching
- Candidate source for `engine/registry`: model-family and adapter registration
- Candidate source for `engine/artifacts/backend`: model artifact loading and backend selection
- Candidate source for `adapters/text`: decoder, embedding, and BERT service runtimes
