# Text2Loc Visionary - initial scaffold

This scaffold prepares the ground for an enhanced, problem-driven Text2Loc visionary branch.
- It introduces a pluggable architecture with modules for NL understanding, vector search, and a configurable pipeline.
- All new components are designed to be enabled/disabled via a YAML config.

## How to use (high level)
1) Configure features in `config/default.yaml`.
2) Run `python main.py` to exercise the pipeline with stubbed components.
3) Replace stubs with real implementations as you iterate.

## Design notes
- Non-destructive: does not modify existing visionary code paths.
- Emphasizes modular interfaces and testability.
- Focus on clarity of data flow and configuration-driven behavior.
