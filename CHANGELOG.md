# Changelog

All notable changes to RustyMeshSplat. See
[`docs/ENGINEERING_NOTES.md`](docs/ENGINEERING_NOTES.md) for the reasoning and
current design state behind these.

## [0.1.3] — 2026-06-22

Robustness pass focused on COLMAP 4.0 support and difficult captures, plus the
substrate for comparing reconstruction methods. Consolidates the
`dreamy-goldberg` line onto the release branch.

### Fixed
- **COLMAP 4.0 option-namespace breakage.** COLMAP 4.0 (GLOMAP merge) moved
  shared feature/matching options out of `SiftExtraction.*`/`SiftMatching.*`
  into `FeatureExtraction.*`/`FeatureMatching.*`, so `--SiftExtraction.max_image_size`
  errored with "unrecognised option" and feature extraction never started.
  meshsplat now probes each subcommand's `--help` and emits the flag spelling the
  installed binary understands (COLMAP 3.x and 4.x both work), with a
  version-gated fallback for `max_image_size`. Verified against COLMAP 4.0.3.

### Added
- **CUDA-first COLMAP acquisition.** The Windows auto-download fetches the CUDA
  prebuilt when an NVIDIA GPU is detected (`--colmap-build auto|cuda|nocuda`).
- **`--sfm-mapper auto|global|incremental`** (default `auto`): prefers the GLOMAP
  global mapper when the COLMAP build exposes it; `global` fails loudly on builds
  without it rather than silently using incremental.
- **Sequential→exhaustive auto-escalation:** with `--capture auto`, a large set
  whose sequential pass registers under half the photos retries once with
  exhaustive matching to bridge disconnected views.
- **Method provenance.** `--tag`, plus a derived default tag, burned into
  `splat.<tag>.ply` / `mesh.<tag>.*` / `report.<tag>.json`, with `method` and
  `tag` fields in `report.json` — run several pose/splat methods into one output
  dir and compare them directly.
- **`docs/ENGINEERING_NOTES.md`**: conclusions, decisions, validation status, and
  the locked `--poses`/`--splat` backend design.

### Notes
- The orthogonal `--poses`/`--splat` model-backend design (AnySplat, LGTM) is
  locked but not yet implemented — see the engineering notes.

## [0.1.2] — earlier
- SH degree 3 default + `--sh-degree`.

## [0.1.1] — earlier
- Point-cloud meshing, watertight closing, pycolmap-free COLMAP-text export.

## [0.1.0] — earlier
- Initial: photos → gaussian splat + mesh on a universal GPU (wgpu), Windows
  release CI.
