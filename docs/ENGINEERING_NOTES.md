# RustyMeshSplat — Engineering Notes & Decisions

A durable record of the investigation, conclusions, and locked design behind the
`v0.1.3` line. Written so the work can be picked up locally (or by a future
session) without re-deriving anything. Companion to the user-facing `README.md`;
this file is the *why* and the *where we are*.

**Snapshot — 2026-06-22, v0.1.3**

| Area | State |
|---|---|
| SfM registration collapse (the original problem) | **Fixed** — mapper + matching ladder; verified logic, pending full local run |
| COLMAP 4.0 option-namespace breakage | **Fixed** — `--help`-probed flags; verified against the 4060 Ti's real COLMAP 4.0.3 |
| CUDA-first COLMAP acquisition | **Shipped** — auto-detects NVIDIA |
| Method provenance (compare backends in one dir) | **Shipped** |
| `--poses` / `--splat` orthogonal backend design | **Locked**, not yet implemented (Stage B/C below) |
| AnySplat / LGTM model backends | **Designed, not built** — next work |

Everything except the model backends is built, clippy-clean, and unit-tested in
CI. What CI **cannot** cover — and what therefore needs first-run validation on
the RTX 4060 Ti — is called out explicitly in [§7](#7-validation-status--how-to-continue-locally).

---

## 1. The original problem: COLMAP registration collapse

The trigger for this whole line of work: on object-orbit / multi-session photo
sets, COLMAP registered only a handful of images and the reconstruction was
near-empty. Root causes and the levers added for each:

- **Wrong mapper for large/unordered sets.** GLOMAP's global mapper (merged into
  COLMAP 4.0) is far more robust here than incremental. `--sfm-mapper auto` (now
  the default) prefers `global_mapper` whenever the COLMAP build exposes it, and
  falls back to incremental on older builds.
- **Sequential matching on non-sequential data.** A large set captured as
  multiple sessions collapses under sequential matching. With `--capture auto`,
  if a sequential pass registers **under half** the photos, meshsplat now
  **auto-escalates once to exhaustive matching** to bridge the disconnected view
  graph (`pipeline.rs`, the attempt loop). `--capture unordered` forces
  exhaustive from the start — the right call for an object orbit.
- **Silent downgrades hid the problem.** `--sfm-mapper global` now **fails loudly**
  if the build has no `global_mapper`, instead of quietly using incremental and
  pretending the user got what they asked for.

These are in `crates/msplat-colmap/src/pipeline.rs::run_sfm`.

---

## 2. COLMAP 4.0 (GLOMAP merge) — the option-namespace migration

**The bug, as it surfaced on the 4060 Ti (COLMAP 4.0.3, CUDA build):**

```
E... base_option_manager.cc:264] Failed to parse options -
     unrecognised option '--SiftExtraction.max_image_size'.
```

Feature extraction died at the argument parser before doing any work — both the
GPU attempt and the CPU fallback, because the offending flag is shared by both.

**Root cause.** COLMAP 4.0.0 (GLOMAP merge, 2026-03) reorganized its option
manager. *Shared* feature/matching options moved out of the SIFT-specific
namespaces into general ones:

- `SiftExtraction.*` → `FeatureExtraction.*` for shared options
  (`max_image_size`, `use_gpu`, …)
- `SiftMatching.*` → `FeatureMatching.*` likewise
- Genuinely SIFT-only options (e.g. `max_num_features`) **stay** under
  `SiftExtraction.*`

Corroboration: the COLMAP changelog entry *"Share feature extraction
max_image_size option"* (3.13.0), the 4.1 docs showing `--FeatureExtraction.type
SIFT` alongside `--SiftExtraction.max_num_features`, and — decisively — probing
the actual binary on the target machine:

```
> colmap feature_extractor --help | Select-String "image_size|use_gpu"
  --FeatureExtraction.use_gpu arg (=1)
  --FeatureExtraction.max_image_size arg (=-1)   # note: new default -1 = no downscale
```

`ImageReader.*` (`camera_model`, `single_camera`, `mask_path`) is **unchanged** —
the parser flagged `max_image_size`, not the `ImageReader` args that precede it
in our command line, which means those namespaces still resolve in 4.0.3.

**The fix — don't hardcode either spelling; ask the binary.**
`ColmapBinary::subcommand_help()` captures `colmap <sub> --help`, and
`pipeline.rs::flag_for(help, leaf, fallback)` derives the fully-qualified
`--Namespace.leaf` flag this build actually exposes. One binary — COLMAP 3.x or
4.x — gets options it recognises, with no version branching in the call sites.
Applied to `max_image_size`, `use_gpu` (extractor + matcher), and the sequential
`overlap` flag.

Belt-and-suspenders: `max_image_size` additionally has a **version-gated
fallback** (`major_version() >= 4` → `FeatureExtraction`) for the one rename the
changelog explicitly confirms, in case a build doesn't print parseable help. The
probe is primary; the gate is insurance. Unit tests in `pipeline.rs` lock the
parser against representative 3.x and 4.x help text.

Confirmed: `colmap feature_extractor --help` **does** print the full option list,
so the probe mechanism is sound on this binary.

---

## 3. CUDA-first COLMAP acquisition

On Windows, when no COLMAP is found, meshsplat auto-downloads the official
prebuilt. `--colmap-build auto` (default) now fetches the **CUDA** build when an
NVIDIA adapter is present — SIFT extraction + matching are the dominant cost on
large sets and the CUDA build is dramatically faster — and the no-CUDA build
otherwise. Force with `--colmap-build cuda|nocuda`; `MESHSPLAT_COLMAP_URL`
overrides the URL entirely. No effect when `--colmap` points at an existing
install. Implementation: `crates/msplat-colmap/src/fetch.rs`.

---

## 4. Method provenance — compare backends in one output dir

Every run now carries a **method tag** so several pose/splat methods can be run
into the same `-o` directory and compared without clobbering each other:

- `--tag <name>` sets it; otherwise `default_tag()` derives one from the
  reconstruction method (e.g. `colmap`, `mapanything`, `mast3r`).
- Tag is sanitized to be filename-safe and burned into artifact names:
  `splat.<tag>.ply`, `mesh.<tag>.*`, `report.<tag>.json`.
- `report.json` gains two fields: `method` (human-readable
  `poses=… · train=brush(shN) · mesh=…` descriptor) and `tag`.

Code: `crates/meshsplat/src/main.rs` (`default_tag`, `sanitize_tag`, the run
header) and `crates/meshsplat/src/report.rs` (`method`/`tag` fields, tagged
`save`). This is the substrate the comparison matrix in §6 rides on.

---

## 5. The model-backend design (LOCKED)

Decided with Nic; orthogonal flags so pose source and splat/render producer vary
independently:

```
--poses   colmap | mapanything | mast3r | anysplat     # Stage 1: pose source
--splat   brush  | anysplat    | lgtm                  # Stage 2: producer (default brush)
--lgtm-backbone  noposplat | depthsplat                # pose-free vs posed LGTM
```

Each artifact self-tags (§4), so one `-o` dir yields the full comparison:

| Command | Tag | Produces |
|---|---|---|
| `--poses colmap` | `colmap` | splat + mesh (baseline) |
| `--poses mapanything` | `mapanything` | splat + mesh |
| `--poses anysplat --splat brush` | `anysplat-poses` | AnySplat **poses** → Brush splat + mesh |
| `--splat anysplat` | `anysplat` | AnySplat's **native** 3DGS `.ply` → mesh |
| `--splat lgtm --lgtm-backbone noposplat` | `lgtm-noposplat` | 4K renders + PSNR/SSIM (pose-free) |
| `--splat lgtm --lgtm-backbone depthsplat` | `lgtm-depthsplat` | 4K renders + metrics (uses `--poses` poses) |

**Principle: native output + tag, mesh where possible.**

- **AnySplat** emits a real 3DGS `.ply` → flows straight into the existing TSDF
  mesh stage, so `--splat anysplat` gives a comparable splat *and* mesh. Its
  poses are separately usable via `--poses anysplat`. Both are real outputs of
  the one forward pass.
- **LGTM** stays at native **4K renders + PSNR/SSIM** with a tagged checkpoint
  and metrics in `report.json`; **no mesh bake** (its output isn't a splat we can
  fuse the same way). Both backbones (`noposplat` pose-free, `depthsplat` posed)
  are exposed.
- **CUDA**: auto-detect NVIDIA → use CUDA for these backends, matching the
  COLMAP acquisition policy.

These backends follow the established pattern (`crates/msplat-ffpose`): a
version-pinned **PEP 723** Python script run in a **uv**-provisioned subprocess,
keeping the meshsplat binary pure Rust. Nothing here is implemented yet — see §8.

---

## 6. Why the model backends aren't validated here

The model integrations (AnySplat, LGTM) can be **written** in this environment
but **not run**: the container has no GPU, no CUDA, and no model weights. The
same is true of the CUDA COLMAP path. So those land as "validate on first run"
against the 4060 Ti, where gsplat/LGTM also compile their CUDA kernels.

What **is** fully exercised in CI: the Rust build, clippy, unit tests (including
the `flag_for` namespace-probe tests), and all the SfM control logic.

---

## 7. Validation status & how to continue locally

**Confirmed against the real target (COLMAP 4.0.3 on the 4060 Ti):** the option
namespaces the fix resolves to (`--FeatureExtraction.max_image_size`,
`--FeatureExtraction.use_gpu`) match the binary's own `--help`. Pending: a full
end-to-end run to confirm `registered_images` climbs.

**To continue locally:**

1. **Get the binary.** Either download `meshsplat-windows-x64.zip` from the
   `v0.1.3` GitHub Release, or `cargo build --release -p meshsplat`.
2. **Validate the registration fix** (the original problem) on the burst set:
   ```powershell
   $colmap = (gci "D:\...\colmap-x64-windows-cuda" -r -Filter COLMAP.bat | select -First 1).FullName
   & $exe $sub -o "$bh\rustymeshsplat\out\burst" --colmap $colmap `
       --single-camera --capture unordered --quality high --splat-only
   ```
   Expect: feature extraction now passes `--FeatureExtraction.*` flags and runs
   the CUDA SIFT path instead of dying at the parser; exhaustive matching;
   `registered_images` climbs toward the full set.
3. **Then** build the backends: Stage B (AnySplat), Stage C (LGTM) — §8.

**Release / build mechanism.** `.github/workflows/release.yml` builds
Windows/Linux/macOS `meshsplat` binaries. On a `v*` tag push it publishes a
GitHub Release with the zipped artifacts; via `workflow_dispatch` it builds any
branch and uploads artifacts (no public release) — useful for testing a branch
build before tagging. `v0.1.3` is cut from the release branch
`claude/rust-gpu-photogrammetry-e2mpez`.

---

## 8. Roadmap

**Stage B — AnySplat** (`--splat anysplat`, `--poses anysplat`)
- PEP 723 script: AnySplat forward pass → native 3DGS `.ply` + COLMAP-format
  poses; Rust plumbing mirrors `msplat-ffpose`.
- `.ply` → existing TSDF mesh stage; poses → existing dataset path.
- First real *comparable asset* from a feed-forward model.

**Stage C — LGTM** (`--splat lgtm`, `--lgtm-backbone {noposplat|depthsplat}`)
- 4K renders + PSNR/SSIM; tagged checkpoint; metrics into `report.json`; no mesh.

**Longer-term (from README):** learned matching front-end (ALIKED + LightGlue via
COLMAP 4's ONNX path), full-res training after feed-forward poses, COLMAP-rescue
hybrid, Poisson `--mesh poisson`, depth-TSDF (RaDe-GS), mesh-in-the-loop (MILo /
MeshSplatting) on wgpu, glTF `KHR_gaussian_splatting` export.

---

## Appendix — source map

| Concern | Location |
|---|---|
| Option-namespace probe (COLMAP 3.x/4.x) | `crates/msplat-colmap/src/pipeline.rs::flag_for`; `locate.rs::{subcommand_help, major_version}` |
| SfM ladder (mapper select, auto-escalation, fail-loud) | `crates/msplat-colmap/src/pipeline.rs::run_sfm` |
| GPU→CPU SIFT fallback | `pipeline.rs::run_with_gpu_fallback` |
| CUDA-first COLMAP download | `crates/msplat-colmap/src/fetch.rs` |
| Method provenance | `crates/meshsplat/src/main.rs` (`default_tag`, `sanitize_tag`); `report.rs` |
| Feed-forward backend pattern (template for B/C) | `crates/msplat-ffpose/` |
| Release/build CI | `.github/workflows/release.yml` |
