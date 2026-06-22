# RustyMeshSplat

**Photos in → gaussian splat + mesh out. Any GPU. No CUDA. One executable.**

`meshsplat` is a standalone Rust CLI that takes a directory of ordinary,
uncalibrated photos (a drone sweep of a region, a turntable scan of a mounted
insect, a walk-around of a statue) and produces:

- a trained **3D gaussian splat** (`splat.ply`) — viewable in any splat viewer
- a **triangle mesh** with vertex colors (`mesh.ply`, `mesh.obj`, `mesh.glb`)
  — usable in Blender, Unity, Unreal, or any glTF pipeline
- a machine-readable `report.json` with timings and statistics

with live progress bars and ETAs for every stage.

## Why this exists

Every advanced photogrammetry / radiance-field tool assumes CUDA, cuDNN, ONNX
runtimes, or a Python environment that breaks differently on every Windows
machine. This project's bedrock is the **universal GPU layer instead**:

| Layer | Tech | Runs on |
|---|---|---|
| Training & compute | [Burn](https://burn.dev) + [wgpu](https://wgpu.rs) (via [Brush](https://github.com/ArthurBrussee/brush)) | Vulkan, DirectX 12, Metal, OpenGL — AMD, Intel, NVIDIA, Apple |
| Camera poses | [COLMAP](https://colmap.github.io) binary (BSD), auto-downloaded on Windows | CPU (GPU optional, never required) |
| Camera poses, *optional* (`--poses mapanything`) | [MapAnything](https://github.com/facebookresearch/map-anything) in a [uv](https://docs.astral.sh/uv/)-provisioned Python subprocess | CUDA / Apple-silicon GPU (CPU possible, slow) |
| Mesh extraction | Pure Rust (rayon) TSDF fusion + surface nets | CPU |

No CUDA toolchain, no Python, no ONNX. On Windows it just uses DX12/Vulkan
through your normal graphics driver.

## Fail-proof GPU bring-up (Windows first-class)

`meshsplat` treats "the GPU just works" as a hard requirement, not a hope.
At startup it walks a **selection ladder** instead of trusting one backend:

1. Every backend is probed in reliability order for your OS —
   Windows: **Vulkan → DirectX 12 → OpenGL**, macOS: Metal,
   Linux: Vulkan → OpenGL.
2. Within each backend, real GPUs are ranked discrete → integrated → virtual;
   software rasterizers (WARP, llvmpipe) are excluded unless you opt in.
3. Each candidate is actually brought up (device + queue created with the
   exact feature set the trainer needs). **Any failure falls through to the
   next candidate** — a broken Vulkan driver never takes the run down when
   DX12 works, which is the single most common Windows failure mode.
4. Only if the whole ladder is exhausted do you get one clean, actionable
   error — and it happens **in milliseconds, before pose estimation runs**,
   never after twenty minutes of COLMAP.

You stay in control when you want to be:

| Flag | Effect |
|---|---|
| `--gpu-backend dx12` | Pin a backend (`vulkan`, `dx12`, `metal`, `gl`) |
| `--gpu-index N` | Pin a specific adapter from the `--doctor` list |
| `--allow-software` | Last-resort software rasterizer (WARP/llvmpipe): glacial, but turns "no GPU" into "still got a model" |
| `--doctor` | Show every adapter on every backend, in selection order |

Works on any Windows 10/11 machine with a working graphics driver — AMD,
Intel, NVIDIA, laptop iGPUs included. There is nothing to install: no CUDA
toolkit, no cuDNN, no ONNX runtime, no Python environment. If a game can
render on the machine, meshsplat can train on it. (Mesh extraction,
`--mesh-only`, is pure CPU and needs no GPU at all.)

Prebuilt binaries (`meshsplat-windows-x64.zip` and friends) are produced by
CI for every tagged release.

## Usage

```text
# the whole pipeline, balanced quality
meshsplat ./my_photos -o ./out

# quick preview
meshsplat ./my_photos --quality fast

# turntable / rotated-specimen captures (ordered frames, one camera)
meshsplat ./insect_frames --capture sequential --single-camera

# sparse / low-overlap captures where COLMAP fails: feed-forward poses
# (needs uv + a CUDA/Metal GPU; weights download on first run)
meshsplat ./sparse_photos --poses mapanything

# only train the splat, skip meshing
meshsplat ./my_photos --splat-only

# mesh an existing splat (optionally with poses for better normals)
meshsplat ./out/splat.ply --mesh-only --sparse ./out/work/colmap/sparse/0

# mesh an external point cloud (Metashape/COLMAP/any .ply or .obj) directly,
# closed into a fuller watertight-ish solid, with cameras to orient normals
meshsplat ./dense_cloud.ply --mesh-only --watertight --cameras ./cameras.txt

# what does my machine support?
meshsplat --doctor
```

Key flags: `--iters`, `--max-splats`, `--max-resolution` (training);
`--grid-res`, `--opacity-min`, `--smooth-iters`, `--watertight` (meshing);
`--cpu-sfm`, `--colmap <PATH>`, `--poses <BACKEND>` (pose estimation). See
`meshsplat --help`.

### Inputs

- **Directory of photos** (`.jpg/.png/.tif/...`): full pipeline. EXIF focal
  lengths are used automatically by COLMAP when present.
- **Posed dataset** (COLMAP `sparse/` or `sparse/0/`, binary or text, or
  nerfstudio `transforms.json`): pose estimation is skipped. This ingests
  the raw output of MapAnything's / VGGT's `demo_colmap.py` directly.
- **Splat `.ply`** with `--mesh-only`: only mesh extraction runs.
- **External point cloud** (`.ply` ASCII or binary, or `.obj`) with
  `--mesh-only`: a dense cloud from Metashape, COLMAP, or any tool is meshed
  directly — normals are estimated when the file lacks them, and `--cameras`
  orients them. Add `--watertight` to wrap it into a fuller closed solid.

### Difficult scenes (low overlap, turntables, glossy specimens)

Classical SfM has known failure modes; the pipeline gives you the right
levers for each (informed by the 2025–2026 pose-estimation literature):

- **Turntable / rotated-specimen captures** (insects, minerals): a rotating
  object in front of a static camera is *geometrically identical* to a camera
  orbiting a static object — if background features dominate, COLMAP poses the
  room instead of the object and the reconstruction fails. Fixes, in order of
  impact:
  1. **Mask the background**: `--masks <DIR>` (per-image PNGs named
     `<image>.<ext>.png`, black = ignored). Generate masks with any
     segmentation tool (e.g. SAM2, rembg).
  2. Put a **textured / fiducial-marked mat on the turntable** so it rotates
     with the specimen — disambiguates symmetric objects and recovers full
     360° instead of a hemisphere.
  3. Use `--capture sequential` (shares one camera model across frames and
     matches neighboring frames).
- **Low-overlap / sparse / wide-baseline sets**: COLMAP needs ~60–80% overlap
  to shine. When it registers too few images, feed-forward pose transformers
  (VGGT, π³, MapAnything — CO3Dv2 pose AUC ~88 vs ~25 for classical SfM on
  object-centric data) are the state of the art. **`--poses mapanything`
  runs one for you** (see [Feed-forward pose estimation](#feed-forward-pose-estimation---poses)
  below). meshsplat also still **accepts COLMAP-format exports directly**:
  run e.g. VGGT's `demo_colmap.py` yourself, then point meshsplat at the
  resulting dataset directory (it detects `sparse/` or `sparse/0/` — binary
  or text — and skips its own SfM).
- **Hundreds of images**: the default `--sfm-mapper auto` uses the GLOMAP
  global mapper whenever your COLMAP build exposes it (merged into COLMAP in
  4.0.0, 2026‑03; 1–2 orders of magnitude faster at comparable accuracy) and
  falls back to incremental on older builds. Force it with `--sfm-mapper
  global` (which errors on builds without the `global_mapper` command), or pin
  the classic path with `--sfm-mapper incremental`.
- **Many sessions / unordered piles**: prefer `--capture unordered` so every
  pair is matched. With the default `--capture auto`, a large set first tries
  sequential matching; if that registers under half the photos (the classic
  multi-session collapse), meshsplat automatically retries once with exhaustive
  matching to bridge the disconnected views.
- **Glossy / low-texture surfaces**: more photos with smaller angular steps,
  diffuse lighting, and masks over specular hotspots help SIFT survive.
  (Detector-free learned matchers à la LoFTR/RoMa are on the roadmap via
  COLMAP 4's ONNX ALIKED+LightGlue support.)

### COLMAP acquisition

`meshsplat` looks for COLMAP on `$PATH`, at `--colmap`, or in `$MESHSPLAT_COLMAP`.
On Windows, if none is found it downloads the official prebuilt into the user
cache automatically — `--colmap-build auto` (default) fetches the **CUDA** build
when an NVIDIA GPU is detected (dramatically faster SIFT extraction + matching
on large sets) and the no-CUDA build otherwise; force either with
`--colmap-build cuda|nocuda`, or point `MESHSPLAT_COLMAP_URL` at any zip.
On Linux/macOS install it once: `sudo apt install colmap` / `brew install colmap`.

**COLMAP 3.x and 4.x both work.** COLMAP 4.0 (the GLOMAP merge, 2026‑03)
relocated several option namespaces (e.g. `SiftExtraction.max_image_size` →
`FeatureExtraction.max_image_size`). meshsplat probes the installed binary's own
`--help` and emits the flag spelling it understands, so a single build adapts to
either version with no configuration.

### Comparing reconstruction methods (provenance tags)

Run several pose/splat methods into the **same** `-o` directory and compare them
side by side: every run carries a **method tag** that is burned into its artifact
names — `splat.<tag>.ply`, `mesh.<tag>.*`, `report.<tag>.json` — so nothing gets
clobbered. `report.json` also records `method` (a
`poses=… · train=brush(shN) · mesh=…` descriptor) and `tag`. The tag defaults
to the reconstruction method (`colmap`, `mapanything`, `mast3r`, …); override it
with `--tag <name>`. This is the basis for the orthogonal `--poses` / `--splat`
backend matrix described in
[`docs/ENGINEERING_NOTES.md`](docs/ENGINEERING_NOTES.md).

### Feed-forward pose estimation (`--poses`)

`--poses mapanything` replaces COLMAP's Stage 1 with
[MapAnything](https://github.com/facebookresearch/map-anything), Meta's
feed-forward metric 3D reconstruction transformer — the right tool when
COLMAP registers too few images (sparse captures, wide baselines,
object-centric orbits). The core promise stays intact: the meshsplat binary
remains pure Rust. The model runs in a subprocess from a bundled,
version-pinned [PEP 723](https://peps.python.org/pep-0723/) script whose
Python environment [uv](https://docs.astral.sh/uv/) provisions automatically
on first use; the script exports a COLMAP-format model that flows into the
same training/meshing stages.

Requirements and behavior:

- **uv** on `$PATH` (or `--uv <PATH>` / `$MESHSPLAT_UV`). No uv? Point
  `$MESHSPLAT_FFPOSE_PYTHON` at a Python that already has
  `mapanything[colmap]` installed.
- A **CUDA or Apple-silicon GPU** for sensible speed. `--poses-allow-cpu`
  permits CPU inference (minutes per scene). This GPU requirement applies
  *only* to this optional backend — the rest of meshsplat still needs no CUDA.
- **First run** resolves the Python environment and downloads model weights
  (~2–3 GB) into the Hugging Face / meshsplat caches; later runs start fast.
- **EXIF focal lengths** are passed to the model as per-view intrinsics
  (MapAnything's multi-modal inference accepts any subset of calibration
  inputs and measurably improves with them). `--poses-no-exif` disables this.
- The exported dataset uses the model-resolution processed images that the
  predicted intrinsics describe, so the splat trains at the model's working
  resolution (~0.5 MP). For maximum-fidelity captures with good overlap,
  classical COLMAP at full resolution may still win — try both.

Weights and licensing — pick deliberately:

| Flag | Model | License |
|---|---|---|
| *(default)* `--poses-model apache` | `facebook/map-anything-apache` | Apache-2.0 (commercial OK) |
| `--poses-model best` | `facebook/map-anything` | CC BY-NC 4.0 (non-commercial) |
| `--poses mast3r` | [MASt3R](https://github.com/naver/mast3r) + sparse global alignment, via MapAnything's wrapper | CC BY-NC-SA 4.0 (non-commercial) |

meshsplat prints a license notice whenever a non-commercial option is used.
`--poses-model` also accepts any Hugging Face model id compatible with
`MapAnything.from_pretrained`.

## Pipeline

1. **Camera poses** — COLMAP feature extraction → matching (exhaustive for
   small sets, sequential for large ordered sets, auto-escalating to exhaustive
   if a sequential pass registers under half the photos) → mapping (GLOMAP
   global mapper by default on COLMAP 4.0+, else incremental).
   GPU SIFT is attempted and falls back to CPU transparently. With
   `--poses mapanything|mast3r`, a feed-forward model produces the poses
   instead (uv-provisioned Python subprocess emitting the same COLMAP-format
   model).
2. **Splat training** — [Brush](https://github.com/ArthurBrussee/brush)'s
   gaussian splat trainer on Burn/wgpu, driven as a library with our progress
   UI. Exports a standard 3DGS `splat.ply`.
3. **Mesh extraction** — opacity-filtered gaussians are treated as oriented
   surface samples (normal = flattest covariance axis, oriented toward the
   nearest camera), fused into a truncated signed distance field, surfaced
   with naive surface nets (unknown space produces no phantom geometry),
   Laplacian-smoothed, vertex-colored from the splats, and exported as
   PLY / OBJ / GLB.

### Roadmap

- Orthogonal **`--poses` / `--splat` backends** (AnySplat, LGTM) so the pose
  source and the splat/render producer vary independently and tag their outputs
  for direct comparison — design locked in
  [`docs/ENGINEERING_NOTES.md`](docs/ENGINEERING_NOTES.md)
- Learned matching front-end (ALIKED + LightGlue via COLMAP 4's ONNX path)
  for low-texture / wide-baseline captures
- Full-resolution training after feed-forward poses: back-project
  MapAnything's intrinsics from the processed crops onto the original photos
- COLMAP-rescue hybrid: feed a partial COLMAP solve (poses + intrinsics for
  the registered subset) into MapAnything's multi-modal inference to complete
  the scene
- Poisson surface reconstruction as `--mesh poisson`: smoothly hallucinate
  unseen surface for fully complete 360° objects from partial scans (the
  next phase beyond `--watertight`, which seals holes/cavities but won't
  invent large never-observed regions)
- GPU depth-map rendering + depth TSDF fusion (RaDe-GS-style) for higher
  mesh fidelity
- [MILo](https://github.com/Anttwo/MILo) / [MeshSplatting](https://meshsplatting.github.io/)-style
  mesh-in-the-loop training, ported to wgpu compute
- glTF `KHR_gaussian_splatting` export once the Khronos extension ratifies
- masks / background removal for turntable captures

## Building

```bash
cargo build --release
# binary at target/release/meshsplat
```

Rust 1.85+ (edition 2024). The first build compiles the Burn/wgpu stack and
takes a while. `Cargo.lock` pins Brush's tested revisions of burn and the
patched wgpu fork — keep it checked in.

## Workspace layout

| Crate | Purpose |
|---|---|
| `crates/meshsplat` | CLI binary: stage orchestration, progress UI, GPU probe |
| `crates/msplat-colmap` | COLMAP locate/auto-download, SfM pipeline runner, sparse-model parsing |
| `crates/msplat-ffpose` | Feed-forward poses: uv runtime discovery, bundled MapAnything script, subprocess orchestration |
| `crates/msplat-mesh` | Splat/point-cloud (PLY/OBJ) loaders, normal estimation, TSDF fusion + optional watertight closing, surface nets, PLY/OBJ/GLB export |

## License

Apache-2.0. Builds on [Brush](https://github.com/ArthurBrussee/brush)
(Apache-2.0), [Burn](https://burn.dev) (Apache-2.0/MIT),
[wgpu](https://wgpu.rs) (Apache-2.0/MIT), and orchestrates
[COLMAP](https://colmap.github.io) (BSD-3).
