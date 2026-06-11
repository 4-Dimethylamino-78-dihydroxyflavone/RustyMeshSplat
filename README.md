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
| Mesh extraction | Pure Rust (rayon) TSDF fusion + surface nets | CPU |

No CUDA toolchain, no Python, no ONNX. On Windows it just uses DX12/Vulkan
through your normal graphics driver.

## Usage

```text
# the whole pipeline, balanced quality
meshsplat ./my_photos -o ./out

# quick preview
meshsplat ./my_photos --quality fast

# turntable / rotated-specimen captures (ordered frames, one camera)
meshsplat ./insect_frames --capture sequential --single-camera

# only train the splat, skip meshing
meshsplat ./my_photos --splat-only

# mesh an existing splat (optionally with poses for better normals)
meshsplat ./out/splat.ply --mesh-only --sparse ./out/work/colmap/sparse/0

# what does my machine support?
meshsplat --doctor
```

Key flags: `--iters`, `--max-splats`, `--max-resolution` (training);
`--grid-res`, `--opacity-min`, `--smooth-iters` (meshing); `--cpu-sfm`,
`--colmap <PATH>` (pose estimation). See `meshsplat --help`.

### Inputs

- **Directory of photos** (`.jpg/.png/.tif/...`): full pipeline. EXIF focal
  lengths are used automatically by COLMAP when present.
- **Posed dataset** (COLMAP `sparse/` or nerfstudio `transforms.json`):
  pose estimation is skipped.
- **Splat `.ply`** with `--mesh-only`: only mesh extraction runs.

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
  object-centric data) are the state of the art. They are Python/PyTorch
  research code, so meshsplat doesn't bundle them; instead it **accepts their
  COLMAP-format exports directly**: run e.g. VGGT's `demo_colmap.py` on your
  photos, then point meshsplat at the resulting dataset directory (it detects
  `sparse/` — binary or text format — and skips its own SfM).
- **Hundreds of images**: `--sfm-mapper global` uses the GLOMAP global mapper
  (integrated in COLMAP 4+, 1–2 orders of magnitude faster at comparable
  accuracy); it falls back to incremental automatically on older COLMAP builds.
- **Glossy / low-texture surfaces**: more photos with smaller angular steps,
  diffuse lighting, and masks over specular hotspots help SIFT survive.
  (Detector-free learned matchers à la LoFTR/RoMa are on the roadmap via
  COLMAP 4's ONNX ALIKED+LightGlue support.)

### COLMAP acquisition

`meshsplat` looks for COLMAP on `$PATH`, at `--colmap`, or in `$MESHSPLAT_COLMAP`.
On Windows, if none is found it downloads the official prebuilt
(`colmap-x64-windows-nocuda.zip`) into the user cache automatically.
On Linux/macOS install it once: `sudo apt install colmap` / `brew install colmap`.

## Pipeline

1. **Camera poses** — COLMAP feature extraction → matching (exhaustive for
   small sets, sequential for ordered/turntable sets) → incremental mapping.
   GPU SIFT is attempted and falls back to CPU transparently.
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

- Learned matching front-end (ALIKED + LightGlue via COLMAP 4's ONNX path)
  for low-texture / wide-baseline captures
- Optional feed-forward pose backend (VGGT/π³-class) once portable
  (non-PyTorch) inference for those models exists
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
| `crates/msplat-mesh` | Splat PLY loader, TSDF fusion, surface nets, PLY/OBJ/GLB export |

## License

Apache-2.0. Builds on [Brush](https://github.com/ArthurBrussee/brush)
(Apache-2.0), [Burn](https://burn.dev) (Apache-2.0/MIT),
[wgpu](https://wgpu.rs) (Apache-2.0/MIT), and orchestrates
[COLMAP](https://colmap.github.io) (BSD-3).
