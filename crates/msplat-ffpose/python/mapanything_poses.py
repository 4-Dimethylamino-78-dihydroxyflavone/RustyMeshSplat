# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "mapanything{{EXTRAS}} @ git+https://github.com/facebookresearch/map-anything@c845b8f4f6cde0c20aecd87573656c3f69f5b2b0",
#     "torch>=2.6",
#     "torchvision",
#     "numpy",
#     "pillow",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'win32'" }]
# torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'win32'" }]
# ///
"""Feed-forward camera poses for meshsplat via MapAnything.

Runs MapAnything (or MASt3R through MapAnything's wrapper) on a directory of
photos and writes a COLMAP-format reconstruction:

    <output-dir>/images/                       processed (model-resolution) images
    <output-dir>/sparse/{cameras,images,points3D}.txt

The COLMAP model is written directly in text format (no pycolmap / open3d):
the model already predicts intrinsics, cam2world poses, pointmaps and colors,
which is everything the COLMAP files need. meshsplat and Brush both read text
COLMAP models.

Protocol: stdout carries one JSON object per line ("stage", "status",
"progress", "error", "done"); everything else, including library prints,
goes to stderr. meshsplat parses stdout to drive its progress UI.
"""

import argparse
import glob
import json
import os
import sys
import urllib.request
from pathlib import Path

# Library code (torch, prints inside mapanything) writes to stdout; reroute it
# so the JSON event channel stays clean.
_emit_stream = sys.stdout
sys.stdout = sys.stderr

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TOTAL_STAGES = 4
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "tif", "tiff", "bmp")
MAST3R_CKPT_URL = (
    "https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
# Cap on the seed point cloud written to points3D.txt (Brush only needs a
# reasonable number of points to initialise gaussians).
MAX_SEED_POINTS = 400_000


def emit(**kw):
    print(json.dumps(kw), file=_emit_stream, flush=True)


def stage(name, index):
    emit(type="stage", name=name, index=index, total=TOTAL_STAGES)


def status(message):
    emit(type="status", message=message)


def fail(message, hint=None):
    event = {"type": "error", "message": message}
    if hint:
        event["hint"] = hint
    emit(**event)
    sys.exit(2)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--backend", choices=["mapanything", "mast3r"], default="mapanything")
    ap.add_argument("--model-id", default="facebook/map-anything-apache")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--no-exif-intrinsics", action="store_true")
    ap.add_argument("--cache-dir", default=None, help="Checkpoint cache for the mast3r backend")
    return ap.parse_args()


def pick_device(args, torch):
    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if device == "cpu" and not args.allow_cpu:
        fail(
            "No CUDA or Metal GPU is available for feed-forward pose estimation",
            hint="re-run with --poses-allow-cpu (expect minutes per scene), "
            "or use the default --poses colmap",
        )
    return device


def list_image_paths(images_dir):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(images_dir, f"*.{ext}")))
        paths.extend(glob.glob(os.path.join(images_dir, f"*.{ext.upper()}")))
    return sorted(set(paths))


def exif_intrinsics(path):
    """Pinhole K from the 35mm-equivalent focal length EXIF tag, or None."""
    import numpy as np
    from PIL import ExifTags, Image

    try:
        with Image.open(path) as img:
            width, height = img.size
            exif = img.getexif()
            ifd = exif.get_ifd(ExifTags.IFD.Exif) if exif else {}
    except Exception:
        return None
    f35 = ifd.get(ExifTags.Base.FocalLengthIn35mmFilm)
    if not f35 or float(f35) <= 0:
        return None
    # 35mm-equivalent focal length is relative to a 36mm-wide frame.
    fx = float(f35) / 36.0 * width
    return np.array(
        [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def build_mapanything_views(args, image_paths):
    """Views for MapAnything.infer, with EXIF-derived intrinsics when present."""
    intrinsics = []
    if not args.no_exif_intrinsics:
        intrinsics = [exif_intrinsics(p) for p in image_paths]
    if any(k is not None for k in intrinsics):
        from PIL import Image

        from mapanything.utils.image import preprocess_inputs

        n = sum(k is not None for k in intrinsics)
        status(f"Using EXIF focal lengths as intrinsics for {n}/{len(image_paths)} images")
        raw_views = []
        for path, k in zip(image_paths, intrinsics):
            with Image.open(path) as img:
                view = {"img": img.convert("RGB")}
            if k is not None:
                view["intrinsics"] = k
            raw_views.append(view)
        return preprocess_inputs(raw_views)
    from mapanything.utils.image import load_images

    return load_images(image_paths)


def run_mapanything(args, device, image_paths):
    import torch

    from mapanything.models import MapAnything

    status(f"Loading {args.model_id} (first run downloads ~2 GB from Hugging Face)")
    model = MapAnything.from_pretrained(args.model_id).to(device)
    model.eval()

    stage("Estimating poses", 3)
    views = build_mapanything_views(args, image_paths)
    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=device == "cuda",
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )
    return outputs


def ensure_mast3r_checkpoint(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cache_dir / Path(MAST3R_CKPT_URL).name
    if ckpt.is_file():
        return ckpt
    status("Downloading the MASt3R checkpoint (~2.6 GB, one-time)")
    tmp = ckpt.with_suffix(".pth.partial")
    with urllib.request.urlopen(MAST3R_CKPT_URL) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                emit(type="progress", done=done >> 20, total=total >> 20)
    os.replace(tmp, ckpt)
    return ckpt


def run_mast3r(args, device, image_paths, cache_dir):
    import torch

    from mapanything.models import model_factory
    from mapanything.utils.image import load_images
    from mapanything.utils.inference import postprocess_model_outputs_for_inference

    ckpt = ensure_mast3r_checkpoint(cache_dir)
    status("Loading MASt3R + sparse global alignment (CC BY-NC-SA, non-commercial)")
    model = model_factory(
        "mast3r",
        name="mast3r",
        ckpt_path=str(ckpt),
        cache_dir=str(cache_dir / "mast3r_cache"),
    ).to(device)
    model.eval()

    stage("Estimating poses", 3)
    views = load_images(image_paths, norm_type="dust3r", resolution_set=512)
    for i, view in enumerate(views):
        view["img"] = view["img"].to(device)
        # The wrapper derives per-view names from label/instance lists, which
        # load_images does not provide in that shape.
        view["label"] = ["scene"]
        view["instance"] = [str(i)]
    with torch.no_grad():  # the wrapper re-enables grad for the alignment
        raw = model(views)
    outputs = postprocess_model_outputs_for_inference(
        raw_outputs=raw, input_views=views, apply_mask=False, mask_edges=False
    )
    # The MASt3R wrapper has no validity mask; synthesize one from depth.
    for pred in outputs:
        pred["mask"] = pred["depth_z"] > 0
    return outputs


def rotmat_to_qvec(R):
    """Rotation matrix -> COLMAP quaternion [qw, qx, qy, qz] (COLMAP's own
    eigenvalue method, numerically robust for any valid rotation)."""
    import numpy as np

    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0.0, 0.0, 0.0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0.0, 0.0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0.0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec = -qvec
    return qvec


def write_colmap_text(outputs, image_names, output_dir):
    """Write a COLMAP text model (cameras/images/points3D.txt) and the
    processed images directly from MapAnything predictions."""
    import numpy as np
    from PIL import Image

    sparse_dir = os.path.join(output_dir, "sparse")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    camera_lines, image_lines = [], []
    seed_xyz, seed_rgb = [], []
    per_view_cap = max(1, MAX_SEED_POINTS // max(1, len(outputs)))

    for i, (pred, name) in enumerate(zip(outputs, image_names)):
        cam_id = i + 1
        rgb = np.clip(pred["img_no_norm"][0].cpu().numpy(), 0.0, 1.0)  # (H, W, 3)
        rgb_u8 = (rgb * 255.0).astype(np.uint8)
        height, width = rgb_u8.shape[:2]
        Image.fromarray(rgb_u8).save(os.path.join(images_dir, name))

        K = pred["intrinsics"][0].cpu().numpy()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        camera_lines.append(
            f"{cam_id} PINHOLE {width} {height} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}"
        )

        # COLMAP stores world->camera; MapAnything predicts camera->world.
        cam2world = pred["camera_poses"][0].cpu().numpy()
        world2cam = np.linalg.inv(cam2world)
        qw, qx, qy, qz = rotmat_to_qvec(world2cam[:3, :3])
        tx, ty, tz = world2cam[:3, 3]
        image_lines.append(
            f"{cam_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{tx:.8f} {ty:.8f} {tz:.8f} {cam_id} {name}"
        )
        image_lines.append("")  # empty POINTS2D line

        # Collect a bounded, masked subset of coloured points for the seed cloud.
        pts = pred["pts3d"][0].cpu().numpy().reshape(-1, 3)
        cols = rgb_u8.reshape(-1, 3)
        if "mask" in pred:
            valid = pred["mask"][0].cpu().numpy().reshape(-1).astype(bool)
        else:
            valid = np.ones(len(pts), dtype=bool)
        valid &= np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 0)
        pts, cols = pts[valid], cols[valid]
        if len(pts) > per_view_cap:
            stride = len(pts) // per_view_cap + 1
            pts, cols = pts[::stride], cols[::stride]
        seed_xyz.append(pts)
        seed_rgb.append(cols)

    xyz = np.concatenate(seed_xyz) if seed_xyz else np.zeros((0, 3))
    rgb = np.concatenate(seed_rgb) if seed_rgb else np.zeros((0, 3), np.uint8)
    point_lines = [
        f"{j + 1} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} 0"
        for j, ((x, y, z), (r, g, b)) in enumerate(zip(xyz, rgb))
    ]

    _write_block(
        os.path.join(sparse_dir, "cameras.txt"),
        "# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        camera_lines,
    )
    _write_block(
        os.path.join(sparse_dir, "images.txt"),
        "# Image list: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   (second line per image is POINTS2D, left empty)",
        image_lines,
    )
    _write_block(
        os.path.join(sparse_dir, "points3D.txt"),
        "# 3D points: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
        point_lines,
    )
    return len(image_names), len(xyz)


def _write_block(path, header, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def main():
    args = parse_args()

    stage("Loading PyTorch", 1)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        fail(
            f"PyTorch is not importable: {exc}",
            hint="when not using uv, point MESHSPLAT_FFPOSE_PYTHON at a Python "
            "environment with mapanything installed (pip install "
            "'mapanything @ git+https://github.com/facebookresearch/map-anything')",
        )
    device = pick_device(args, torch)
    status(f"Compute device: {device}")

    image_paths = list_image_paths(args.images_dir)
    if len(image_paths) < 2:
        fail(f"Need at least 2 images, found {len(image_paths)} in {args.images_dir}")
    image_names = [os.path.basename(p) for p in image_paths]

    stage("Loading model weights", 2)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path.home() / ".cache" / "meshsplat" / "ffpose"
    if args.backend == "mast3r":
        outputs = run_mast3r(args, device, image_paths, cache_dir)
    else:
        outputs = run_mapanything(args, device, image_paths)

    stage("Exporting COLMAP model", 4)
    os.makedirs(args.output_dir, exist_ok=True)
    views, points = write_colmap_text(outputs, image_names, args.output_dir)
    emit(type="done", views=views, points=points)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # surface the cause on both channels
        import traceback

        traceback.print_exc()
        fail(f"{type(exc).__name__}: {exc}")
