import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def link_or_copy(src: Path, dst: Path) -> None:
    """Create a fast staging copy. Prefer hard links, fall back to copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def discover_image_dirs(input_root: Path):
    """
    Yield directories that contain images directly.
    We keep the structure relative to input_root.
    """
    for current_dir, _dirs, files in os.walk(input_root):
        image_files = []
        for name in files:
            ext = Path(name).suffix.lower()
            if ext in SUPPORTED_EXTS:
                image_files.append(Path(current_dir) / name)

        if image_files:
            yield Path(current_dir), image_files


def stage_images(source_files, stage_dir: Path):
    stage_dir.mkdir(parents=True, exist_ok=True)
    for src in source_files:
        dst = stage_dir / src.name
        link_or_copy(src, dst)


def write_temp_config(config_path: Path, images_path: Path, output_folder: Path, issues_folder: Path, issue_log_file: Path, args, img_format: str):
    config = {
        "model_path": str(args.model_path).replace("\\", "/"),
        "images_path": str(images_path).replace("\\", "/") + "/",
        "detection_conf_thresh": args.detection_conf_thresh,
        "gpu_avail": args.gpu_avail,
        "img_format": img_format,
        "img_width": 0,
        "img_height": 0,
        "blur_radius": args.blur_radius,
        "output_folder": str(output_folder).replace("\\", "/"),
        "issues_folder": str(issues_folder).replace("\\", "/"),
        "issue_log_file": str(issue_log_file).replace("\\", "/"),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description="Batch blur images under a root directory.")
    parser.add_argument("--input-root", default="input_root", help="Root directory to scan for image folders.")
    parser.add_argument("--output-root", default="blurred_root", help="Root directory for blurred outputs.")
    parser.add_argument("--issues-root", default="issues_root", help="Root directory for problematic images.")
    parser.add_argument("--issue-log", default="issues_root/issues_log.jsonl", help="JSONL log path for issues.")
    parser.add_argument("--model-path", default="model/best.pt", help="Path to YOLO model.")
    parser.add_argument("--detection-conf-thresh", type=float, default=0.1, help="YOLO confidence threshold.")
    parser.add_argument("--gpu-avail", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--blur-radius", type=int, default=31, help="Gaussian blur radius; odd values work best.")
    parser.add_argument(
        "--yolo-config-dir",
        default=".yolo",
        help="Writable Ultralytics config directory used to avoid Windows permission issues.",
    )
    parser.add_argument(
        "--keep-existing-output",
        action="store_true",
        help="Keep existing files inside output folders instead of clearing them first.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    input_root = (project_root / args.input_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    issues_root = (project_root / args.issues_root).resolve()
    issue_log_file = (project_root / args.issue_log).resolve()
    yolo_config_dir = (project_root / args.yolo_config_dir).resolve()
    blur_script = project_root / "blur_images.py"

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not blur_script.exists():
        raise FileNotFoundError(f"blur_images.py not found: {blur_script}")

    output_root.mkdir(parents=True, exist_ok=True)
    issues_root.mkdir(parents=True, exist_ok=True)
    issue_log_file.parent.mkdir(parents=True, exist_ok=True)
    yolo_config_dir.mkdir(parents=True, exist_ok=True)

    if issue_log_file.exists() and not args.keep_existing_output:
        issue_log_file.unlink()

    dirs = list(discover_image_dirs(input_root))
    if not dirs:
        print(f"No image folders found under {input_root}")
        return

    print(f"Found {len(dirs)} image folders under {input_root}")

    env = os.environ.copy()
    env["YOLO_CONFIG_DIR"] = str(yolo_config_dir)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    with tempfile.TemporaryDirectory(prefix="blur_batch_", dir=str(project_root)) as tmp_root:
        tmp_root_path = Path(tmp_root)

        for index, (source_dir, image_files) in enumerate(dirs, start=1):
            rel_dir = source_dir.relative_to(input_root)
            target_dir = output_root / rel_dir
            issues_dir = issues_root / rel_dir
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            issues_dir.parent.mkdir(parents=True, exist_ok=True)

            if target_dir.exists() and not args.keep_existing_output:
                shutil.rmtree(target_dir)
            if issues_dir.exists() and not args.keep_existing_output:
                shutil.rmtree(issues_dir)

            ext_groups = defaultdict(list)
            for file_path in image_files:
                ext_groups[file_path.suffix.lower()].append(file_path)

            # Skip folders with no supported image extensions after filtering.
            if not ext_groups:
                continue

            print(f"[{index}/{len(dirs)}] {source_dir} -> {target_dir}")

            # Process one extension group at a time to stay compatible with the existing single-folder script.
            for ext, files in sorted(ext_groups.items()):
                stage_dir = tmp_root_path / f"stage_{index}_{ext.lstrip('.')}"
                if stage_dir.exists():
                    shutil.rmtree(stage_dir)
                stage_images(files, stage_dir)

                temp_config = tmp_root_path / f"config_{index}_{ext.lstrip('.')}.yaml"
                write_temp_config(temp_config, stage_dir, target_dir, issues_dir, issue_log_file, args, ext)

                cmd = [
                    sys.executable,
                    str(blur_script),
                    "--config",
                    str(temp_config),
                ]
                result = subprocess.run(cmd, cwd=str(project_root), env=env)
                if result.returncode != 0:
                    print(f"  Failed for {source_dir} ({ext}) with exit code {result.returncode}")

    print(f"Batch processing finished. Outputs are under: {output_root}")


if __name__ == "__main__":
    main()
