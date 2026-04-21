import os
import glob
import json
import shutil
from pathlib import Path
import cv2
import pybboxes as pbx
import yaml
import argparse
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument("--config", help = "path of the training configuartion file", required = True)
args = parser.parse_args()

#Reading the configuration file
with open(args.config, 'r', encoding='utf-8-sig') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

issues_folder = config.get("issues_folder")
issue_log_file = config.get("issue_log_file")

if os.path.exists("runs"):
    shutil.rmtree("runs")

image_folder = config['images_path']
image_files = sorted(glob.glob(os.path.join(image_folder, "*" + config["img_format"])))


def append_issue_log(record):
    if not issue_log_file:
        return
    log_path = Path(issue_log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def copy_to_issues(src_path, issue_type, message, copied=True, exception_text=""):
    issue_path = ""
    if issues_folder:
        issue_dir = Path(issues_folder)
        issue_dir.mkdir(parents=True, exist_ok=True)
        dst = issue_dir / Path(src_path).name
        try:
            shutil.copy2(src_path, dst)
            issue_path = str(dst)
        except Exception as copy_exc:
            copied = False
            message = f"{message}; copy_failed={copy_exc}"
    append_issue_log(
        {
            "input_path": str(src_path),
            "issue_type": issue_type,
            "message": message,
            "copied_to_issue_folder": copied and bool(issue_path),
            "issue_path": issue_path,
            "exception": exception_text,
        }
    )


try:
    model = YOLO(config["model_path"])

    if config["gpu_avail"]:
        _ = model(
            source=config['images_path'],
            save=False,
            save_txt=True,
            conf=config['detection_conf_thresh'],
            device='cuda:0',
            project='runs/detect/',
            name="yolo_images_pred",
        )
    else:
        _ = model(
            source=config['images_path'],
            save=False,
            save_txt=True,
            conf=config['detection_conf_thresh'],
            device='cpu',
            project="runs/detect/",
            name="yolo_images_pred",
        )
except Exception as e:
    print(f"YOLO inference failed: {e}")
    for image_path in image_files:
        copy_to_issues(image_path, "inference_failed", "YOLO model initialization or inference failed", exception_text=str(e))
    raise SystemExit(0)

annot_dir = f'runs/detect/yolo_images_pred/labels/'

image_folder = config['images_path']
output_folder = config['output_folder']

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def classify_and_blur(image_path):
    image_name = os.path.basename(image_path)
    label_path = os.path.join(annot_dir, Path(image_name).stem + ".txt")

    if not os.path.exists(image_path):
        copy_to_issues(image_path, "read_failed", "input image path does not exist", copied=False)
        return

    image = cv2.imread(image_path)
    if image is None:
        copy_to_issues(image_path, "read_failed", "cv2.imread returned None")
        return

    if (not os.path.exists(label_path)) or os.path.getsize(label_path) == 0:
        copy_to_issues(image_path, "no_detection", "no YOLO detections found")
        return

    try:
        bboxes = []
        with open(label_path, "r", encoding="utf-8") as fin:
            for line in fin.readlines():
                values = line.strip().split()
                if len(values) < 5:
                    raise ValueError(f"Malformed label line: {line.strip()}")
                coords = [float(item) for item in values[1:]]
                img_height, img_width = image.shape[:2]
                bbox = pbx.convert_bbox(coords, from_type="yolo", to_type="voc", image_size=(img_width, img_height))
                bboxes.append([int(v) for v in bbox])
    except Exception as e:
        copy_to_issues(image_path, "parse_failed", "failed while parsing YOLO label file", exception_text=str(e))
        return

    try:
        image = blur_regions(image, bboxes)
    except Exception as e:
        copy_to_issues(image_path, "blur_failed", "failed while blurring detected regions", exception_text=str(e))
        return

    output_file = Path(image_name).stem + '_blurred.jpg'
    output_path = os.path.join(output_folder, output_file)
    try:
        ok = cv2.imwrite(output_path, image)
        if not ok:
            raise IOError(f"cv2.imwrite returned False for {output_path}")
    except Exception as e:
        copy_to_issues(image_path, "write_failed", "failed while writing blurred output", exception_text=str(e))
        return
def blur_regions(image, regions):
    """
    Blurs the image, given the x1,y1,x2,y2 cordinates using Gaussian Blur.
    """
    for region in regions:
        x1,y1,x2,y2 = region
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        roi = image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (config['blur_radius'], config['blur_radius']), 0)
        image[y1:y2, x1:x2] = blurred_roi
    return image


for image_path in image_files:
    try:
        classify_and_blur(image_path)
    except Exception as e:
        copy_to_issues(image_path, "unexpected_error", "unexpected error during image processing", exception_text=str(e))

print(f"@@ The bluured images are saved in Directory -------> {config['output_folder']}")
