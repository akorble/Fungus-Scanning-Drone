

# - fine-tunes a yolov8 model on a yolo-format dataset
# - runs inference on drone images
# - classes: Tree, Fungus
# - associates fungus detections withdetected trees
# - computes fraction of trees infected, infections per tree, avg infections among infected trees
# - outputs per-image and aggregated csv reports

# note: comments intentionally lowercase per user preference.


from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import os
import math
import csv
import json
from statistics import mean
import numpy as np

# ultralytics yolov8 (pip install ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
    ) from e


# ---------------------------
# simple classes for objects
# ---------------------------

@dataclass
class Fungus:
    xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coordinates)
    score: float
    class_id: int = 1  # default 1 => fungus
    centroid: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.xyxy
        self.centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Tree:
  
    xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    class_id: int = 0  # default 0 => tree
    centroid: Tuple[float, float] = field(init=False)
    fungus_hits: List[Fungus] = field(default_factory=list)

    def __post_init__(self):
        x1, y1, x2, y2 = self.xyxy
        self.centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def add_fungus(self, fungus: Fungus):
        self.fungus_hits.append(fungus)

    @property
    def infection_count(self) -> int:
        return len(self.fungus_hits)

    @property
    def infected(self) -> bool:
        return self.infection_count > 0



def bbox_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """compute intersection over union for two xyxy boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def centroid_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def map_fungi_to_trees(trees: List[Tree], fungi: List[Fungus], iou_threshold: float = 0.05) -> None:
    """


      - for each fungus, find tree with highest iou > threshold. if none, assign to nearest tree by centroid.
      - if no trees present (trees list empty), nothing assigned (fallback handled upstream)
    """
    if not trees:
        return

    for f in fungi:
        best_tree = None
        best_iou = 0.0
        for t in trees:
            iou = bbox_iou(f.xyxy, t.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_tree = t

        if best_tree is not None and best_iou >= iou_threshold:
            best_tree.add_fungus(f)
            continue

        # fallback to nearest centroid match
        nearest = min(trees, key=lambda tr: centroid_distance(tr.centroid, f.centroid))
        nearest.add_fungus(f)


# ---------------------------
# yolov8 training + inference
# ---------------------------

def train_yolo_model(
    data_yaml_path: str,
    epochs: int = 50,
    model_type: str = "yolov8n.pt",  # small base model; change to yolov8s/m/l/x as needed
    imgsz: int = 640,
    save_dir: str = "runs/train/fungus_run",
    batch: int = 16,
    lr: float = 0.01,
):
    """
    fine-tune a yolov8 model on provided dataset yaml.
    data_yaml_path should be a yaml file with train/val paths and nc/classes definitions.
    example data yaml:
    train: /path/to/images/train
    val: /path/to/images/val
    nc: 2
    names: ['tree', 'fungus']
    """
    # model training via ultralytics
    print(f"starting training: model_type={model_type}, epochs={epochs}, data={data_yaml_path}")
    model = YOLO(model_type)
    model.train(data=data_yaml_path, epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr, project="runs/train", name=os.path.basename(save_dir))
    # after training, ultralytics will save best weights; return path to best.pt
    # ultralytics usually writes to runs/train/<name>/weights/best.pt
    candidate = os.path.join("runs", "train", os.path.basename(save_dir), "weights", "best.pt")
    if os.path.exists(candidate):
        print(f"training finished. best weights at: {candidate}")
        return candidate
    else:
        print("warning: best weights not found at expected path. returning default model_type.")
        return model_type


def load_model(weights_path: str):
    """load a yolov8 model for inference."""
    print(f"loading model from: {weights_path}")
    model = YOLO(weights_path)
    return model


def run_inference_on_folder(model, images_folder: str, conf: float = 0.25, iou: float = 0.45) -> Dict[str, Dict]:
    """
    run inference on all images in a folder.
    returns a dict keyed by image filename with raw prediction info (list of detections).
    expected classes: 0 -> tree, 1 -> fungus (by dataset)
    """
    img_paths = []
    for fname in sorted(os.listdir(images_folder)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img_paths.append(os.path.join(images_folder, fname))

    results = {}
    if not img_paths:
        print("no images found in folder:", images_folder)
        return results

    # ultralytics can process list of images
    print(f"running inference on {len(img_paths)} images with conf={conf}")
    preds = model.predict(source=img_paths, conf=conf, iou=iou, verbose=False)

    for res in preds:
        # each res corresponds to an image
        img_path = res.orig_img_path if hasattr(res, "orig_img_path") else res.path
        detections = []
        # res.boxes: xyxy tensor, conf, cls
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            results[os.path.basename(img_path)] = {"trees": [], "fungi": []}
            continue

        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()  # [x1,y1,x2,y2]
            score = float(b.conf[0].cpu().numpy())
            cls = int(b.cls[0].cpu().numpy())
            detections.append({"xyxy": tuple(xyxy), "score": score, "class": cls})

        trees = [Tree(xyxy=d["xyxy"], score=d["score"], class_id=d["class"]) for d in detections if d["class"] == 0]
        fungi = [Fungus(xyxy=d["xyxy"], score=d["score"], class_id=d["class"]) for d in detections if d["class"] == 1]
        results[os.path.basename(img_path)] = {"trees": trees, "fungi": fungi}

    return results


# ---------------------------
# analysis & reporting
# ---------------------------

def analyze_predictions(predictions: Dict[str, Dict], fallback_one_tree_per_image: bool = True) -> Dict:
    """
    for each image:
      - map fungi to trees
      - compute total trees, affected trees, infections per tree
    aggregate:
      - fraction of trees impacted = total_affected_trees / total_trees
      - avg infections among infected trees
    returns a dictionary with per-image and aggregated stats.
    """
    per_image = {}
    total_trees = 0
    total_affected_trees = 0
    infections_per_infected_tree = []

    for img_name, data in predictions.items():
        trees: List[Tree] = data["trees"]
        fungi: List[Fungus] = data["fungi"]

        # fallback: if no tree detections and fallback enabled, assume 1 tree per image
        if not trees and fallback_one_tree_per_image:
            # create a dummy tree spanning the image (0,0 to large), and assign everything to it
            # note: we don't know image size here; using large bbox so mapping puts fungi into it
            fallback_tree = Tree(xyxy=(0, 0, 10000, 10000), score=1.0, class_id=0)
            trees = [fallback_tree]

        # map fungi to trees
        map_fungi_to_trees(trees, fungi)

        # compute stats
        n_trees = len(trees)
        n_affected = sum(1 for t in trees if t.infected)
        infections_counts = [t.infection_count for t in trees if t.infected]

        per_image[img_name] = {
            "n_trees": n_trees,
            "n_affected": n_affected,
            "infections_counts": infections_counts,
            "mean_infections_per_infected_tree": mean(infections_counts) if infections_counts else 0.0,
        }

        total_trees += n_trees
        total_affected_trees += n_affected
        infections_per_infected_tree.extend(infections_counts)

    aggregated = {
        "total_images": len(predictions),
        "total_trees": total_trees,
        "total_affected_trees": total_affected_trees,
        "fraction_trees_impacted": (total_affected_trees / total_trees) if total_trees > 0 else 0.0,
        "avg_infections_among_infected_trees": mean(infections_per_infected_tree) if infections_per_infected_tree else 0.0,
    }

    return {"per_image": per_image, "aggregated": aggregated}


def save_report(report: Dict, out_csv: str = "fungus_report.csv"):
    """save per-image rows and aggregated summary to a csv for further analysis."""
    per_image = report["per_image"]
    aggregated = report["aggregated"]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "n_trees", "n_affected", "mean_infections_per_infected_tree", "infections_counts_json"])
        for img, stats in per_image.items():
            writer.writerow([
                img,
                stats["n_trees"],
                stats["n_affected"],
                stats["mean_infections_per_infected_tree"],
                json.dumps(stats["infections_counts"])
            ])

        # write aggregated summary
        writer.writerow([])
        writer.writerow(["aggregated_total_images", aggregated["total_images"]])
        writer.writerow(["aggregated_total_trees", aggregated["total_trees"]])
        writer.writerow(["aggregated_total_affected_trees", aggregated["total_affected_trees"]])
        writer.writerow(["fraction_trees_impacted", aggregated["fraction_trees_impacted"]])
        writer.writerow(["avg_infections_among_infected_trees", aggregated["avg_infections_among_infected_trees"]])

    print(f"report saved to {out_csv}")


# ---------------------------
# example main usage
# ---------------------------

def main_example():
    """
    quick example workflow:

    1) prepare dataset in yolov8 format and a data yaml file.
       example data yaml:
       train: /path/to/images/train
       val: /path/to/images/val
       nc: 2
       names: ['tree','fungus']

    2) train: call train_yolo_model(data_yaml_path, epochs=..., model_type='yolov8n.pt')
       or provide a pre-trained weights file path to load_model()

    3) inference: call run_inference_on_folder(model, images_folder)

    4) analyze: call analyze_predictions(predictions)

    5) save: call save_report(report)

    adjust thresholds as needed (confidence, iou).
    """

    # ---------- user params - edit these ----------
    data_yaml = "data/fungus_data.yaml"  # path to your dataset yaml
    images_to_scan = "drone_images"  # folder with drone images to run inference on
    use_train = False  # set true to train / fine-tune model. set false to load pretrained weights
    weights_path = "yolov8n.pt"  # or path to your custom weights
    output_csv = "fungus_report.csv"
    # -----------------------------------------------

    if use_train:
        # train and get weights path
        best_weights = train_yolo_model(data_yaml_path=data_yaml, epochs=30, model_type=weights_path, save_dir="fungus_run")
        model = load_model(best_weights)
    else:
        model = load_model(weights_path)

    # run inference
    preds = run_inference_on_folder(model, images_to_scan, conf=0.25, iou=0.45)

    # analyze
    report = analyze_predictions(preds, fallback_one_tree_per_image=True)

    # save
    save_report(report, out_csv=output_csv)

    # print summary
    print("aggregated results:")
    for k, v in report["aggregated"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main_example()
