import os
import json
import argparse
from PIL import Image
from torch.linalg import vander
import ast

def parse_json(text):
    """Parse JSON from text, handling markdown code fences if present."""
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1])  # Remove first and last lines
    # Remove any whitespace
    text = text.strip()
    return text

def load_annotation(json_path):
    """Load per-image annotation JSON containing bboxes + labels."""
    with open(json_path, "r") as f:
        return json.load(f)

def ensure_category(categories, label, label_to_id, supercategory="none"):
    """Register category if new; return category_id."""
    if label not in label_to_id:
        new_id = len(label_to_id) + 1
        label_to_id[label] = new_id
        categories.append({
            "id": new_id,
            "name": label,
            "supercategory": supercategory
        })
    else:
        # If already exists, optionally update the supercategory if given
        if supercategory != "none":
            cat_id = label_to_id[label]
            for c in categories:
                if c["id"] == cat_id:
                    c["supercategory"] = supercategory
                    break

    return label_to_id[label]

def main(images_dir, output_path):

    image_entries = []
    annotation_entries = []
    categories = []
    label_to_id = {}

    annotation_id = 1
    image_id = 1

    for fname in sorted(os.listdir(images_dir)):
        if not (fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg")):
            continue

        image_path = os.path.join(images_dir, fname)

        # Matching JSON file (same name, different extension)
        basename = os.path.splitext(fname)[0]
        json_path = os.path.join(images_dir, basename + ".json")
        if not os.path.isfile(json_path):
            print(f"[WARN] No JSON for {fname}, skipping.")
            continue

        # Load image size
        with Image.open(image_path) as img:
            w, h = img.size

        # Register image entry
        image_entries.append({
            "id": image_id,
            "file_name": fname,
            "width": w,
            "height": h
        })

        # Load annotations for this image
        data = load_annotation(json_path)

        # Expected format for each object:
        # {
        #   "bbox": [x, y, width, height],
        #   "label": "car",
        #   "noun_phrase": "...",   # optional
        # }
        #
        objects = data.get("answer", [])
        if 'json' in objects[0]:
            objects = ast.literal_eval(parse_json(objects[0]))


        for obj in objects:
            bbox = obj["bbox_2d"]
            bbox[0] = bbox[0] / 1000 * w
            bbox[2] = bbox[2] / 1000 * w
            bbox[1] = bbox[1] / 1000 * h
            bbox[3] = bbox[3] / 1000 * h

            supercat = "none"
            noun_phrase = obj["label"]

            if 'person' in noun_phrase:
                label = 'person'
            elif ('car' in noun_phrase
                  or 'sedan' in noun_phrase
                  or 'van' in noun_phrase
                  or 'SUV' in noun_phrase
                  or 'vehicle' in noun_phrase
            ):
                supercat = 'vehicle'
                label = 'car'
            elif 'motorcycle' in noun_phrase:
                supercat = 'vehicle'
                label = 'motorcycle'
            elif 'truck' in noun_phrase:
                supercat = 'vehicle'
                label = 'truck'
            elif 'motorcycle' in noun_phrase:
                supercat = 'vehicle'
                label = 'motorcycle'
            elif 'weapon' in noun_phrase:
                label = 'weapon'
            elif 'flag' in noun_phrase:
                label = 'flag'
            elif 'solar' in noun_phrase:
                label = 'solar panel'
            elif 'building' in noun_phrase:
                label = 'building'
            elif 'tree' in noun_phrase:
                label = 'tree'
            else:
                label = 'misc.'

            category_id = ensure_category(categories, label, label_to_id, supercat)

            annotation_entries.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": [],
                "noun_phrase": noun_phrase
            })
            annotation_id += 1

        image_id += 1

    # Build final COCO-like structure
    coco = {
        "info": {
            "description": "Generated COCO-like dataset",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [],
        "images": image_entries,
        "categories": categories,
        "annotations": annotation_entries
    }

    # Save
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"[DONE] COCO dataset saved to: {output_path}")
    print(f"Images: {len(image_entries)}  Annotations: {len(annotation_entries)}  Categories: {len(categories)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Folder containing images + per-image JSON files")
    parser.add_argument("--output", required=True, help="Output COCO JSON file path")
    args = parser.parse_args()

    main(args.images_dir, args.output)
