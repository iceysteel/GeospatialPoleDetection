#!/usr/bin/env python3
"""
Convert VLM-classified detections into a COCO-format training dataset
for fine-tuning GroundingDINO.

Groups detections by source image, creates:
- data/training/images/ (symlinks to originals)
- data/training/annotations.json (COCO format)

Positive labels = VLM "pole" detections (high confidence)
Hard negatives = images where VLM classified detections as non-pole
  (these get empty annotation lists, teaching model NOT to detect them)
"""
import os, json, argparse
from collections import defaultdict
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def build_coco_dataset(training_data_path, output_dir, include_negatives=True):
    with open(training_data_path) as f:
        data = json.load(f)

    poles = data['high_conf_poles'] + data['medium_conf_poles']
    negatives = data['hard_negatives'] if include_negatives else []

    print(f"Building training set from {len(poles)} poles + {len(negatives)} hard negatives")

    # Group detections by source image
    # Poles: group all pole bboxes per image
    # Negatives: track which images have ONLY negative detections
    image_poles = defaultdict(list)  # image_path -> [bbox, ...]
    image_negs = defaultdict(list)

    for det in poles:
        image_poles[det['image_path']].append(det)
    for det in negatives:
        image_negs[det['image_path']].append(det)

    # Images with poles = positive training examples
    # Images with ONLY negatives (no poles) = hard negative examples (empty annotations)
    positive_images = set(image_poles.keys())
    negative_only_images = set(image_negs.keys()) - positive_images

    print(f"Positive images (have poles): {len(positive_images)}")
    print(f"Hard negative images (no poles): {len(negative_only_images)}")

    # Build COCO dataset
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    coco = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 0, 'name': 'pole', 'supercategory': 'infrastructure'}],
    }

    img_id = 0
    ann_id = 0

    # Positive images
    for img_path in sorted(positive_images):
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path)
        w, h = img.size
        img.close()

        # Symlink image
        fname = f"img_{img_id:05d}.png"
        link_path = os.path.join(images_dir, fname)
        if not os.path.exists(link_path):
            os.symlink(os.path.abspath(img_path), link_path)

        coco['images'].append({
            'id': img_id,
            'file_name': fname,
            'width': w,
            'height': h,
            'source_path': img_path,
        })

        # Add pole annotations
        for det in image_poles[img_path]:
            x1, y1, x2, y2 = det['bbox']
            bw, bh = x2 - x1, y2 - y1
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 0,
                'bbox': [x1, y1, bw, bh],  # COCO format: [x, y, width, height]
                'area': bw * bh,
                'iscrowd': 0,
                'vlm_confidence': det.get('vlm_confidence', 0),
            })
            ann_id += 1

        img_id += 1

    # Hard negative images (empty annotations — teach model these aren't poles)
    if include_negatives:
        for img_path in sorted(negative_only_images):
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path)
            w, h = img.size
            img.close()

            fname = f"img_{img_id:05d}.png"
            link_path = os.path.join(images_dir, fname)
            if not os.path.exists(link_path):
                os.symlink(os.path.abspath(img_path), link_path)

            coco['images'].append({
                'id': img_id,
                'file_name': fname,
                'width': w,
                'height': h,
                'source_path': img_path,
                'is_negative': True,
            })
            # No annotations for this image — it's a hard negative
            img_id += 1

    # Save
    ann_path = os.path.join(output_dir, 'annotations.json')
    with open(ann_path, 'w') as f:
        json.dump(coco, f, indent=2)

    # Summary
    n_pos = sum(1 for im in coco['images'] if not im.get('is_negative'))
    n_neg = sum(1 for im in coco['images'] if im.get('is_negative'))
    print(f"\nCOCO dataset saved to {output_dir}/")
    print(f"  Images: {len(coco['images'])} ({n_pos} positive, {n_neg} negative)")
    print(f"  Annotations: {len(coco['annotations'])} pole bboxes")
    print(f"  Categories: {[c['name'] for c in coco['categories']]}")

    return coco


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', default='data/auto_labels/training_data.json')
    parser.add_argument('--output', default='data/training')
    parser.add_argument('--no-negatives', action='store_true')
    args = parser.parse_args()
    build_coco_dataset(args.training_data, args.output, not args.no_negatives)
