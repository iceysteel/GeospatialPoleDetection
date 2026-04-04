#!/usr/bin/env python3
"""
Fine-tune GroundingDINO-Base on VLM-labeled pole detection data.
Freezes the Swin-B backbone, trains detection head + text-visual fusion.

Usage:
  python src/finetune_gdino.py --epochs 20
  python src/finetune_gdino.py --epochs 20 --eval-after  # run eval after training
"""
import sys, os, json, time, argparse, math
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.utils.data
from PIL import Image
from gpu_utils import get_device, gpu_memory_report

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAINING_DIR = os.path.join(DATA_DIR, 'training')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'gdino_finetuned')
TEXT_PROMPT = "utility pole. power pole. telephone pole."


class PoleDetectionDataset(torch.utils.data.Dataset):
    """COCO-format dataset for GDino fine-tuning."""

    def __init__(self, annotations_path, processor, max_size=800):
        with open(annotations_path) as f:
            coco = json.load(f)

        self.processor = processor
        self.max_size = max_size
        self.images_dir = os.path.dirname(annotations_path)

        # Build image lookup
        self.images = []
        img_anns = {}
        for ann in coco['annotations']:
            img_anns.setdefault(ann['image_id'], []).append(ann)

        for img_info in coco['images']:
            img_path = os.path.join(self.images_dir, 'images', img_info['file_name'])
            if not os.path.exists(img_path):
                continue
            anns = img_anns.get(img_info['id'], [])
            self.images.append({
                'path': img_path,
                'width': img_info['width'],
                'height': img_info['height'],
                'annotations': anns,
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image = Image.open(img_info['path']).convert('RGB')
        orig_w, orig_h = image.size

        # Resize if needed
        scale = min(1.0, self.max_size / max(orig_w, orig_h))
        if scale < 1.0:
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        else:
            new_w, new_h = orig_w, orig_h

        # Build target boxes in [cx, cy, w, h] normalized format (what GDino expects)
        boxes = []
        for ann in img_info['annotations']:
            x, y, w, h = ann['bbox']  # COCO format [x, y, w, h]
            # Scale to resized image, then normalize
            cx = (x + w / 2) * scale / new_w
            cy = (y + h / 2) * scale / new_h
            nw = w * scale / new_w
            nh = h * scale / new_h
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0.001, min(1, nw))
            nh = max(0.001, min(1, nh))
            boxes.append([cx, cy, nw, nh])

        labels = [0] * len(boxes)  # all category 0 = "pole"

        # Process through GDino processor
        encoding = self.processor(
            images=image,
            text=TEXT_PROMPT,
            return_tensors='pt',
        )

        # Squeeze batch dim (DataLoader will re-add it)
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.squeeze(0)

        # Build targets
        if boxes:
            target_boxes = torch.tensor(boxes, dtype=torch.float32)
            target_labels = torch.tensor(labels, dtype=torch.long)
        else:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_labels = torch.zeros((0,), dtype=torch.long)

        encoding['labels'] = [{
            'boxes': target_boxes,
            'class_labels': target_labels,
        }]

        return encoding


def collate_fn(batch):
    """Custom collate that handles variable-size targets."""
    # Separate inputs and labels
    labels = [item.pop('labels')[0] for item in batch]

    # Stack inputs (processor outputs have same shape after processing)
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            # Pad to max size in batch
            max_shape = [max(v.shape[i] for v in vals) for i in range(len(vals[0].shape))]
            padded = []
            for v in vals:
                pad_sizes = []
                for i in range(len(v.shape) - 1, -1, -1):
                    pad_sizes.extend([0, max_shape[i] - v.shape[i]])
                padded.append(torch.nn.functional.pad(v, pad_sizes))
            collated[k] = torch.stack(padded)
        else:
            collated[k] = vals

    collated['labels'] = labels
    return collated


def freeze_backbone(model):
    """Freeze the Swin backbone, keep detection head + fusion trainable."""
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if 'backbone' in name or 'input_proj_vision' in name:
            param.requires_grad = False
            frozen += 1
        else:
            trainable += 1
    print(f"Frozen: {frozen} params (backbone), Trainable: {trainable} params (head + fusion)")
    return model


def train(args):
    device = get_device(0)
    print(f"Device: {device}")

    # Load model and processor
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print(f"Loading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device)
    model = freeze_backbone(model)
    model.train()
    gpu_memory_report()

    # Dataset
    ann_path = os.path.join(args.training_dir, 'annotations.json')
    dataset = PoleDetectionDataset(ann_path, processor, max_size=args.max_size)
    print(f"Dataset: {len(dataset)} images")

    # Split into train/val (90/10)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Cosine LR schedule with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_loss = float('inf')
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items() if k != 'labels'}
            labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch['labels']]
            inputs['labels'] = labels

            outputs = model(**inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(1, n_batches)

        # Validation
        model.eval()
        val_loss = 0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items() if k != 'labels'}
                labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch['labels']]
                inputs['labels'] = labels
                outputs = model(**inputs)
                val_loss += outputs.loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(1, n_val_batches)
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        print(f"  Epoch {epoch+1}/{args.epochs} | train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} lr={lr:.2e} | {elapsed:.0f}s",
              flush=True)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODEL_DIR, 'best')
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    -> Saved best model (val_loss={best_val_loss:.4f})")

    # Save final model
    final_path = os.path.join(MODEL_DIR, 'final')
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to {MODEL_DIR}/best and {MODEL_DIR}/final")

    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='IDEA-Research/grounding-dino-base')
    parser.add_argument('--training-dir', default='data/training')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max-size', type=int, default=800)
    parser.add_argument('--eval-after', action='store_true', help='Run eval after training')
    args = parser.parse_args()

    # Step 1: Build training set if not exists
    ann_path = os.path.join(args.training_dir, 'annotations.json')
    if not os.path.exists(ann_path):
        print("Building training set first...")
        from build_training_set import build_coco_dataset
        build_coco_dataset('data/auto_labels/training_data.json', args.training_dir)
        print()

    # Step 2: Train
    model, processor = train(args)

    # Step 3: Optional eval
    if args.eval_after:
        print(f"\n{'='*60}")
        print("Running evaluation with fine-tuned model...")
        print(f"{'='*60}")
        os.system(f'{sys.executable} src/eval_testarea.py --device cuda:0 '
                  f'--gdino-model {MODEL_DIR}/best')


if __name__ == '__main__':
    main()
