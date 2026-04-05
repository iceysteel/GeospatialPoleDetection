#!/usr/bin/env python3
"""
Fine-tune SAM3 on pole detection data.
Uses the SAM3 detection head with frozen backbone, training on our COCO dataset.

Approach: freeze the ViT backbone + text encoder, train detection head + neck.
This is similar to our GDino fine-tuning strategy.
"""
import sys, os, json, time, math, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.utils.data
from PIL import Image
from gpu_utils import get_device, gpu_memory_report

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'sam3_finetuned')

SAM3_CKPT = os.path.join(os.path.expanduser("~"),
    ".cache/huggingface/hub/models--bodhicitta--sam3/snapshots/cba430d22f6fdc3f06ad3841274ec7bb55885f2f/sam3.pt")


class PoleDatasetSAM3(torch.utils.data.Dataset):
    """COCO dataset adapter for SAM3 fine-tuning."""

    def __init__(self, annotation_file, image_root, max_size=1024):
        with open(annotation_file) as f:
            coco = json.load(f)

        self.image_root = image_root
        self.max_size = max_size
        self.cat_names = {c['id']: c['name'] for c in coco['categories']}

        # Group annotations by image
        self.samples = []
        anns_by_img = {}
        for ann in coco['annotations']:
            anns_by_img.setdefault(ann['image_id'], []).append(ann)

        for img_info in coco['images']:
            img_path = os.path.join(image_root, img_info['file_name'])
            if os.path.exists(img_path):
                self.samples.append({
                    'path': img_path,
                    'width': img_info['width'],
                    'height': img_info['height'],
                    'annotations': anns_by_img.get(img_info['id'], []),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        w, h = image.size

        # Get boxes in normalized [cx, cy, w, h] format
        boxes = []
        for ann in sample['annotations']:
            x, y, bw, bh = ann['bbox']  # COCO [x, y, w, h]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            boxes.append([cx, cy, nw, nh])

        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.zeros(len(boxes), dtype=torch.long),
            'category_text': self.cat_names.get(0, 'telephone pole'),
        }


def freeze_backbone(model):
    """Freeze backbone, keep detection decoder trainable."""
    frozen = trainable = 0
    for name, param in model.named_parameters():
        # Freeze ViT backbone and text encoder, keep decoder/neck/head trainable
        if any(k in name for k in ['trunk', 'text_encoder', 'patch_embed', 'pos_embed']):
            param.requires_grad = False
            frozen += 1
        else:
            trainable += 1
    print(f"Frozen: {frozen} params, Trainable: {trainable} params")
    return model


def train(args):
    # Enable bf16 globally (SAM3 requirement)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = 'cuda'
    print(f"Device: {device}")

    # Build SAM3 model
    import sam3 as sam3_module
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_root = os.path.join(os.path.dirname(sam3_module.__file__), '..')
    bpe_path = os.path.join(sam3_root, 'assets', 'bpe_simple_vocab_16e6.txt.gz')

    print("Loading SAM3...")
    model = build_sam3_image_model(
        bpe_path=bpe_path, device=device,
        checkpoint_path=SAM3_CKPT, load_from_HF=False, eval_mode=False
    )
    model = freeze_backbone(model)
    gpu_memory_report()

    # Dataset
    train_ds = PoleDatasetSAM3(
        os.path.join(args.training_dir, 'sam3', 'train.json'),
        os.path.join(args.training_dir, 'images'),
    )
    val_ds = PoleDatasetSAM3(
        os.path.join(args.training_dir, 'sam3', 'val.json'),
        os.path.join(args.training_dir, 'images'),
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Processor for inference
    processor = Sam3Processor(model, confidence_threshold=0.1)

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(train_ds) * args.epochs
    warmup_steps = min(50, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_score = 0
    t_start = time.time()
    step = 0

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for idx in range(len(train_ds)):
            sample = train_ds[idx]
            image = sample['image']
            gt_boxes = sample['boxes']  # normalized [cx, cy, w, h]
            text = sample['category_text']

            if len(gt_boxes) == 0:
                continue

            try:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    # Use processor for image preprocessing
                    with torch.no_grad():
                        state = processor.set_image(image)

                    # Forward text
                    text_outputs = model.backbone.forward_text([text], device=device)
                    state["backbone_out"].update(text_outputs)
                    state["geometric_prompt"] = model._get_dummy_prompt()

                    # Forward grounding in eval mode (no find_target needed)
                    model.eval()
                    outputs = model.forward_grounding(
                        backbone_out=state["backbone_out"],
                        find_input=processor.find_stage,
                        geometric_prompt=state["geometric_prompt"],
                        find_target=None,
                    )
                    model.train()

                    # Get predictions
                    pred_boxes = outputs["pred_boxes"]  # (N, 4) normalized cxcywh
                    pred_logits = outputs["pred_logits"]  # (N, C)

                    # Simple matching loss: for each GT box, find nearest pred
                    gt = gt_boxes.to(device)  # (M, 4)
                    with torch.amp.autocast('cuda', enabled=False):
                        pred_b = pred_boxes.float()
                        gt_b = gt.float()

                        # Pairwise L1 distance
                        cost = torch.cdist(pred_b, gt_b, p=1)  # (N, M)
                        # Hungarian-like: assign each GT to nearest pred
                        min_costs, matched_pred = cost.min(dim=0)  # (M,)

                        # Box regression loss on matched predictions
                        box_loss = min_costs.mean()

                        # Classification loss: matched preds should have high logit
                        pos_logits = pred_logits[matched_pred].squeeze(-1)
                        neg_mask = torch.ones(len(pred_logits), dtype=torch.bool, device=device)
                        neg_mask[matched_pred] = False
                        neg_logits = pred_logits[neg_mask].squeeze(-1)

                        cls_loss = -torch.log(pos_logits.sigmoid() + 1e-6).mean()
                        if len(neg_logits) > 0:
                            cls_loss += -torch.log(1 - neg_logits.sigmoid() + 1e-6).mean() * 0.5

                        loss = box_loss + cls_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                step += 1

                train_losses.append(loss.item())

            except Exception as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

        avg_loss = sum(train_losses) / max(len(train_losses), 1)

        # Quick validation: count detections on val set
        model.eval()
        val_dets = 0
        val_gt = 0
        with torch.no_grad():
            for idx in range(len(val_ds)):
                sample = val_ds[idx]
                val_gt += len(sample['boxes'])
                try:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        state = processor.set_image(sample['image'])
                        state = processor.set_text_prompt(state=state, prompt=sample['category_text'])
                        val_dets += len(state['boxes'])
                except:
                    pass

        elapsed = time.time() - t_start
        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} val_dets={val_dets} (gt={val_gt}) lr={lr:.2e} | {elapsed:.0f}s",
              flush=True)

        # Save best (by detection count closest to GT)
        score = -abs(val_dets - val_gt)  # closer to GT count = better
        if score > best_val_score or epoch == 0:
            best_val_score = score
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best.pt'))
            print(f"    -> Saved best model")

    # Save final
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'final.pt'))
    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Models saved to {MODEL_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-dir', default='data/training')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
