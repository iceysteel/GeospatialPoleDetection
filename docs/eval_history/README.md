# Evaluation History

Tracking F1 scores at **10m match radius** (industry standard) against verified GT (94 poles).

| Run | Date | Pipeline | Config | P(10m) | R(10m) | F1(10m) | Notes |
|-----|------|----------|--------|--------|--------|---------|-------|
| 001 | 2026-04-05 | GDino-ft + MASt3R + VLM | GD>=0.22 VLM>=0.9 | 45.1% | 44.6% | 0.448 | First verified GT baseline |
| 001 | 2026-04-05 | GDino-ft + MASt3R + VLM | GD>=0.35 VLM>=0.7 | 40.2% | 48.9% | 0.441 | Higher recall variant |
| 001 | 2026-04-05 | GDino-ft + MASt3R raw | no VLM | 28.5% | 59.8% | 0.386 | No VLM filtering |
| 001 | 2026-04-05 | SAM3 + AerialMegaDepth→ortho | base SAM3 thresh=0.10 | 21.4% | 6.4% | 0.098 | Oblique→ortho projection |
| 002 | 2026-04-05 | SAM3-LoRA v2 + AerialMegaDepth→ortho | thresh=0.10, clean GT training | 17.0% | 16.0% | 0.165 | LoRA v2 with MASt3R-projected training data |
| 003 | 2026-04-05 | AutoResearch baseline | SAM3 base + AerialMegaDepth, grid cells | 21.0% | 81.9% | 0.335 | New eval harness, all grid cells |
| 004 | 2026-04-05 | AutoResearch (20 experiments) | SAM3 thresh=0.40, ortho=60m | 56.9% | 61.7% | 0.592 | Autonomous loop, 6 improvements |
| 005 | 2026-04-05 | AutoResearch extended (30 exp) | SAM3 thresh=0.45, ortho=60m | 61.1% | 58.5% | 0.598 | Best from autonomous loop |
| 006 | 2026-04-05 | Holdout eval (verified GT) | Same config as test | 81.1% | 63.8% | 0.714 | NOT overfit — holdout > test |
