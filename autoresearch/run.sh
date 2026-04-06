#!/bin/bash
# AutoResearch Loop Runner — Extended capabilities
# Agent can now do deeper changes: fine-tuning, architectural modifications, etc.

cd "$(dirname "$0")/.."

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "========================================"
    echo "AutoResearch Iteration $ITERATION"
    echo "$(date)"
    echo "========================================"

    claude --dangerously-skip-permissions -p "You are an autonomous ML research agent optimizing a pole detection pipeline.
The metric to optimize is AVERAGE F1@10m across TEST and HOLDOUT areas.
This prevents overfitting — if test improves but holdout drops, that's overfit.
Current best avg F1@10m: check autoresearch.jsonl for latest.

YOUR TASK: Make changes to improve F1@10m. You have EXTENDED capabilities now.

WHAT YOU CAN DO:
- Modify autoresearch/pipeline.py (parameters, architecture, logic)
- Run training scripts (fine-tune SAM3-LoRA, retrain models)
- Modify detection prompts, thresholds, post-processing
- Add VLM filtering steps (Qwen 3.5 27B via ollama)
- Change MASt3R settings, ortho crop strategies
- Ensemble multiple detectors
- Add new post-processing (score calibration, spatial filtering)
- Take up to 30 minutes per iteration if needed for training

STEPS:
1. Read autoresearch/program.md for context and history
2. Read autoresearch/population.json and autoresearch/autoresearch.jsonl
3. Read autoresearch/pipeline.py
4. Decide on a change — can be quick (parameter tweak) or deep (fine-tune a model)
5. Implement the change
6. git add -A && git commit -m 'experiment: <description>'
7. Run: python autoresearch/prepare.py
8. If F1 IMPROVED: keep commit, update population.json and program.md
   If F1 WORSENED: git revert HEAD --no-edit, update program.md with failure
9. Log to autoresearch/autoresearch.jsonl
10. git add autoresearch/ && git commit -m 'autoresearch: log iteration'

CONSTRAINTS:
- Must use SAM3 for detection in oblique views
- Must use MASt3R for oblique→ortho cross-view mapping
- Do NOT modify prepare.py
- Hardware: 2x NVIDIA 3090 (24GB each), 256GB RAM
- Qwen 3.5 27B available via ollama on localhost:11434

IMPORTANT: Parameter tuning has plateaued at F1=0.592. Think DEEPER:
- Can you add VLM post-filtering to remove false positives?
- Can you fine-tune SAM3-LoRA with better data/hyperparams?
- Can you use multi-view information more cleverly?
- Can you improve the MASt3R projection accuracy?
- Can you combine multiple detection strategies?

Do NOT pause to ask. Just run the experiment. GO."

    echo "Iteration $ITERATION complete"
    sleep 5
done
