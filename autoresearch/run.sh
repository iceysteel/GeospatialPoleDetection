#!/bin/bash
# AutoResearch Loop Runner — Deep exploration with web research
cd "$(dirname "$0")/.."

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "========================================"
    echo "AutoResearch Iteration $ITERATION"
    echo "$(date)"
    echo "========================================"

    claude --dangerously-skip-permissions -p "You are an autonomous ML research agent. Your goal: improve F1@10m for pole detection.

CURRENT BEST: F1@10m = 0.714 (P=81.1%, R=63.8%). Parameter tuning is EXHAUSTED.

YOU MUST TRY SOMETHING FUNDAMENTALLY DIFFERENT. Search the web for ideas.

APPROACH:
1. Read autoresearch/program.md and autoresearch/autoresearch.jsonl for history
2. Read autoresearch/pipeline.py to understand the current pipeline
3. SEARCH THE WEB for techniques to improve geospatial object detection:
   - Better SAM3 prompting strategies (visual exemplars, few-shot)
   - MASt3R tricks for better oblique→ortho projection
   - Post-processing techniques for pole detection in aerial imagery
   - Spatial reasoning (poles follow streets at regular intervals)
   - Multi-scale detection approaches
   - Novel deduplication or consensus strategies
   - Any papers on utility pole detection from aerial/satellite imagery
4. Based on your research, implement ONE change to autoresearch/pipeline.py
5. git add -A && git commit -m 'experiment: <description>'
6. Run: python autoresearch/prepare.py
7. If F1 IMPROVED: keep, update population.json + program.md
   If F1 WORSENED: git revert HEAD --no-edit, update program.md with what failed
8. Log result to autoresearch/autoresearch.jsonl
9. git add autoresearch/ && git commit -m 'autoresearch: log iteration'

WHAT YOU CAN DO:
- Modify autoresearch/pipeline.py (any changes)
- Search the web for papers, techniques, code examples
- Run training scripts if needed (up to 30 min per iteration)
- Add new post-processing, spatial filtering, scoring adjustments
- Change detection strategy, MASt3R settings, dedup logic
- Install new packages if needed

CONSTRAINTS:
- Must use SAM3 for detection in oblique views
- Must use MASt3R for oblique→ortho mapping
- Do NOT modify prepare.py
- 2x NVIDIA 3090 (24GB each), 256GB RAM available

THINK DEEPLY. The easy wins are gone. What would a research paper do differently?

GO."

    echo "Iteration $ITERATION complete"
    sleep 5
done
