#!/bin/bash
# AutoResearch Loop Runner
# Launches Claude Code to autonomously optimize the pipeline
#
# Usage: ./autoresearch/run.sh
#
# The agent will:
# 1. Read program.md + population.json + recent git log
# 2. Propose ONE change to pipeline.py
# 3. Commit, run prepare.py, evaluate F1@10m
# 4. Keep if improved, revert if not
# 5. Log to autoresearch.jsonl, update program.md
# 6. Repeat

cd "$(dirname "$0")/.."

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "========================================"
    echo "AutoResearch Iteration $ITERATION"
    echo "$(date)"
    echo "========================================"

    claude --print "You are an autonomous ML research agent optimizing a pole detection pipeline.

YOUR TASK: Make ONE change to autoresearch/pipeline.py to improve F1@10m.

STEPS:
1. Read autoresearch/program.md for context, constraints, and what's been tried
2. Read autoresearch/population.json for top performing configs
3. Read git log --oneline -20 for recent experiment history
4. Read autoresearch/pipeline.py to understand current state
5. Make ONE targeted change (mutate a parameter, try a new approach)
6. git add autoresearch/pipeline.py && git commit -m 'experiment: <description>'
7. Run: python autoresearch/prepare.py
8. Read the F1@10m result
9. If F1 IMPROVED over best in population.json:
   - Update population.json with new config
   - Update program.md with what worked
   - Keep the commit
10. If F1 WORSENED:
    - git revert HEAD --no-edit
    - Update program.md with what failed
11. Log result to autoresearch/autoresearch.jsonl

CONSTRAINTS:
- Must use SAM3 for detection
- Must use MASt3R for oblique→ortho mapping
- Do NOT modify prepare.py
- Make only ONE change per iteration
- Do NOT pause to ask — just run the experiment

GO."

    echo "Iteration $ITERATION complete"
    sleep 5
done
