#!/bin/bash
# AutoResearch Loop Runner
# Uses Claude Code with --dangerously-skip-permissions to allow autonomous file edits
#
# Usage: ./autoresearch/run.sh

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

YOUR TASK: Make ONE change to autoresearch/pipeline.py to improve F1@10m.

STEPS:
1. Read autoresearch/program.md for context, constraints, and what's been tried
2. Read autoresearch/population.json for top performing configs
3. Read autoresearch/autoresearch.jsonl for recent experiment results
4. Read autoresearch/pipeline.py to understand current state
5. Make ONE targeted change (mutate a parameter, try a new approach)
6. Run: git add autoresearch/pipeline.py && git commit -m 'experiment: <description>'
7. Run: python autoresearch/prepare.py
8. Read the F1@10m result from the output
9. If F1 IMPROVED over best in population.json:
   - Update population.json with new config
   - Update program.md with what worked
   - Keep the commit
10. If F1 WORSENED:
    - Run: git revert HEAD --no-edit
    - Update program.md with what failed
11. Append result to autoresearch/autoresearch.jsonl
12. git add autoresearch/ && git commit -m 'autoresearch: log iteration $ITERATION'

CONSTRAINTS:
- Must use SAM3 for detection in oblique views
- Must use MASt3R for oblique→ortho cross-view mapping
- Do NOT modify prepare.py
- Make only ONE change per iteration
- Do NOT pause to ask — just run the experiment
- Stay within 10 minute timeout

Current best F1@10m from autoresearch.jsonl — beat it!

GO."

    echo "Iteration $ITERATION complete"
    sleep 5
done
