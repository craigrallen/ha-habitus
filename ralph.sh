#!/usr/bin/env bash
# ralph.sh — Autonomous Habitus coding loop (Ralph Wiggum style)
# Usage:  bash ralph.sh <iterations>
# Example: bash ralph.sh 5
#
# Each iteration: Claude picks a task from PRD.md, implements it,
# passes quality gates, commits, and logs progress to progress.txt.
# The loop exits early if Claude signals <promise>COMPLETE</promise>.

set -e

# Add Python dev tools to PATH (Windows — installed via pip but not on PATH by default)
export PATH="/c/Users/Widemind/AppData/Local/Python/pythoncore-3.14-64/Scripts:$PATH"

# Allow launching claude CLI from inside an existing Claude Code session
unset CLAUDECODE

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PRD="$REPO_DIR/PRD.md"
PROGRESS="$REPO_DIR/progress.txt"
AGENTS="$REPO_DIR/AGENTS.md"

if [ -z "$1" ]; then
  echo "Usage: bash ralph.sh <iterations>"
  echo "Example: bash ralph.sh 5"
  exit 1
fi

ITERATIONS=$1
echo "============================================"
echo " Ralph Wiggum — Habitus autonomous coding"
echo " Iterations: $ITERATIONS"
echo " $(date)"
echo "============================================"
echo ""

cd "$REPO_DIR"

for ((i=1; i<=ITERATIONS; i++)); do
  echo ""
  echo "--- Iteration $i / $ITERATIONS  [$(date +%H:%M:%S)] ---"
  echo ""

  result=$(claude --dangerously-skip-permissions -p \
"@PRD.md @progress.txt @AGENTS.md

You are working autonomously on the Habitus Home Assistant add-on.

INSTRUCTIONS:
1. Read PRD.md to find the highest-priority incomplete task (not yet in progress.txt).
2. Read the relevant source files before making any changes.
3. Implement ONLY that single task — one task per iteration.
4. Run quality gates in this order:
   a. ruff check habitus/habitus/
   b. black --check habitus/habitus/
   c. mypy habitus/habitus/
   d. pytest --cov=habitus/habitus --cov-report=term-missing
   Fix any failures before committing.
5. Commit with a descriptive message (feat: or fix: prefix).
6. Append a one-line entry to progress.txt:
   [$(date '+%Y-%m-%d %H:%M')] TASK-XXX: brief description
7. If ALL high-priority tasks (TASK-001 through TASK-006) are complete, output exactly:
   <promise>COMPLETE</promise>

CONSTRAINTS:
- Only modify files in habitus/habitus/, tests/, or habitus/www/
- Never modify PRD.md, progress.txt, AGENTS.md, CLAUDE.md, or ralph.sh
- One feature per commit
- Do not skip quality gates")

  echo "$result"

  if [[ "$result" == *"<promise>COMPLETE</promise>"* ]]; then
    echo ""
    echo "============================================"
    echo " All high-priority tasks complete!"
    echo "============================================"
    exit 0
  fi
done

echo ""
echo "============================================"
echo " Ralph finished $ITERATIONS iterations."
echo " Check progress.txt for completed tasks."
echo "============================================"
