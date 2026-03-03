Start the Ralph Wiggum autonomous coding loop for Habitus.

The argument is the number of iterations: $ARGUMENTS

Run:
```bash
bash ralph.sh $ARGUMENTS
```

If no number is given, default to 3 iterations.

Before starting:
1. Show the user which tasks in PRD.md are still incomplete (not in progress.txt)
2. Confirm the quality gates are available (ruff, black, mypy, pytest)
3. Start the loop

Monitor the output and summarise what was completed after the loop finishes.
