Show the current Habitus project status.

1. Read progress.txt and list all completed tasks
2. Read PRD.md and identify remaining incomplete tasks
3. Show a summary table:

| Status | Count | Tasks |
|--------|-------|-------|
| Done   | N     | TASK-XXX, ... |
| Remaining | N  | TASK-XXX, ... |

4. Show the next recommended task to work on
5. Run a quick check: `pytest --co -q` to count available tests
6. Show the last git commit message and date
