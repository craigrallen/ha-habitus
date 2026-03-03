Turn a feature idea into structured PRD tasks for Habitus.

Feature description: $ARGUMENTS

Steps:
1. Understand the feature in the context of Habitus architecture (read CLAUDE.md and relevant source files if needed)
2. Break it into 1–4 discrete, implementable tasks
3. For each task, specify:
   - Task ID (next available TASK-XXX number from PRD.md)
   - Title
   - Which file(s) to modify
   - Description of what to implement
   - Output artifacts (JSON files, new functions, etc.)
   - Test requirements
   - Acceptance criteria (quality gates)
4. Ask the user to confirm before appending to PRD.md
5. If confirmed, append the new tasks to PRD.md under the appropriate batch

Keep tasks small enough that one can be completed in a single autonomous iteration.
