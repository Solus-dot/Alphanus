---
name: git
description: Inspect and update Git repositories inside the workspace with guarded, structured Git operations.
allowed-tools: git_status git_log git_diff git_show git_branch_list git_branch_create git_branch_switch git_add git_commit git_fetch git_pull git_push git_init
metadata:
  version: "1.0.0"
  tags:
    - git
    - version-control
    - repository
    - branch
  triggers:
    keywords:
      - git
      - branch
      - commit
      - diff
      - status
      - repository
---
Use these tools for focused Git work inside the workspace.

Rules:
- Keep all Git operations inside the configured workspace root.
- Use the structured Git tools here instead of raw shell commands for supported Git tasks.
- Do not request or manage credentials. Git commands use the system Git credential configuration.
- Use `git_status`, `git_log`, `git_diff`, and `git_show` for inspection.
- Use `git_add` only for paths the user intends to stage.
- Use `git_commit` only with a non-empty message. It commits staged changes only and rejects no-op commits.
- Use `git_branch_switch` only when the working tree is clean; the tool enforces this.
- Use `git_pull` without an explicit mode unless the user requests otherwise; it defaults to `--rebase`.
- Use `git_push` only when the user has explicitly confirmed the push via the tool's confirmation parameter.
- Never attempt force-push modes; the tool rejects them.
- Use `git_init` only for an explicit target path inside the workspace. It is forbidden at the workspace root, and nested repository initialization is blocked when the target is already under another Git repository.
