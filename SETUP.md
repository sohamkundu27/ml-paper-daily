# SETUP — how the daily automation works

This repo runs itself. A cron job invokes Claude Code in headless mode once a
day to advance one paper by a single quarter-pass. No human interaction is
required.

## What runs, and when

- **Schedule:** twice a day. The **09:00** slot always runs (guaranteeing at
  least one commit every day); the **17:00** slot runs only ~50% of the time
  (`SKIP_PROBABILITY=50`, a coin flip inside the script). Net effect: a
  **random 1–2 commits per day**.
- **Crontab lines:**
  ```
  0 9 * * *  /bin/bash /home/soham/ml-paper-daily/automation/daily_paper_task.sh
  0 17 * * * SKIP_PROBABILITY=50 /bin/bash /home/soham/ml-paper-daily/automation/daily_paper_task.sh
  ```
- **Driver script:** [`automation/daily_paper_task.sh`](automation/daily_paper_task.sh)
  — it prepends nvm's bin to `PATH` and sets `GIT_SSH_COMMAND` so it works under
  cron's minimal environment.
- **Model:** `claude-haiku-4-5-20251001` (the cheapest current model), headless
  (`claude -p ... --permission-mode acceptEdits --output-format json`), wrapped
  in `timeout` (20 min).
- **Cost cap per run:** `--max-budget-usd 0.75` on new-paper runs,
  `--max-budget-usd 0.50` on continuation runs (so ≤ ~$1.25 on a 2-commit day).

## The daily state machine

State lives in `~/ml-paper-daily/.paper_state` (git-ignored):

```
QUARTER=<0-4>
SLUG=<slug or empty>
```

| QUARTER on wake | Action taken                                  | New state        |
|-----------------|-----------------------------------------------|------------------|
| 0 (or missing)  | Pick a NEW paper, implement **pass 1**        | QUARTER=1, SLUG  |
| 1               | Implement **pass 2** of current paper         | QUARTER=2        |
| 2               | Implement **pass 3** of current paper         | QUARTER=3        |
| 3               | Implement **pass 4** (final), then reset      | QUARTER=0, empty |

So each paper takes four passes. With a random 1–2 passes landing per day, a
paper completes every ~2–4 days; then the cycle restarts with a brand-new
paper. This repeats indefinitely.

The prompts driving the agent are
[`automation/prompt_new_paper.txt`](automation/prompt_new_paper.txt) and
[`automation/prompt_continue_paper.txt`](automation/prompt_continue_paper.txt).

## Where the log lives

All output — timestamped state-machine lines plus the raw JSON from each Claude
run — is appended to:

```
~/ml-paper-daily/paper_repo_cron.log
```

(This file is git-ignored, so it stays local.)

## How to check progress

```bash
# What the state machine has been doing:
grep -E '^\[' ~/ml-paper-daily/paper_repo_cron.log | tail -20

# Current position in the cycle:
cat ~/ml-paper-daily/.paper_state

# Full history is just the git log / the papers/ folders:
cd ~/ml-paper-daily && git log --oneline -20
ls papers/
```

Each paper folder's `NOTES.md` records what was actually implemented vs.
simplified, pass by pass.

## How to pause / resume / stop

- **Pause:** comment out both `daily_paper_task.sh` lines in your crontab.
  ```bash
  crontab -l | sed 's#^\(.*daily_paper_task.sh\)#\# \1#' | crontab -
  ```
  (Or run `crontab -e` and put a `#` in front of each line.)
- **Resume:** remove the leading `# ` from both lines via `crontab -e`.
- **Stop entirely:** delete both lines with `crontab -e`.
- **Run once manually (costs API usage, makes a real commit + push):**
  ```bash
  /bin/bash ~/ml-paper-daily/automation/daily_paper_task.sh
  ```

## Concurrency & safety notes

- A lock (`~/.ml-paper-daily.lock`) prevents overlapping runs. The script
  prefers `flock`, and falls back to macOS `shlock` (stale-lock aware) or an
  atomic `mkdir` lock.
- The agent runs with a restricted tool allow-list (read/edit/write/grep, web
  search on new-paper days, and git/python/pytest/pip in Bash only).
- Non-interactive `git push` works over **SSH**: `origin` is
  `git@github.com:sohamkundu27/ml-paper-daily.git`, and the script exports
  `GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"`
  so cron never blocks on a passphrase or host-key prompt. The SSH key at
  `~/.ssh/id_ed25519` is already authorized on GitHub.

## Platform notes (Linux)

- This machine is **Linux**; cron runs via the system `cron` daemon
  (`systemctl is-active cron` → `active`), which picks up the user crontab
  automatically. No launchd, no `gh`.
- Cron runs with a **minimal `PATH`**, so the script prepends nvm's bin
  (`/home/soham/.nvm/versions/node/v24.16.0/bin`) and resolves the `claude` CLI
  by absolute path. If you upgrade Node via nvm, update that path (the script
  tries `command -v claude` first, then falls back to it).
- `--max-turns` is omitted (unsupported by this CLI build; cost is bounded by
  `--max-budget-usd` instead). The tool allow-list authorizes both `python`/
  `pip` and `python3`/`pip3` spellings for portability.
