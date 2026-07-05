# SETUP — how the daily automation works

This repo runs itself. A cron job invokes Claude Code in headless mode once a
day to advance one paper by a single quarter-pass. No human interaction is
required.

## What runs, and when

- **Schedule:** every day at **09:00** (local machine time).
- **Crontab line:**
  ```
  0 9 * * * /bin/bash $HOME/ml-paper-daily/automation/daily_paper_task.sh
  ```
- **Driver script:** [`automation/daily_paper_task.sh`](automation/daily_paper_task.sh)
- **Model:** `claude-haiku-4-5-20251001` (the cheapest current model), headless
  (`claude -p ... --output-format json`).
- **Cost cap per run:** `--max-budget-usd 0.75` on new-paper days,
  `--max-budget-usd 0.50` on continuation days.

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

So each paper takes four days (one quarter per day); on the fifth day the cycle
restarts with a brand-new paper. This repeats indefinitely.

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

- **Pause:** comment out the line in your crontab.
  ```bash
  crontab -l | sed 's#^\(0 9 .*daily_paper_task.sh\)#\# \1#' | crontab -
  ```
  (Or run `crontab -e` and put a `#` in front of the line.)
- **Resume:** uncomment it (remove the leading `# `) via `crontab -e`.
- **Stop entirely:** remove the line with `crontab -e`.
- **Run once manually (costs API usage):**
  ```bash
  /bin/bash ~/ml-paper-daily/automation/daily_paper_task.sh
  ```

## Concurrency & safety notes

- A lock (`~/.ml-paper-daily.lock`) prevents overlapping runs. The script
  prefers `flock`, and falls back to macOS `shlock` (stale-lock aware) or an
  atomic `mkdir` lock.
- The agent runs with a restricted tool allow-list (read/edit/write/grep, web
  search on new-paper days, and git/python/pytest/pip in Bash only).
- Non-interactive `git push` works because `gh` is configured as the git
  credential helper (`gh auth setup-git`).

## Platform notes (macOS)

- This machine is macOS, so cron is managed by **launchd** (there is no
  `systemctl`). The `cron` daemon (`/usr/sbin/cron`) is active and picks up the
  user crontab automatically.
- If cron ever appears not to run, grant **Full Disk Access** to `/usr/sbin/cron`
  in *System Settings → Privacy & Security*, which some macOS versions require
  for cron jobs that touch user files.
- Deviations from a generic Linux setup, made so the pipeline actually runs
  here: `--max-turns` is omitted (unsupported by this CLI build; cost is bounded
  by `--max-budget-usd` instead), and the tool allow-list includes `python3`/
  `pip3` since macOS has no bare `python`/`pip`.
