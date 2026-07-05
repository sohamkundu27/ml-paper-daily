#!/bin/bash
#
# daily_paper_task.sh
#
# Runs once per day (via cron). Advances the ml-paper-daily project by exactly
# ONE quarter-pass:
#   - QUARTER=0  -> start a NEW paper (implement pass 1)
#   - QUARTER=1  -> implement pass 2 of the current paper
#   - QUARTER=2  -> implement pass 3 of the current paper
#   - QUARTER=3  -> implement pass 4 (final) of the current paper, then reset
#
# State lives in ~/ml-paper-daily/.paper_state (git-ignored):
#     QUARTER=<0-4>
#     SLUG=<slug or empty>
#
# All work is delegated to Claude Code in headless (`-p`) mode on the cheapest
# model. Cost is bounded per run by --max-budget-usd.
#
# NOTE (macOS/portability):
#   * `flock` is not shipped on macOS; we prefer flock when available and fall
#     back to macOS `shlock` (which detects stale locks via PID), then mkdir.
#   * `python`/`pip` do not exist on macOS (only `python3`/`pip3`), so the
#     allowed-tools list authorizes both spellings.
#   * This CLI build has no `--max-turns` flag, so cost is bounded solely by
#     `--max-budget-usd`.

set -uo pipefail

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO="$HOME/ml-paper-daily"
STATE_FILE="$REPO/.paper_state"
SLUG_FILE="$REPO/.current_paper_slug"
LOG="$REPO/paper_repo_cron.log"
LOCK="$HOME/.ml-paper-daily.lock"   # kept OUTSIDE the repo so it's never committed

MODEL="claude-haiku-4-5-20251001"
NEW_PAPER_TOOLS="Read,Edit,Write,Grep,WebSearch,Bash(git:*),Bash(python:*),Bash(python3:*),Bash(pytest:*),Bash(pip:*),Bash(pip3:*)"
CONTINUE_TOOLS="Read,Edit,Write,Grep,Bash(git:*),Bash(python:*),Bash(python3:*),Bash(pytest:*),Bash(pip:*),Bash(pip3:*)"
NEW_PAPER_BUDGET="0.75"
CONTINUE_BUDGET="0.50"

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] $*" >> "$LOG"
}

# ---------------------------------------------------------------------------
# Single-instance lock (prefer flock; fall back to shlock, then mkdir)
# ---------------------------------------------------------------------------
acquire_lock() {
    if command -v flock >/dev/null 2>&1; then
        exec 9>"$LOCK"
        if ! flock -n 9; then
            log "lock skipped — another run holds $LOCK (flock)"
            exit 0
        fi
        return 0
    fi

    if command -v shlock >/dev/null 2>&1; then
        if shlock -f "$LOCK" -p "$$"; then
            trap 'rm -f "$LOCK"' EXIT
            return 0
        fi
        log "lock skipped — another run holds $LOCK (shlock)"
        exit 0
    fi

    # Last-resort portable lock: mkdir is atomic on POSIX filesystems.
    if mkdir "$LOCK.d" 2>/dev/null; then
        trap 'rmdir "$LOCK.d" 2>/dev/null' EXIT
        return 0
    fi
    log "lock skipped — another run holds $LOCK.d (mkdir)"
    exit 0
}

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
write_state() {
    # $1 = QUARTER, $2 = SLUG
    printf 'QUARTER=%s\nSLUG=%s\n' "$1" "$2" > "$STATE_FILE"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
acquire_lock

mkdir -p "$REPO"
cd "$REPO" || { log "FATAL — cannot cd into $REPO"; exit 1; }

log "run started (cwd=$REPO)"

# Sync with remote before doing any work.
git pull --rebase --autostash >> "$LOG" 2>&1 && log "git pull ok" || log "git pull returned non-zero (continuing)"

# --- Read state (default QUARTER=0 if missing/invalid) ---
QUARTER=0
SLUG=""
if [[ -f "$STATE_FILE" ]]; then
    QUARTER="$(grep -E '^QUARTER=' "$STATE_FILE" | head -1 | cut -d= -f2- | tr -d '[:space:]')"
    SLUG="$(grep -E '^SLUG=' "$STATE_FILE" | head -1 | cut -d= -f2- | tr -d '[:space:]')"
fi
[[ "$QUARTER" =~ ^[0-4]$ ]] || QUARTER=0

log "state: QUARTER=$QUARTER SLUG=${SLUG:-<none>}"

# Guard against corrupted state: mid-paper but no slug -> restart cleanly.
if [[ "$QUARTER" != "0" && -z "$SLUG" ]]; then
    log "corrupted state (QUARTER=$QUARTER but empty SLUG) — resetting to QUARTER=0"
    write_state 0 ""
    QUARTER=0
fi

# ===========================================================================
# BRANCH A — start a new paper (pass 1)
# ===========================================================================
if [[ "$QUARTER" == "0" ]]; then
    log "branch: NEW PAPER (pass 1) — invoking claude ($MODEL)"

    PROMPT="$(cat "$REPO/automation/prompt_new_paper.txt")"

    claude -p "$PROMPT" \
        --model "$MODEL" \
        --allowedTools "$NEW_PAPER_TOOLS" \
        --max-budget-usd "$NEW_PAPER_BUDGET" \
        --output-format json >> "$LOG" 2>&1
    rc=$?
    log "claude (new paper) exited rc=$rc"

    # Read the slug the agent was asked to write.
    NEW_SLUG=""
    if [[ -f "$SLUG_FILE" ]]; then
        NEW_SLUG="$(tr -d '[:space:]' < "$SLUG_FILE")"
    fi

    if [[ -n "$NEW_SLUG" ]]; then
        write_state 1 "$NEW_SLUG"
        log "new paper started: slug=$NEW_SLUG — state set to QUARTER=1"
    else
        # Agent did not produce a slug (likely failed). Stay at QUARTER=0 so
        # tomorrow retries a fresh paper instead of continuing a phantom one.
        write_state 0 ""
        log "new paper run produced no .current_paper_slug — staying at QUARTER=0 to retry tomorrow"
    fi

    log "run finished"
    exit 0
fi

# ===========================================================================
# BRANCH B — continue current paper (pass 2, 3, or 4)
# ===========================================================================
PASS_NUM=$(( QUARTER + 1 ))
log "branch: CONTINUE paper slug=$SLUG pass=$PASS_NUM — invoking claude ($MODEL)"

TEMPLATE="$(cat "$REPO/automation/prompt_continue_paper.txt")"
PROMPT="${TEMPLATE//\{SLUG\}/$SLUG}"
PROMPT="${PROMPT//\{PASS_NUM\}/$PASS_NUM}"

claude -p "$PROMPT" \
    --model "$MODEL" \
    --allowedTools "$CONTINUE_TOOLS" \
    --max-budget-usd "$CONTINUE_BUDGET" \
    --output-format json >> "$LOG" 2>&1
rc=$?
log "claude (continue pass $PASS_NUM) exited rc=$rc"

if [[ "$PASS_NUM" -ge 4 ]]; then
    write_state 0 ""
    rm -f "$SLUG_FILE"
    log "paper finished: slug=$SLUG (pass 4 complete) — state reset to QUARTER=0 for next paper"
else
    write_state "$PASS_NUM" "$SLUG"
    log "pass $PASS_NUM completed for slug=$SLUG — state set to QUARTER=$PASS_NUM"
fi

log "run finished"
exit 0
