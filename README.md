# ml-paper-daily

A running log of recent machine-learning papers, implemented **incrementally**.

Every day an automated pipeline (Claude Code in headless mode, driven by cron)
advances one paper by roughly **a quarter of its core method** — not a full,
faithful reproduction. Over four days a single paper goes from "foundational
building block" to "small end-to-end demo on toy data." Then a new paper is
picked and the cycle repeats.

The point is **breadth + honesty**, not completeness:

- Papers are picked from recent, well-regarded work (computer vision,
  transformers/architecture, and other genuinely notable areas).
- Each implementation is small, runnable, plain Python/NumPy/PyTorch, and is
  actually executed before being committed. Broken code is never committed.
- Because a paper is only implemented one quarter per day, large chunks are
  deliberately **simplified or stubbed**. That is by design.

## How to read this repo

Each paper lives in its own folder:

```
papers/<slug>/
    NOTES.md   <-- read this first
    ...        <-- code + minimal tests
```

**`papers/<slug>/NOTES.md` is the source of truth.** It documents the paper
(title, arXiv link, authors, a paraphrased summary), the 4-pass plan, and —
crucially — **what is actually implemented vs. what was simplified or skipped,
and why.** Do not assume a folder is a complete reproduction; assume the
opposite unless NOTES.md says otherwise.

## The 4 passes

1. **Pass 1** — the most foundational piece (small, runnable, tested).
2. **Pass 2** — the mechanism that makes the paper distinctive.
3. **Pass 3** — another real increment, or an honest simplification of a
   harder part.
4. **Pass 4** — a small end-to-end demo on toy/synthetic data, plus a final
   honest summary in NOTES.md.

## Automation

See [`SETUP.md`](SETUP.md) for how the daily cron pipeline works, where logs
live, how to check progress, and how to pause it. The scripts and prompts that
drive everything live in [`automation/`](automation/).
