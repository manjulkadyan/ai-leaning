# Lesson-writer brief

Every agent that writes or rewrites a lesson gets this file, then [lesson-checklist.md](lesson-checklist.md), then its own lesson-map entry. Read both files in full before writing anything. The checklist is part of the job: a lesson that fails any item is not finished.

## The reader

The reader is an experienced Android/Kotlin engineer (8+ years) who knows **nothing** about Python's internals, ML, or the maths behind it. They study alone, in about 90–120 minutes a day, and they read every word. They told us what they want: **every area covered, and every point explained in detail.** When in doubt, explain more, not less.

They are strong programmers. Don't explain what a variable or a loop *is*. Do explain how Python's version differs from Kotlin's, what happens underneath, and how it goes wrong.

## Step 1: build the topic inventory before writing

Before writing a single paragraph, write an inventory of everything a competent practitioner knows about this lesson's topic. Use these as references:

- the official documentation's table of contents for the topic (docs.python.org tutorial and library reference, numpy.org user guide, pandas user guide, and so on);
- one standard textbook chapter or course unit on it.

Tag each item:

- **must**: the reader will hit this in the next month of the course, or in their first real ML project. Teach it fully (Step 2).
- **should**: useful and common. Teach it at least briefly, with one example and its output.
- **later**: real but advanced. List it in a short "Going further" section with a link, so the reader knows it exists.

If the "must" list won't fit in the lesson's time, **don't compress**. Stop and tell the integrator which items need their own lesson. Splitting a topic is fine; skipping or skimming it is not.

Put the inventory in your final report, with each item marked covered (and its section heading) or moved elsewhere.

## Step 2: teach every "must" concept with the full pattern

Every "must" concept gets all six parts, in this order. A concept is not taught until it has all six.

1. **What it is.** One or two plain-language sentences. Define every new term the first time it appears, even ones that feel obvious (`dtype`, `axis`, "view", "iterable", "hashable").
2. **Why it exists.** What problem it solves, and when the reader will reach for it, in ML or in everyday code.
3. **Kotlin comparison** (Python topics) or **intuition first** (maths topics). For Python: show the Kotlin they already know, then the Python, and name what's the same and what's different. For maths: an analogy or picture before any symbol.
4. **A minimal example with its real output.** Short enough to read in ten seconds. Every line that produces a value shows that value.
5. **How it works underneath.** The mental model: what's in memory, which objects exist, or a step-by-step trace of the execution. Use a diagram or animation when the idea is spatial (references, memory layout, axes, broadcasting).
6. **How it goes wrong.** At least one realistic mistake, with the **real error message or wrong output**, how to recognise it, and the fix.

Then, where it helps, a second, more realistic example from ML or Android.

For maths, add: the formula introduced one symbol at a time, a worked numeric example by hand **before** the code, then the code reproducing the hand result.

Rules that apply throughout:

- A table may summarise concepts **after** they have been taught. It never replaces teaching them.
- No "obviously", "simply", "just", or "as you know". If it were obvious, it wouldn't need writing.
- When a line of code uses something not yet explained (a method, an operator, a format spec like `:.1%`), either explain it right there in one sentence or don't use it yet.
- Walk through tricky code line by line in prose or comments: comprehensions, slicing with negative steps, mutable defaults, `axis` reductions, broadcasting, recursion.
- Each section ends with the reader able to *do* something new. If a section only describes, add an example they can run.

## Lesson structure

Title `# Day N: ...`, then `**Phase P · Week W · ~M minutes**`, then these sections in order:

1. **Objectives**: a task list (`- [ ]`). Each one is something the reader can *do*, and each one is taught in the lesson.
2. **Prerequisites**: relative links to earlier lessons (`day-05-vectors.md`), naming what's used from each.
3. `---`, then **The big idea**: the motivation and an analogy, in plain words, before any detail.
4. **Core sections**, each concept taught with the six-part pattern.
5. At least one `!!! android "Android bridge"` that is technically accurate.
6. `!!! warning "Common mistake"` boxes, placed where the mistake happens, each showing the real error.
7. **Going further**: the "later" items from the inventory, one line each with a link.
8. `---`, then **Exercises**: 6–8 exercises, graded:
    - 2 warm-ups, practising one thing each;
    - 2–3 core exercises, combining things;
    - 1 "predict the output" exercise (answer first, then run);
    - 1 "find and fix the bug" exercise;
    - 1 challenge exercise.
9. **Solutions** in a collapsed `??? success "Solutions"` block. **Every** exercise gets a full solution that:
    - is runnable on its own: it re-creates its inputs, never relying on a name from an earlier block;
    - shows its real output;
    - ends with a sentence or two on *why* it works, or why the bug happened.
10. **Resources**: 5–7 free links you're sure exist.
11. **Checklist**: a task list mirroring the objectives.
12. A final line, exactly as the lesson map gives it: `**Next:** Day M: <title> (coming soon)`, as plain text, not a link. The integrator links it.

Review and mini-project days use their own structure:

1. Objectives.
2. "What this week gave you": a recap with a Mermaid concept map.
3. "Self-check": 8–10 questions with collapsed answers.
4. "Re-do from blank": the hardest exercise of the week.
5. The **Mini-project**: spec, starter code, milestones, stretch goals, a full reference solution in a collapsed block, and how to verify it works.
6. "Weekly retro" template.
7. Resources and Checklist.
8. One animated SVG summarising the week.

### Length

There is no length cap. A 90–120 minute lesson that follows Step 2 is usually **600–900 lines**. A lesson under 450 lines almost certainly skipped something; check the inventory again. Set the minutes in the header to what a beginner will actually need, including exercises.

### Depth references

Match the depth of these lessons:

- `docs/lessons/phase-1/day-08-shapes-indexing.md` and `day-09-broadcasting.md` (Python/NumPy, including prediction exercises);
- `day-22-norms-distances.md` and `day-25-projections-pca.md` (maths);
- `day-17-pandas-cleaning.md` (data work);
- `day-20-week3-review.md` (review days).

**Do not** use Days 1–5 as references: they are being rewritten for being too thin (see [review-week1.md](review-week1.md)).

## Code: it has to be perfect

- **Real output only.** Run every snippet with `python3 -I` from a scratch directory, not the repo, and paste the actual output. Never write "≈", "~" or "something like" for a value the code prints. Timings are the one exception: they depend on the machine, so say so ("on my laptop; yours will differ").
- **Runnable in page order.** Every Python block on the site has a ▶ Run button (Pyodide: Python 3.14, numpy 2.4, pandas 3.0, matplotlib 3.10). Blocks on a page share one namespace, top to bottom, like notebook cells. So:
    - imports and data loading come early;
    - never use a name before the block that defines it, and that includes solutions and illustrative snippets like `match device:`;
    - a block that errors on purpose is short and says so in a comment;
    - never read `sys.argv`, call `input()`, start threads, or write files other than `fig.savefig(...)` of a figure you embed.
- **Blocks that can't run in the browser**, such as multi-file packages, `torch`, `pip`, shell commands, or anything reading the local disk, start with a comment: `# Run this on your computer (or in Colab), not in the browser.`
- **Check runnability** by concatenating all blocks in page order and running them in one namespace from a scratch directory. The integrator's runner script does the same, and any failure sends the lesson back.
- Use `rng = np.random.default_rng(<seed>)` for anything random.
- **Libraries:** locally numpy 2.5, pandas 3.0, matplotlib 3.11. No scipy, sklearn or torch in runnable blocks unless the lesson is about them. Elsewhere, show them only as clearly marked "on your computer" code.
- **pandas 3:** Copy-on-Write and the `str` dtype are the default. Never use deprecated APIs (`inplace` on chained calls, `DataFrame.append`, `applymap`).
- **Dataset:** the shared Play Store data lives at `docs/assets/data/play_store_apps.csv` and `developers.csv`. Read `docs/datasets.md` for the columns and the deliberate mess. Load it in lessons with:
  ```python
  BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
  apps = pd.read_csv(BASE + "play_store_apps.csv")
  ```
  When running the code yourself, read the local copy instead.
- **Facts:** never invent facts, APIs or URLs. If you're unsure of an exact URL, link the site's top page. Android bridges must be accurate (for example, LiteRT input is NHWC `[1, H, W, 3]`, while PyTorch uses NCHW). List anything you're not 100% sure of in your report.

## Visuals

The reader wants rich illustrations and animations.

- **SVGs:** make 2–4 hand-written SVG diagrams, at least one animated with SMIL (`<animate>`, `<animateTransform>`, `<animateMotion>`).
    - Size and type: viewBox 520–760 wide, Helvetica/Arial.
    - Colours: light fills `#e8eaf6`, `#e0f2f1`, `#fff3e0`, `#fff59d`; strokes `#3949ab`, `#00897b`, `#ef6c00`, `#e53935`; captions `#555`.
    - Put `class="animated"` on an animated root `<svg>`.
- **Files:** save SVGs to `docs/assets/img/<phase>/dayNN-<name>.svg`. Embed them with:
  ```
  <figure class="diagram" markdown>
  ![Alt text describing what the picture shows](../../assets/img/phase-1/dayNN-name.svg)
  <figcaption>Animated: one-sentence caption.</figcaption>
  </figure>
  ```
- **Figures from data:** plot with matplotlib, save as SVG with a white background and a fixed font size, and embed like the SVGs.
- **Render-check:** check every image as a PNG (the Playwright helpers are in the scratchpad `shot/` folder), at two or more moments for animations. Fix overlapping text, clipped elements and misaligned arrows.
- **Mermaid:** quote every node label, and never start a label with "1.".
- **TeX:** none in headings or admonition titles; use plain text or Unicode there. In text, use MathJax: `\( ... \)` inline and `\[ ... \]` for display.
- **Widgets** (only if your lesson-map entry asks for one): a self-contained `docs/javascripts/<name>.js`, plus an optional CSS file. Follow `vector-playground.js`: mount on a class, use a `document$.subscribe` guard, draw SVG from JS, use no libraries. Test it in a browser with Playwright before and after an interaction.

## Rules

- Write only your own lesson file and its assets, plus any widget files your entry names. Don't edit `mkdocs.yml`, the lessons index, the plans, `extra.css` or other lessons. Don't run git, and don't message the user.
- Check the build with `python3 -m mkdocs build 2>&1 | grep -E "WARN|ERROR"`. Warnings about a page not being in the nav are fine; broken links or images from your file are not.

## Final report

Send the integrator:

1. the file paths you created;
2. a 3-line summary;
3. the **topic inventory** with each item's coverage;
4. the filled-in **checklist** from `lesson-checklist.md`, each item ticked, or explained if not;
5. every external URL you used;
6. everything you are not 100% sure is correct.
