# Lesson checklist

This checklist is used twice for every lesson:

1. **By the writer**, before reporting back. Paste it into your final report, ticked, with a note on any item not met.
2. **By a separate reviewer agent**, who did not write the lesson. The reviewer reads the lesson as the target reader would (an Android engineer, new to Python and ML) and returns each failed item with the line number and a suggested fix. The writer fixes everything the reviewer finds. The lesson ships only when the reviewer passes it.

The integrator then runs the code and build checks at the bottom, and checks the reviewer's report.

## Coverage

- [ ] A topic inventory exists, built from the official docs' contents plus one textbook or course unit, with each item tagged must, should or later.
- [ ] Every **must** item is taught in full (the six-part pattern below).
- [ ] Every **should** item has at least one explained example with output.
- [ ] Every **later** item is listed under "Going further" with a link.
- [ ] Nothing the lesson map entry names is missing.
- [ ] Nothing a later lesson in the course map relies on is left for "later" without saying which lesson covers it.

## Depth: the six-part pattern, for each must concept

- [ ] **What**: a plain definition, with every new term defined where it first appears.
- [ ] **Why**: the problem it solves and when the reader will use it.
- [ ] **Bridge**: the Kotlin comparison for Python topics, or an intuition or picture before the formula for maths topics.
- [ ] **Example**: a minimal example, with real output for every line that produces a value.
- [ ] **Underneath**: the mental model, a trace, or a diagram of what actually happens.
- [ ] **Goes wrong**: a realistic mistake, shown with its real error message or wrong output, and the fix.

The reviewer also checks:

- [ ] No concept is introduced only in a table, a list item or a single sentence.
- [ ] No code uses a method, operator, argument or format spec that hasn't been explained yet.
- [ ] Tricky code is walked through step by step.
- [ ] Maths: symbols are introduced one at a time, and a worked example by hand comes before the code, which reproduces the hand result.
- [ ] No "obviously", "simply", "just" or "as you know".
- [ ] Reading as a beginner, the reviewer found no place where they'd ask "why?" or "what does that mean?" and get no answer. List any such place found.

## Structure

- [ ] The header line has the right phase, week and a realistic number of minutes.
- [ ] Every objective is taught in the lesson, and the checklist at the end mirrors the objectives.
- [ ] The prerequisites link earlier lessons and say what's used from each.
- [ ] There is at least one accurate Android bridge.
- [ ] Common-mistake boxes sit where the mistake happens, each with a real error.
- [ ] Exercises: 6–8, graded, including predict-the-output, find-the-bug and a challenge.
- [ ] **Every** exercise has a solution that runs on its own, shows real output, and explains why.
- [ ] There are 5–7 free resources whose URLs you are confident exist.
- [ ] The last line is the exact `**Next:**` line from the lesson map.
- [ ] The length fits the content (typically 600–900 lines for 90–120 minutes). Under 450 lines needs a reason.

## Visuals

- [ ] There are 2–4 diagrams, at least one animated, each explaining a concept rather than decorating.
- [ ] Every image was render-checked as a PNG, and animations were checked at two or more moments: no overlaps, no clipping.
- [ ] Every image has alt text describing what it shows, and a caption.
- [ ] Figures from data were made by the lesson's own code, so they match its numbers.
- [ ] There is no TeX in headings or admonition titles, and every Mermaid label is quoted.

## Code and build (integrator, after the reviewer passes)

- [ ] All blocks, solutions included, run in page order in one namespace with the runner script (`rt/test.py`): 0 failures. Blocks that can't run in the browser are labelled.
- [ ] Every printed value on the page matches that run, and none of them is approximate (timings excepted, and marked as machine-dependent).
- [ ] `mkdocs build --strict` passes on the lesson's own commit.
- [ ] The lesson is wired into the nav, the lessons index, the month plan, the progress page and the previous lesson's `Next:` line.
- [ ] Any widget is registered in `mkdocs.yml` and was tested in a browser.
