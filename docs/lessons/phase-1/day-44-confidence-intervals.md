# Day 44: Confidence Intervals and the Bootstrap

**Phase 1 · Week 7 · ~120 minutes**

## Objectives

- [ ] Say what a 95% confidence interval promises, and spot the three common ways people misread it.
- [ ] Build a normal-approximation interval for a mean, \( \bar{x} \pm 1.96\,s/\sqrt{n} \), by hand and in NumPy, and explain where the 1.96 comes from.
- [ ] Show by simulation that about 95% of such intervals catch the true mean, and why small samples need a slightly bigger multiplier (Student's \( t \)).
- [ ] Run the bootstrap (resample with replacement) to get an interval for a median and for a difference in means, with no formula at all.
- [ ] Build an interval for a proportion, and know when the textbook formula breaks (rare events, zero counts) and what to use instead.
- [ ] Put error bars on a model's test-set accuracy, and use an interval to decide whether a staged rollout is safe to widen.

**Prerequisites.** [Day 43](day-43-sampling.md) (population vs sample, the sampling distribution, the standard error \( \sigma/\sqrt{n} \), dividing by \( n - 1 \)), [Day 40](day-40-joint-clt.md) (the central limit theorem), [Day 37](day-37-distributions.md) (the normal distribution, z-scores, the 68–95–99.7 rule), [Day 39](day-39-expectation-variance.md) (sample mean vs true mean), [Day 20](day-20-week3-review.md) (do apps with ads rate lower?) and [Day 19](day-19-matplotlib.md) (plotting).

---

## The big idea

You survey 40 apps and their average rating is 4.20. Is the average over *all* apps 4.20? Almost certainly not exactly: a different 40 apps would have given 4.15 or 4.26. A single number hides that wobble. A **confidence interval** puts the wobble back: "somewhere between 4.08 and 4.32".

**Analogy: a ring toss at a fair.** A peg is fixed in the ground: that is the true value you want (say, the average rating of every app). You can't see the peg; it's behind a curtain. You throw a ring, and wherever it lands is your interval. Two facts:

- The peg never moves. The *ring* is what's random: throw again (take a new sample) and it lands somewhere else.
- You can't tell whether *this* throw caught the peg. But you know your throwing arm: a "95% ring" is sized so that 95 out of 100 throws catch it.

That is exactly what "95% confidence" means: a statement about **the method**, not about one particular interval. Today you'll build that ring three ways:

| Method | Recipe | Use it for |
|---|---|---|
| **Normal approximation** | estimate \( \pm \) 1.96 standard errors | means and proportions with a decent sample |
| **Student's \( t \)** | estimate \( \pm \) a slightly bigger multiplier | means from small samples |
| **Bootstrap** | resample your own data thousands of times | anything else: medians, differences, percentiles, F1 scores |

**Why an ML engineer cares.** "Model B gets 93.5% accuracy, model A gets 92.0%." On 500 test images, each of those numbers is uncertain by about ±2.3 points, so that 1.5-point "win" may be noise. Every honest benchmark, A/B test and crash-rate dashboard needs an interval next to its number.

## A quick recap: the standard error

Day 43 built the key fact. If you take a random sample of \( n \) values from a population with mean \( \mu \) ("mu", the true mean) and standard deviation \( \sigma \) ("sigma", the true spread), then the sample mean \( \bar{x} \) ("x bar") is itself random, and:

- it is centred on \( \mu \) (on average, it's right);
- its standard deviation, called the **standard error** (SE), is \( \sigma / \sqrt{n} \);
- by the central limit theorem (Day 40), its distribution is close to a normal bell curve once \( n \) is a few dozen.

You almost never know \( \sigma \), so you plug in the sample standard deviation \( s \) (computed with \( n - 1 \), as Day 43 explained):

\[
\text{SE} \approx \frac{s}{\sqrt{n}}
\]

We'll use the same "population" as Days 39 and 43: the 462 rated apps in the Play Store dataset. Treating them as the whole world means we *know* the true mean, so we can check whether our intervals catch it (lesson tested on pandas 3.0):

```python
import numpy as np
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")      # or a local copy from the Datasets page
apps = apps.drop_duplicates(ignore_index=True)        # the 6 duplicate rows (Day 17)
apps["category"] = apps["category"].str.strip().str.title()   # "GAMES", "Games " -> "Games"

ratings = apps["rating"].dropna().to_numpy()   # our "population": every rated app
len(ratings)                                   # 462
mu = ratings.mean()                            # the true mean, normally unknown
sigma = ratings.std()                          # the true spread
mu, sigma                                      # (4.165584415584416, 0.4533731016292602)
```

(You can download the CSV from the [Datasets](../../datasets.md) page and read it from a local path instead.)

## The normal-approximation interval for a mean

### Where 1.96 comes from

Day 37's 68–95–99.7 rule said that 95% of a normal distribution lies within about 2 standard deviations of its centre. The precise number is **1.96**: exactly 95% of a normal curve lies between \( -1.96 \) and \( +1.96 \) standard deviations (the 2.5% left over in each tail makes 5%).

Apply that to the sampling distribution of \( \bar{x} \), whose centre is \( \mu \) and whose standard deviation is the SE:

\[
\text{in 95\% of samples:}\quad \mu - 1.96\,\text{SE} \;\le\; \bar{x} \;\le\; \mu + 1.96\,\text{SE}
\]

Now the one clever step. "\( \bar{x} \) is within 1.96 SE of \( \mu \)" is the same sentence as "\( \mu \) is within 1.96 SE of \( \bar{x} \)": distance is distance, whichever end you stand at. So in 95% of samples,

\[
\bar{x} - 1.96\,\text{SE} \;\le\; \mu \;\le\; \bar{x} + 1.96\,\text{SE}
\]

and that is the interval. We only flipped which number sits in the middle, but it changes everything: the left side needed \( \mu \), which we don't know; the right side needs only \( \bar{x} \) and the SE, which we can compute from the sample.

### The formula, one symbol at a time

\[
\bar{x} \;\pm\; z \cdot \frac{s}{\sqrt{n}}
\]

- \( \bar{x} \) is the sample mean: your best single guess, called the **point estimate**.
- \( s / \sqrt{n} \) is the estimated standard error: how much \( \bar{x} \) typically wobbles.
- \( z \) is the **multiplier** for the confidence level you want: 1.645 for 90%, 1.96 for 95%, 2.576 for 99%. Each is the number of standard deviations that captures that share of a normal curve.
- \( z \cdot s/\sqrt{n} \) is the **margin of error**: the half-width of the interval. "\( \pm \)" means "subtract it for the lower end, add it for the upper end".

**Worked example.** You time an app's cold start on 36 test runs: the mean is \( \bar{x} = 820 \) ms and the standard deviation is \( s = 120 \) ms.

1. Standard error: \( 120 / \sqrt{36} = 120 / 6 = 20 \) ms.
2. Margin of error: \( 1.96 \times 20 = 39.2 \) ms.
3. Interval: \( 820 - 39.2 = 780.8 \) to \( 820 + 39.2 = 859.2 \) ms.

So: "cold start is 820 ms, 95% CI [781, 859] ms". Want 99% instead? The margin becomes \( 2.576 \times 20 = 51.5 \) ms, giving [768.5, 871.5]: more confidence costs a wider ring.

```python
xbar_ms, s_ms, n_ms = 820, 120, 36
se_ms = s_ms / np.sqrt(n_ms)
se_ms                                          # 20.0
xbar_ms - 1.96 * se_ms, xbar_ms + 1.96 * se_ms # (780.8, 859.2)
```

Now on real data. Survey 40 random apps and build the interval from that sample alone, as if we didn't know \( \mu \):

```python
rng = np.random.default_rng(44)
sample = rng.choice(ratings, size=40)   # 40 apps, drawn with replacement (independent draws)
sample[:8]                              # array([3.8, 4.6, 3.6, 4.1, 4.3, 4.1, 3.4, 4.2])

xbar = sample.mean()                    # point estimate
s = sample.std(ddof=1)                  # sample sd, dividing by n - 1 (Day 43)
se = s / np.sqrt(len(sample))           # estimated standard error
xbar, s, se                             # (4.2025, 0.39124226870382095, 0.06186083430178436)
xbar - 1.96 * se, xbar + 1.96 * se      # (4.081252764768502, 4.323747235231497)
```

We drew with replacement so the 40 draws are independent, as if from a huge population; Day 43 discussed why sampling a sizeable chunk of a small population without replacement shrinks the wobble a little. Wrap it in a function for later:

```python
def ci_mean(x, z=1.96):
    """Normal-approximation confidence interval for the mean of x."""
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - z * se, x.mean() + z * se

lo, hi = ci_mean(sample)
lo, hi                     # (4.081252764768502, 4.323747235231497)
bool(lo <= mu <= hi)       # True    this ring caught the peg (4.1656)
ci_mean(sample, z=2.576)   # (4.043146490838603, 4.361853509161397)   99%: wider
```

!!! warning "Common mistake: \( \sigma \) of the data vs the standard error"
    The interval uses \( s/\sqrt{n} \), not \( s \). \( \bar{x} \pm 1.96\,s \) is a different thing: roughly the range where 95% of *individual apps* fall, [3.44, 4.97] here, much wider. The confidence interval is about the *average*, and it shrinks as \( n \) grows; the spread of individual values does not. Mix them up and you'll report error bars that are \( \sqrt{n} \) times too wide (or too narrow).

## What "95%" actually means: a simulation

The claim is about the method: *if you repeated the survey many times, about 95% of the intervals would contain \( \mu \)*. With a known population we can simply do that. First, check the ingredient: are 95% of sample means really within 1.96 SE of \( \mu \)?

```python
import matplotlib
matplotlib.use("Agg")            # draw into files, no window (leave this out in Jupyter)
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

samples = rng.choice(ratings, size=(10_000, 40))   # 10,000 surveys of 40 apps, one per row
means = samples.mean(axis=1)                       # 10,000 sample means
se_true = sigma / np.sqrt(40)                      # the true SE (we know sigma here)
se_true                                            # 0.07168458155017289
np.mean(np.abs(means - mu) <= 1.96 * se_true)      # 0.9456   about 95%, as promised

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.hist(means, bins=60, density=True, color="#c5cae9", edgecolor="white")
grid = np.linspace(mu - 4 * se_true, mu + 4 * se_true, 400)
bell = np.exp(-0.5 * ((grid - mu) / se_true) ** 2) / (se_true * np.sqrt(2 * np.pi))
ax.plot(grid, bell, color="#3949ab", lw=2, label="normal curve, sd = SE")
middle = np.abs(grid - mu) <= 1.96 * se_true
ax.fill_between(grid[middle], bell[middle], color="#fff59d", alpha=0.8, label=r"middle 95%: $\mu \pm 1.96\,$SE")
ax.axvline(mu, color="#e53935", ls="--", lw=1.5, label=rf"true mean $\mu$ = {mu:.3f}")
ax.set_xlabel("sample mean rating of 40 apps")
ax.set_ylabel("density")
ax.set_title("10,000 surveys: the sampling distribution of the mean")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("day44-sampling-distribution.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![A histogram of 10,000 sample means of 40 app ratings forms a bell shape centred on the dashed red true mean of 4.166; a blue normal curve with standard deviation equal to the standard error fits it closely, and the yellow-shaded middle 95% runs from about 4.025 to 4.306](../../assets/img/phase-1/day44-sampling-distribution.svg)
<figcaption>The sampling distribution of the mean (Day 43) is close to normal (Day 40), so 95% of sample means land within 1.96 standard errors of the truth.</figcaption>
</figure>

Now turn every one of those 10,000 samples into an interval, using only what that sample knows (its own \( \bar{x} \) and \( s \)), and count how many catch \( \mu \):

```python
ses = samples.std(axis=1, ddof=1) / np.sqrt(40)    # each survey's own estimated SE
los = means - 1.96 * ses
his = means + 1.96 * ses
covered = (los <= mu) & (mu <= his)                # True where the ring caught the peg
covered.mean()                                     # 0.9376
(~covered[:100]).sum()                             # 10    misses among the first 100
```

Here are 20 of those intervals (numbers 81 to 100), dropping in one at a time:

<figure class="diagram" markdown>
![Animation: a vertical dashed red line marks the true mean rating 4.166; twenty horizontal intervals drop in one at a time from top to bottom, each a blue bar with a dot at its sample mean; nineteen cross the red line and one, number 86, sits entirely to the left of it and turns red; a counter on the right goes up to 19 of 20 caught](../../assets/img/phase-1/day44-intervals-dropping.svg)
<figcaption>Animated: each survey of 40 apps throws its own ring. The peg (the true mean) never moves; the intervals do, and now and then one misses.</figcaption>
</figure>

And all of the first 100 at once:

```python
fig, ax = plt.subplots(figsize=(9, 6.2))
first = np.arange(100)
hit = covered[first]
ax.hlines(first[hit], los[first][hit], his[first][hit], color="#3949ab", lw=1.6, label="caught the true mean")
ax.hlines(first[~hit], los[first][~hit], his[first][~hit], color="#e53935", lw=2.4, label="missed")
ax.plot(means[first], first, "o", ms=2.5, color="#333")
ax.axvline(mu, color="#e53935", ls="--", lw=1.3)
ax.set_xlabel("mean rating (95% interval from each survey of 40 apps)")
ax.set_ylabel("survey number")
ax.set_ylim(100, -1)
ax.set_title(f"100 surveys, 100 intervals: {hit.sum()} caught the true mean {mu:.3f}")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
fig.tight_layout()
fig.savefig("day44-hundred-intervals.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![One hundred horizontal confidence intervals stacked from top to bottom, each with a black dot at its sample mean; a dashed red vertical line marks the true mean of 4.166; ninety blue intervals cross it and ten red intervals sit entirely to its left or right; the intervals also differ in width because each survey estimates its own spread](../../assets/img/phase-1/day44-hundred-intervals.svg)
<figcaption>Every interval is computed from one survey alone. Ninety of the first hundred catch the truth; across all 10,000 surveys, 93.8% do.</figcaption>
</figure>

Two things to notice. The intervals have **different widths**, because each survey estimates its own \( s \). And the overall coverage is 93.8%, a bit under the promised 95%. That gap is real, and it has a name.

### Small samples: Student's \( t \)

The 1.96 assumes we know the true SE. We don't: we estimate it with \( s \), and \( s \) wobbles too. Sometimes \( s \) comes out too small, the interval is too narrow, and it misses. With small \( n \) this happens a lot.

The fix was worked out in 1908 by William Gosset, a chemist at the Guinness brewery who published as "Student" (his employer didn't want competitors to know it was doing statistics). Replace 1.96 with a slightly bigger multiplier from **Student's \( t \) distribution** that depends on \( n - 1 \), the "degrees of freedom":

| sample size \( n \) | 5 | 10 | 20 | 40 | 100 | very large |
|---|---|---|---|---|---|---|
| 95% multiplier \( t \) | 2.776 | 2.262 | 2.093 | 2.023 | 1.984 | 1.960 |

(With SciPy installed on your machine, `scipy.stats.t.ppf(0.975, df=n - 1)` gives these numbers.) Let's measure the coverage of both multipliers at each sample size:

```python
t_975 = {5: 2.776, 10: 2.262, 20: 2.093, 40: 2.023, 100: 1.984}   # 95% t multipliers
for n, t in t_975.items():
    surveys = rng.choice(ratings, size=(10_000, n))
    m = surveys.mean(axis=1)
    se_n = surveys.std(axis=1, ddof=1) / np.sqrt(n)
    cover_z = np.mean((m - 1.96 * se_n <= mu) & (mu <= m + 1.96 * se_n))
    cover_t = np.mean((m - t * se_n <= mu) & (mu <= m + t * se_n))
    print(f"n={n:3d}   coverage with 1.96: {cover_z:.3f}   with t={t}: {cover_t:.3f}")
```

```text
n=  5   coverage with 1.96: 0.882   with t=2.776: 0.944
n= 10   coverage with 1.96: 0.916   with t=2.262: 0.945
n= 20   coverage with 1.96: 0.938   with t=2.093: 0.953
n= 40   coverage with 1.96: 0.940   with t=2.023: 0.948
n=100   coverage with 1.96: 0.945   with t=1.984: 0.949
```

With 5 apps the "95%" interval built with 1.96 is really an 88% interval; \( t \) brings it back to about 95%. By \( n = 100 \) the two multipliers barely differ. The rule of thumb: **use \( t \) for means** (it's never worse), and stop worrying about the difference above a hundred or so.

!!! warning "Common mistake: three misreadings of a 95% interval"
    Take the interval [4.08, 4.32] from our first survey.

    1. **"There's a 95% chance that \( \mu \) is between 4.08 and 4.32."** Not in this framework. \( \mu \) is a fixed number; this particular interval either contains it or doesn't (here it does, but in real life you'd never know). The 95% describes how often the *method* succeeds over many samples. (A Bayesian **credible interval**, built from a prior and a posterior as on Day 38, *does* mean "95% probability", at the price of choosing a prior.)
    2. **"95% of apps are rated between 4.08 and 4.32."** No: that range is for the *average*. Individual apps spread from about 3.4 to 5.0.
    3. **"If I survey 40 new apps, their mean will be in [4.08, 4.32] with 95% probability."** No: both the old interval and the new mean wobble, so the true chance is noticeably lower (around 83% on average for large samples).

## The bootstrap: intervals without a formula

The \( \bar{x} \pm 1.96\,\text{SE} \) recipe works for means because statisticians worked out the SE of a mean: \( \sigma/\sqrt{n} \). What is the SE of a **median**? Of a 90th-percentile startup time? Of an F1 score? Of the difference between two medians? There are formulas for a few of these, and none for most.

The **bootstrap** (Bradley Efron, 1979) sidesteps the algebra. The idea in one line: *you can't resample the population, so resample your sample.*

1. Treat your sample of \( n \) values as a stand-in for the population.
2. Draw \( n \) values **from your sample, with replacement** (some values get picked twice, some not at all). That's one **bootstrap resample**.
3. Compute your statistic (mean, median, anything) on the resample.
4. Repeat steps 2 and 3 thousands of times. The spread of those statistics imitates the sampling distribution.
5. The **percentile interval**: the middle 95% of the bootstrap statistics, from the 2.5th to the 97.5th percentile.

The name comes from "pulling yourself up by your bootstraps": the sample is used to judge its own reliability, which sounds impossible but works remarkably well.

**Worked example by hand.** A tiny sample of five cold-start times (in hundreds of ms): \( [2, 3, 5, 6, 9] \), mean 5.0. Three resamples, each five draws with replacement:

```python
tiny = np.array([2., 3., 5., 6., 9.])
for _ in range(3):
    resample = rng.choice(tiny, size=5, replace=True)   # 5 draws, repeats allowed
    print(resample, resample.mean())
# [2. 5. 6. 5. 9.] 5.4
# [6. 9. 3. 2. 5.] 5.0
# [5. 9. 6. 5. 9.] 6.8
```

In the first resample 5 appears twice and 3 never; the mean moves to 5.4. The second happens to use every value once. The third has two 9s and lands at 6.8. Do this 10,000 times and the means pile up into a histogram whose spread tells you how much the mean of five such numbers wobbles.

<figure class="diagram" markdown>
![Animation: five numbered balls 2, 3, 5, 6 and 9 sit in a bag labelled "your sample"; a highlight jumps between balls to draw five of them with replacement, copying each pick into a row, giving 2 5 6 5 9 with mean 5.4, then 6 9 3 2 5 with mean 5.0, then 5 9 6 5 9 with mean 6.8; each mean drops as a dot onto a number line below, and many more dots then pile up into a roughly bell-shaped histogram](../../assets/img/phase-1/day44-bootstrap-resample.svg)
<figcaption>Animated: each bootstrap resample draws n values from the sample itself, with replacement; the pile of resample means stands in for the sampling distribution.</figcaption>
</figure>

In NumPy, all the resamples are one call: a `(n_boot, n)` grid of random picks, then the statistic along each row.

```python
def bootstrap(x, stat, n_boot=10_000, rng=rng):
    """Apply stat(..., axis=1) to n_boot resamples of x drawn with replacement."""
    x = np.asarray(x, dtype=float)
    resamples = rng.choice(x, size=(n_boot, len(x)), replace=True)   # one resample per row
    return stat(resamples, axis=1)

boot_means = bootstrap(sample, np.mean)        # our 40-app survey again
boot_means[:5]                                 # array([4.3   , 4.1575, 4.095 , 4.165 , 4.285 ])
boot_means.std(), se                           # (0.06083848805649266, 0.06186083430178436)
np.percentile(boot_means, [2.5, 97.5])         # array([4.0825, 4.3225])
ci_mean(sample)                                # (4.081252764768502, 4.323747235231497)
```

Sanity check passed: for a mean, where we *have* a formula, the bootstrap's spread (0.0608) matches the formula's SE (0.0619), and the percentile interval [4.083, 4.323] matches \( \bar{x} \pm 1.96\,\text{SE} \) to two decimals. Now use it where there is no simple formula.

### A median: how well rated are games, really?

Ratings are bounded at 5 and a little lopsided, so the **median** (the middle value) is a natural summary. The cleaned data has 114 rated games:

```python
games = apps.loc[apps["category"] == "Games", "rating"].dropna().to_numpy()
len(games), np.median(games)                   # (114, 4.3)

boot_med = bootstrap(games, np.median)
np.percentile(boot_med, [2.5, 97.5])           # array([4.2, 4.4])
values, counts = np.unique(boot_med, return_counts=True)
values                                         # array([4.1 , 4.15, 4.2 , 4.25, 4.3 , 4.35, 4.4 , 4.45, 4.5 ])
counts                                         # array([  92,   58, 2388,  637, 4846,  472, 1495,    3,    9])
```

"The median game is rated 4.3, 95% CI [4.2, 4.4]." The counts show something the mean never does: the bootstrap medians are **lumpy**. Ratings come in steps of 0.1, and the median of an even number of values is either one of them or a halfway point, so only a handful of values ever appear. That's honest, not a bug, and it's why the interval's ends are round numbers.

```python
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(values, counts, width=0.035, color="#80cbc4", edgecolor="#00897b")
lo_g, hi_g = np.percentile(boot_med, [2.5, 97.5])
ax.axvspan(lo_g - 0.02, hi_g + 0.02, color="#fff59d", alpha=0.6, zorder=0, label=f"95% bootstrap interval [{lo_g:.1f}, {hi_g:.1f}]")
ax.axvline(np.median(games), color="#e53935", ls="--", lw=1.5, label=f"median of the sample = {np.median(games):.1f}")
ax.set_xticks(values)
ax.set_xlabel("median rating in a bootstrap resample of the 114 games")
ax.set_ylabel("resamples (of 10,000)")
ax.set_title("Bootstrap medians are lumpy: ratings come in steps of 0.1")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("day44-bootstrap-medians.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![A bar chart of how often each median rating appears across 10,000 bootstrap resamples of 114 games: only nine values occur, from 4.1 to 4.5; 4.3 is the tallest bar at 4,846, then 4.2 at 2,388 and 4.4 at 1,495, with small bars at the half steps; a yellow band covers 4.2 to 4.4 as the 95% interval and a dashed red line marks the sample median 4.3](../../assets/img/phase-1/day44-bootstrap-medians.svg)
<figcaption>The bootstrap distribution of the median rating of games. No formula needed: the middle 95% of the resample medians is the interval.</figcaption>
</figure>

The width depends heavily on how much data each category has. Health & Fitness has only 40 rated apps: its median is 4.25 with a bootstrap interval of about [3.85, 4.35], more than twice as wide. Before ranking categories by their medians (Day 19 did it with bars), check whether the intervals even separate them.

### A difference in means: ads vs no ads, with an interval

Day 20 asked whether free apps with ads are rated lower than free apps without, and found a gap of 0.031 stars that a shuffle test called unremarkable. An interval answers a more useful question: **how big could the real gap plausibly be?**

For a difference of two independent means, the variances add (Day 39), so the standard errors combine like the sides of a right triangle:

\[
(\bar{x}_A - \bar{x}_B) \;\pm\; 1.96\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}
\]

Here \( A \) is "free with ads" and \( B \) is "free without ads"; \( n_A, s_A \) are the size and standard deviation of group A, and the same for B. The bootstrap version: resample **each group separately** (keeping its size), take the difference of the two resample means, and repeat.

```python
free = apps[apps["price_usd"] == 0]     # paid apps never show ads, so compare free with free (Day 20)
ads = free.loc[free["has_ads"] == "Yes", "rating"].dropna().to_numpy()
no_ads = free.loc[free["has_ads"] == "No", "rating"].dropna().to_numpy()
len(ads), len(no_ads)                   # (291, 123)

diff = ads.mean() - no_ads.mean()
se_diff = np.sqrt(ads.var(ddof=1) / len(ads) + no_ads.var(ddof=1) / len(no_ads))
diff, se_diff                           # (0.03128824071745928, 0.04809752555120472)
diff - 1.96 * se_diff, diff + 1.96 * se_diff   # (-0.06298290936290198, 0.12555939079782052)

boot_diff = bootstrap(ads, np.mean) - bootstrap(no_ads, np.mean)   # resample each group on its own
np.percentile(boot_diff, [2.5, 97.5])   # array([-0.06238454,  0.12399757])
```

Formula and bootstrap agree: the gap is somewhere between about \( -0.06 \) and \( +0.13 \) stars. The interval **contains 0**, so the data are consistent with "no difference at all", which matches Day 20's shuffle test (Day 45 makes the link between intervals and tests exact). It also says something the test didn't: even the largest plausible gap is about an eighth of a star. If a product manager asks "do ads hurt ratings?", the honest answer is "not by more than about 0.06 stars in this data, and possibly not at all", plus Day 20's reminder that this is an observational comparison, not an experiment.

!!! warning "Common mistake: \"the intervals overlap, so there's no difference\""
    Comparing two separate intervals by eye is stricter than it looks. Suppose A has mean 10.0 and B has mean 11.5, each with SE 0.5. Their intervals, [9.02, 10.98] and [10.52, 12.48], overlap. But the difference is \( 1.5 \pm 1.96\sqrt{0.5^2 + 0.5^2} = 1.5 \pm 1.39 \), which is [0.11, 2.89] and excludes 0. Always build the interval **of the difference** (formula or bootstrap) instead of eyeballing two.

!!! warning "Common mistake: trusting the bootstrap where it can't work"
    The bootstrap can only reuse values that are in your sample. It struggles with **tiny samples** (five values have only 126 distinct resamples), with **extremes** like the maximum (no resample can exceed the sample's own max), and with **dependent data**: if one user contributes 50 sessions, resample *users*, not sessions, or the interval comes out far too narrow. And no resampling fixes a **biased sample** (Day 43): if only happy users leave ratings, the bootstrap gives you a precise interval around the wrong number.

## An interval for a proportion

A **proportion** is a share: the fraction of test images classified correctly, of sessions that crashed, of users who tapped "Buy". Each item is a 0/1 Bernoulli variable (Day 37), and a proportion is just the mean of those 0s and 1s. So the same recipe applies, with a ready-made standard deviation: a Bernoulli variable with success probability \( p \) has variance \( p(1 - p) \). With \( k \) successes out of \( n \) and \( \hat{p} = k/n \) ("p hat", the observed share):

\[
\hat{p} \;\pm\; 1.96\sqrt{\frac{\hat{p}\,(1 - \hat{p})}{n}}
\]

**Worked example.** A model classifies 460 of 500 test images correctly: \( \hat{p} = 0.92 \).

1. \( \hat{p}(1 - \hat{p}) = 0.92 \times 0.08 = 0.0736 \).
2. Divide by \( n \) and take the square root: \( \sqrt{0.0736 / 500} = \sqrt{0.0001472} \approx 0.01213 \).
3. Margin: \( 1.96 \times 0.01213 \approx 0.0238 \). Interval: [0.896, 0.944].

"92.0% accuracy, 95% CI [89.6%, 94.4%]": a 500-image test set can't tell 90% from 94%.

```python
def ci_prop(k, n, z=1.96):
    """Textbook ("Wald") interval for a proportion k / n."""
    p = k / n
    se = np.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se

ci_prop(460, 500)        # (0.8962201026074543, 0.9437798973925458)
ci_prop(0, 500)          # (0.0, 0.0)   zero crashes: "certainly zero"?! Nonsense.
```

### When the textbook formula breaks: rare events

Day 36 warned about "zero crashes in the beta, ship it!". The formula above says a build with 0 crashes in 500 sessions has a crash rate of exactly 0, with no uncertainty at all, because \( \hat{p}(1 - \hat{p}) = 0 \). That's absurd: a build crashing once every 1,000 sessions would show zero crashes in 500 sessions quite often (Day 37 computed chances like this).

The usual fix is the **Wilson interval** (Edwin Wilson, 1927). Its formula looks busier, but it is what you get by solving "\( p \) is within 1.96 standard errors of \( \hat{p} \)" using the SE of the *true* \( p \), \( \sqrt{p(1-p)/n} \), instead of plugging in \( \hat{p} \):

\[
\frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}
\;\pm\;
\frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}
\]

The first fraction is the new centre: it pulls \( \hat{p} \) slightly toward \( 1/2 \). The extra \( z^2/(4n^2) \) under the square root keeps the width above zero even when \( \hat{p} = 0 \). For large \( n \) and \( \hat{p} \) away from 0 and 1, both extra terms vanish and you're back to the textbook formula.

```python
def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k / n; well behaved even for k = 0 or k = n."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z / (1 + z**2 / n) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)

wilson(460, 500)         # (0.8928931229079627, 0.9407021975260784)   close to the textbook answer
wilson(0, 500)           # (0.0, 0.007624618530903363)   0 crashes: the rate could still be up to 0.76%
```

Is it actually better? Simulate a build whose true crash rate is 0.2% (one in 500 sessions), observed over 500 sessions, 100,000 times, and measure each method's coverage. `rng.binomial(n, p)` (Day 37) draws the crash count directly:

```python
true_p = 0.002
k = rng.binomial(500, true_p, size=100_000)          # crashes in each of 100,000 trial betas
lo_wald, hi_wald = ci_prop(k, 500)
lo_wil, hi_wil = wilson(k, 500)
np.mean((lo_wald <= true_p) & (true_p <= hi_wald))   # 0.63001   the textbook "95%" is really 63%
np.mean((lo_wil <= true_p) & (true_p <= hi_wil))     # 0.92038   much closer to 95%
```

The textbook interval fails badly here: about a third of the time (every time the beta shows 0 crashes) it claims certainty it doesn't have. Wilson isn't perfect either (with counts this small, coverage jumps around depending on the exact \( p \)), but it is the standard recommendation. A rough check before trusting the simple formula: you want at least about 10 successes **and** 10 failures.

## Where this shows up in ML

| Today's idea | Where you'll meet it |
|---|---|
| Interval for a proportion | Error bars on test accuracy, precision and recall; "is 92.0% vs 93.5% real?" needs \( n \) in the thousands |
| Margin \( \propto 1/\sqrt{n} \) | Sizing a test set: halving the error bar needs **four times** as many examples |
| Bootstrap | Intervals for metrics with no formula: F1, AUC, BLEU, median latency; resample the test examples and recompute the metric |
| Interval of a difference | Comparing two models on the same test set (better still, a *paired* bootstrap that resamples examples once and scores both models on them) |
| Small-sample \( t \) | Comparing training runs over 5 random seeds: a mean of 5 runs needs \( t = 2.776 \), not 1.96 |
| Dependent data | Resample whole users, sessions or documents, not individual rows, or the interval is too narrow (the same leakage worry as Day 17) |

How many test images do you need for a ±1 percentage-point margin at around 92% accuracy? Solve \( 1.96\sqrt{0.92 \times 0.08 / n} = 0.01 \):

\[
n = \frac{1.96^2 \times 0.92 \times 0.08}{0.01^2} = \frac{3.8416 \times 0.0736}{0.0001} \approx 2{,}828 \text{ images}
\]

```python
int(np.ceil(1.96**2 * 0.92 * 0.08 / 0.01**2))   # 2828
```

!!! android "Android bridge: a staged-rollout gate"
    A Play Console staged rollout ships a new build to a small percentage of users first, and you watch its crash rate (in Android vitals or Crashlytics) before widening. "0 crashes so far" is exactly the case where the textbook formula lies. A rollout gate that ramps up only when the **upper end** of the Wilson interval for the crash rate is below your budget makes the decision honest:

    ```kotlin
    import kotlin.math.sqrt

    /** 95% Wilson interval for a rate of [k] events in [n] tries. */
    fun wilson(k: Int, n: Int, z: Double = 1.96): ClosedFloatingPointRange<Double> {
        val p = k.toDouble() / n
        val z2 = z * z
        val centre = (p + z2 / (2 * n)) / (1 + z2 / n)
        val half = z / (1 + z2 / n) * sqrt(p * (1 - p) / n + z2 / (4.0 * n * n))
        return (centre - half).coerceAtLeast(0.0)..(centre + half).coerceAtMost(1.0)
    }

    fun safeToRamp(crashes: Int, sessions: Int, budget: Double = 0.005): Boolean =
        wilson(crashes, sessions).endInclusive < budget

    // safeToRamp(0, 500)  == false   upper end 0.0076: zero crashes isn't enough evidence yet
    // safeToRamp(0, 800)  == true    upper end 0.0048
    // safeToRamp(2, 1500) == true    upper end 0.0049
    ```

    The same habit applies to performance work. Jetpack Macrobenchmark reports the minimum, median and maximum of a metric such as `timeToInitialDisplayMs` over a handful of iterations; to say "the new build starts faster", bootstrap the median of each build's iterations and check that the interval of the *difference* excludes zero. With 10 iterations per build, expect wide intervals.

---

## Exercises

1. By hand: a sample of 64 network requests has mean latency 3.2 s and standard deviation 0.8 s. Compute the 95% and the 99% intervals for the mean latency. Check with NumPy.
2. The ratings population has \( \sigma \approx 0.453 \). How many apps must you survey for a 95% margin of error of ±0.05 stars? Then check by simulation: draw 10,000 surveys of that size and measure the typical margin and the coverage.
3. Repeat the coverage simulation for **90%** intervals (\( z = 1.645 \)) with surveys of 100 apps. What share of intervals catch \( \mu \)? Are 90% intervals wider or narrower than 95% ones, and why?
4. Build a 95% interval for the mean rating of **Tools** apps twice: with `ci_mean` and with the bootstrap percentile method. Then bootstrap the **median** of the same apps. Why is the median's interval so oddly placed relative to the sample median?
5. A model scores 870 of 1,000 test examples correctly. Give the textbook and Wilson 95% intervals. How many test examples would you need for a margin of ±1 point at this accuracy?
6. Zero crashes in \( n \) sessions: find the smallest \( n \) for which the Wilson upper bound is below 0.5%. Compare it with the "rule of three", which says the 95% upper bound after zero events in \( n \) tries is about \( 3/n \).

??? success "Solutions"
    ```python
    # 1: SE = 0.8 / sqrt(64) = 0.1;  95%: 3.2 ± 0.196 = [3.004, 3.396];  99%: 3.2 ± 0.2576 = [2.942, 3.458]
    se1 = 0.8 / np.sqrt(64)
    (3.2 - 1.96 * se1, 3.2 + 1.96 * se1), (3.2 - 2.576 * se1, 3.2 + 2.576 * se1)
    # ((3.004, 3.3960000000000004), (2.9424, 3.4576000000000002))
    ```

    ```python
    # 2: 1.96 * sigma / sqrt(n) = 0.05  =>  n = (1.96 * sigma / 0.05)^2
    n2 = int(np.ceil((1.96 * sigma / 0.05) ** 2))
    n2                                                    # 316
    sv2 = rng.choice(ratings, size=(10_000, n2))
    m2 = sv2.mean(axis=1)
    margin2 = 1.96 * sv2.std(axis=1, ddof=1) / np.sqrt(n2)
    margin2.mean(), np.mean(np.abs(m2 - mu) <= margin2)   # (0.04995810652060668, 0.9484)
    ```
    About 316 apps; the simulated margins average 0.050 and 94.8% of the intervals catch \( \mu \). (Our population only has 462 apps, so in real life you'd sample without replacement and do a little better, as Day 43 discussed.)

    ```python
    # 3
    sv3 = rng.choice(ratings, size=(10_000, 100))
    m3 = sv3.mean(axis=1)
    se3 = sv3.std(axis=1, ddof=1) / np.sqrt(100)
    np.mean((m3 - 1.645 * se3 <= mu) & (mu <= m3 + 1.645 * se3))   # 0.8965
    ```
    About 90%, as promised. 90% intervals are **narrower** (multiplier 1.645 instead of 1.96): a smaller ring catches the peg less often. Confidence and width always trade against each other.

    ```python
    # 4
    tools = apps.loc[apps["category"] == "Tools", "rating"].dropna().to_numpy()
    len(tools), tools.mean(), np.median(tools)           # (74, 4.151351351351352, 4.3)
    ci_mean(tools)                                       # (4.046888875324764, 4.255813827377939)
    np.percentile(bootstrap(tools, np.mean), [2.5, 97.5])     # array([4.04861486, 4.25543919])
    np.percentile(bootstrap(tools, np.median), [2.5, 97.5])   # array([4. , 4.3])
    ```
    The two mean intervals agree to two decimals. The median's interval is [4.0, 4.3], with the sample median 4.3 sitting at its **top** end. Exactly 36 of the 74 apps are rated below 4.3 and 11 sit at exactly 4.3, so the middle of the data lands right at the bottom of that pile. A resample with a few more low apps drops the median into the sparse 4.0 to 4.2 region, but pushing it above 4.3 would mean getting past all eleven 4.3s, which rarely happens. With lumpy data, bootstrap intervals for a median are lopsided and step-shaped; that's the data talking, and a reason to report the mean as well.

    ```python
    # 5: p = 0.87;  SE = sqrt(0.87 * 0.13 / 1000) = 0.01063;  margin 0.0208
    ci_prop(870, 1000)      # (0.849155697181244, 0.890844302818756)
    wilson(870, 1000)       # (0.8477315380448507, 0.8894365570012204)
    int(np.ceil(1.96**2 * 0.87 * 0.13 / 0.01**2))   # 4345
    ```
    With 130 errors and 870 successes (both far above 10) the two intervals nearly agree. For ±1 point you'd need about 4,345 examples, more than four times as many, because the margin shrinks only like \( 1/\sqrt{n} \).

    ```python
    # 6: for k = 0 the Wilson upper bound simplifies to z² / (n + z²)
    ns = np.arange(1, 2_000)
    upper = wilson(0, ns)[1]
    int(ns[np.argmax(upper < 0.005)])     # 765
    1.96**2 / (765 + 1.96**2), 3 / 765    # (0.004996607883860603, 0.00392156862745098)
    ```
    765 sessions with no crash before the Wilson upper bound drops below 0.5%. The rule of three gives a smaller bound (0.39%) because it is a **one-sided** bound: it puts all 5% of the doubt above the estimate, while the two-sided Wilson interval splits it, 2.5% each side. Both say the same thing in spirit: zero events in \( n \) tries only proves the rate is below a few divided by \( n \).

## Resources

- [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/): the "Frequentist Inference" chapter animates confidence intervals catching (and missing) the true mean, and has a bootstrap demo.
- [OpenIntro Statistics (free book)](https://www.openintro.org/book/os/): chapters on foundations for inference and inference for numerical data cover confidence intervals for means and proportions, and the \( t \) distribution, gently.
- [Khan Academy: Statistics and probability](https://www.khanacademy.org/math/statistics-probability): the "Confidence intervals" unit has the interpretation, margin of error and practice problems.
- [StatQuest: Bootstrapping Main Ideas!!!](https://www.youtube.com/watch?v=Xz0x-8-cgaQ): Josh Starmer's picture-first explanation of resampling with replacement.
- [Dive into Deep Learning: Probability and Statistics](https://d2l.ai/chapter_preliminaries/probability.html): the same estimation ideas written for deep learning, with code.
- [NumPy docs: `Generator.choice`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html) (the `replace` argument) and [`numpy.percentile`](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html).
- [Android developers: Macrobenchmark overview](https://developer.android.com/topic/performance/benchmarking/macrobenchmark-overview): how startup and frame metrics are measured over repeated iterations, the numbers you'd bootstrap.

## Checklist

- [ ] I can explain the ring-toss picture: the true value is fixed, the interval is random, and 95% describes the method.
- [ ] I can compute \( \bar{x} \pm 1.96\,s/\sqrt{n} \) by hand and say where 1.96 comes from.
- [ ] I simulated thousands of intervals and saw about 95% of them catch the true mean.
- [ ] I know to use Student's \( t \) multiplier for means from small samples.
- [ ] I can bootstrap an interval for any statistic, including a median and a difference in means, and I know its limits.
- [ ] I compare two groups with the interval of the difference, not by eyeballing two overlapping intervals.
- [ ] I can build an interval for a proportion and switch to Wilson for rare events or zero counts.
- [ ] I can put an error bar on a model's accuracy and estimate the test-set size a target margin needs.

**Next:** [Day 45: Hypothesis tests, p-values and A/B tests](day-45-hypothesis-tests.md)
