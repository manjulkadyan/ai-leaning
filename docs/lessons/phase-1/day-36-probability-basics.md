# Day 36: Probability and Random Variables, by Simulation

**Phase 1 · Week 6 · ~90 minutes**

## Objectives

- [ ] Use the words **outcome**, **sample space**, **event** and **probability** correctly, and write \( P(A) \).
- [ ] Explain probability as a **long-run frequency**, and watch it settle in a simulation.
- [ ] Simulate coins, dice and app crashes with `rng = np.random.default_rng(seed)`.
- [ ] Apply the complement, addition and multiplication rules, and check each one by counting and by simulation.
- [ ] Define a discrete **random variable** and its **probability mass function** (PMF), and show a simulated histogram converging to it.
- [ ] Solve the birthday problem twice: by simulation and exactly.
- [ ] Say where probability shows up when a model is trained or used.

**Prerequisites.** [Day 4](day-04-numpy-arrays.md) (NumPy arrays and boolean masks), [Day 9](day-09-broadcasting.md) (broadcasting), [Day 19](day-19-matplotlib.md) (histograms and bar charts) and, for one short section, [Day 17](day-17-pandas-cleaning.md) (loading the Play Store dataset).

---

## The big idea

Until now, every number in this course was certain: \( 2 \times 3 \) is always 6, and the same matrix times the same vector always gives the same answer. Machine learning lives in a world that isn't like that. Will this user tap the ad? Is this email spam? Will this session crash? We can't know for sure, but we can say how *likely* each answer is, and a lot of ML is exactly that: models that output "spam, with probability 0.93".

**Probability** is the number from 0 to 1 that measures "how likely". Today we'll define it in the most practical way possible:

> The probability of an event is the fraction of times it happens, if you repeat the experiment many, many times.

**Analogy.** You already read probabilities every week, on a dashboard. Crashlytics says your app is "99.8% crash-free". Nobody can tell you whether *the next* session will crash. But over a million sessions, about 998,000 won't, and that fraction is remarkably stable from week to week. That's a probability: unpredictable one at a time, predictable in bulk.

Because "repeat many times" is something a computer is very good at, today you'll learn probability the way working engineers check it: **simulate first, then confirm with a formula**. If the two disagree, one of them is wrong, and you'll know to look.

## Outcomes, events and probability

Four words, each with one exact meaning:

- An **experiment** is anything with an uncertain result: roll a die, flip a coin, run one app session.
- An **outcome** is one possible result: the die shows 4.
- The **sample space**, written \( \Omega \) (the Greek capital "omega"), is the set of *all* possible outcomes. For one die, \( \Omega = \{1, 2, 3, 4, 5, 6\} \).
- An **event** is a set of outcomes you care about, usually named with a capital letter. "The roll is even" is the event \( E = \{2, 4, 6\} \). An event "happens" if the outcome is in it.

\( P(E) \), read "the probability of E", is a number between 0 and 1. \( P = 0 \) means "never", \( P = 1 \) means "always", \( P = 0.5 \) means "half the time".

When every outcome is **equally likely** (a fair die, a fair coin), probability is just counting:

\[
P(E) = \frac{\text{number of outcomes in } E}{\text{number of outcomes in } \Omega}
\]

**Worked example.** For a fair die, \( P(\text{even}) = 3/6 = 0.5 \), and the event "5 or 6" has \( P = 2/6 \approx 0.333 \).

In NumPy, an event is a **boolean mask** over the sample space, and the mean of a mask is the fraction of `True`s (each `True` counts as 1, each `False` as 0):

```python
import numpy as np

omega = np.arange(1, 7)                  # the sample space of one die: [1 2 3 4 5 6]
even = omega % 2 == 0                    # the event "even", as a mask over the outcomes
big = omega >= 5                         # the event "5 or 6"
print(omega[even], even.mean())          # P(even) = 3/6
print(omega[big], big.mean())            # P(big)  = 2/6
```

```text
[2 4 6] 0.5
[5 6] 0.3333333333333333
```

Hold on to that trick, "**the mean of a mask is a probability**". It's the single most useful line in this lesson.

## Probability as long-run frequency

Counting works for fair dice. But what's the probability that a session of *your* app crashes? There's no list of equally likely outcomes to count. The answer is the frequency definition: repeat the experiment \( n \) times, count how often the event happened, and divide:

\[
\text{frequency after } n \text{ tries} = \frac{\text{number of times } E \text{ happened}}{n}
\;\longrightarrow\; P(E) \quad \text{as } n \text{ grows}
\]

The arrow means "gets closer and closer to". It doesn't get there in a straight line; it wobbles, and the wobble shrinks.

### Simulating with a random generator

NumPy's random numbers come from a **generator** object. Create one with a **seed**, a number that fixes the whole sequence it will produce, so that your results (and this page's) are reproducible:

```python
rng = np.random.default_rng(36)          # a seeded generator: the same "random" numbers every run
flips = rng.integers(0, 2, size=10)      # 0 = tails, 1 = heads (the upper bound 2 is excluded)
print(flips)
print(flips.mean())                      # the mean of 0s and 1s = the fraction of heads
```

```text
[0 0 0 0 0 1 1 0 1 1]
0.4
```

Four heads in ten flips. Nothing is wrong with the coin; ten flips is just not many. Let's flip 100,000 times and track the **running frequency**, the fraction of heads after each flip. `np.cumsum` (cumulative sum) gives the number of heads so far at every position, and dividing by `1, 2, 3, ...` turns those counts into fractions:

```python
rng = np.random.default_rng(7)
flips = rng.integers(0, 2, size=100_000)
n = np.arange(1, flips.size + 1)         # 1, 2, 3, ..., 100000
running = np.cumsum(flips) / n           # fraction of heads after each flip
for k in [10, 100, 1_000, 10_000, 100_000]:
    print(f"after {k:>7,} flips: {running[k - 1]:.4f}   off by {abs(running[k - 1] - 0.5):.4f}")
```

```text
after      10 flips: 0.7000   off by 0.2000
after     100 flips: 0.5600   off by 0.0600
after   1,000 flips: 0.5160   off by 0.0160
after  10,000 flips: 0.4926   off by 0.0074
after 100,000 flips: 0.5012   off by 0.0012
```

<figure class="diagram" markdown>
![Animation: the running fraction of heads from the real 100,000-flip simulation is drawn from left to right on a log-scale axis of the number of flips. It starts at 1 (the first seven flips were all heads), swings down past 0.5 and back, then hugs the 0.5 line; a yellow funnel 0.5 plus or minus 1 over root n narrows around it. A side panel reveals the frequency at 10, 100, 1,000, 10,000 and 100,000 flips: 0.700, 0.560, 0.516, 0.493 and 0.501](../../assets/img/phase-1/day36-running-frequency.svg)
<figcaption>Animated: the frequency wobbles a lot early on and settles at 0.5. Notice the x-axis is logarithmic: each step to the right is 10 times more flips.</figcaption>
</figure>

Two things to notice:

1. **It settles, but slowly.** The error shrinks roughly like \( 1/\sqrt{n} \): to get 10 times more accurate, you need 100 times more flips. That rule of thumb will come back all through Weeks 6 and 7 (it is the **law of large numbers**, Day 39, and the **standard error**, Day 43).
2. **It's never exactly 0.5.** After 100,000 flips it's 0.5012. A frequency is an *estimate* of a probability, with some noise in it.

### Dice

`rng.integers(low, high)` draws whole numbers from `low` up to **but not including** `high`, like Python's `range`. So a die is `rng.integers(1, 7)`:

```python
rng = np.random.default_rng(1)
rolls = rng.integers(1, 7, size=60_000)  # 60,000 rolls of a fair die; 7 is excluded
print(np.bincount(rolls)[1:])            # how many 1s, 2s, ..., 6s
print((rolls % 2 == 0).mean(), (rolls >= 5).mean())   # compare with 0.5 and 0.3333
```

```text
[10130  9963 10062  9984 10014  9847]
0.49656666666666666 0.3310166666666667
```

`np.bincount` counts how many times each whole number appears (index 0 counts the zeros, which a die never shows, so we drop it with `[1:]`). Each face comes up close to 10,000 times, and the simulated frequencies are close to the counted probabilities from the previous section.

!!! warning "Common mistake: `rng.integers(1, 6)` is a five-sided die"
    The upper bound is **excluded**: `rng.integers(1, 6)` never returns 6. It's the most common bug in dice simulations, and it's silent: nothing crashes, the numbers are just wrong. Check with `np.unique(rolls)` once. (The old `np.random.randint` has the same rule; Python's standard-library `random.randint(1, 6)` is the odd one out and *includes* 6.)

!!! note "Why `default_rng` and not `np.random.seed`"
    Old tutorials call `np.random.seed(0)` and then `np.random.rand()`. That uses one hidden global generator that any library can disturb. `rng = np.random.default_rng(seed)` gives you your own generator object that you pass around, and it's what NumPy's documentation recommends. In this lesson every random block creates its own seeded `rng`, so each one prints the same numbers no matter how many times you run it.

### App crashes

The same idea scales to anything that can be described as "happens with probability \( p \)". `rng.random(size)` draws numbers spread evenly between 0 and 1, so `rng.random(size) < p` is `True` a fraction \( p \) of the time. That one line simulates any yes/no event.

Say each session of your app crashes with probability \( p = 0.002 \) (0.2%):

```python
p_crash = 0.002                          # assume each session crashes with probability 0.2%
rng = np.random.default_rng(2)
crashed = rng.random(1_000_000) < p_crash          # one True/False per session
print(f"crash-free sessions: {1 - crashed.mean():.4%}")

# A small beta channel: only 500 sessions a week. Five different weeks:
for week in range(5):
    small = rng.random(500) < p_crash
    print(f"week {week}: crashes = {small.sum()}, crash-free {1 - small.mean():.1%}")
```

```text
crash-free sessions: 99.8044%
week 0: crashes = 0, crash-free 100.0%
week 1: crashes = 0, crash-free 100.0%
week 2: crashes = 2, crash-free 99.6%
week 3: crashes = 1, crash-free 99.8%
week 4: crashes = 1, crash-free 99.8%
```

A million sessions recover the true rate almost exactly. Five hundred sessions don't: the *same* app, with the *same* true crash probability, looks perfect one week and twice as bad as reality another.

!!! warning "Common mistake: trusting a frequency from a small sample"
    "Zero crashes in the beta, ship it!" With 500 sessions and a true crash rate of 0.2%, you'd expect only one crash a week, and zero is a common result (it happened in two of the five weeks above). A frequency from a few hundred tries can easily be off by a factor of two for rare events. Week 7 (confidence intervals, Day 44) turns "how many tries is enough?" into a number.

## The rules of probability

Three rules cover almost every calculation you'll do. Each one is easiest to see on a picture of every possible outcome. Roll **two** dice: there are \( 6 \times 6 = 36 \) equally likely outcomes, one cell per (first die, second die) pair.

<figure class="diagram" markdown>
![Animation in three steps on a 6 by 6 grid of two-dice outcomes, each cell showing the sum. Step 1 highlights the six diagonal cells where the sum is 7 and shows P equals 6 over 36. Step 2 highlights the bottom row (first die is 6) in orange and the right column (second die is 6) in teal, with the shared corner (6, 6) in red, and shows P(A or B) equals 6/36 plus 6/36 minus 1/36 equals 11/36. Step 3 highlights the 25 cells with no six in blue and shows 1 minus 25/36 equals 11/36, and 1/6 times 1/6 equals 1/36](../../assets/img/phase-1/day36-two-dice.svg)
<figcaption>Animated: an event is a set of cells, and its probability is its share of the 36. The three rules are three ways of counting cells.</figcaption>
</figure>

Before the rules, the two facts every probability obeys: \( 0 \le P(A) \le 1 \) for any event \( A \), and \( P(\Omega) = 1 \) (*something* in the sample space always happens).

### Rule 1: the complement

The **complement** of \( A \), written \( A^c \) ("A complement", also "not A"), is every outcome that is *not* in \( A \). Either \( A \) happens or it doesn't, so:

\[
P(A^c) = 1 - P(A)
\]

It sounds trivial, but it's the best trick in the book for any question that says "**at least one**". Counting the ways to get *at least one* six on two dice is fiddly; counting *no sixes at all* is easy: each die has 5 non-six faces, so \( 5 \times 5 = 25 \) cells. Then \( P(\text{at least one six}) = 1 - 25/36 = 11/36 \approx 0.306 \).

### Rule 2: addition ("or")

\( A \cup B \) (read "A or B", the symbol is *union*) is the event that \( A \) happens, or \( B \) happens, or both. \( A \cap B \) (read "A and B", *intersection*) is the event that both happen.

\[
P(A \cup B) = P(A) + P(B) - P(A \cap B)
\]

Why subtract? Adding \( P(A) + P(B) \) counts the outcomes that are in *both* events twice. In the picture, \( A \) = "first die is 6" is the bottom row (6 cells) and \( B \) = "second die is 6" is the right column (6 cells). They share one cell, (6, 6). So \( A \cup B \) has \( 6 + 6 - 1 = 11 \) cells: \( P = 11/36 \), the same answer the complement gave. Good: two routes, one answer.

If \( A \) and \( B \) can't both happen (they are **mutually exclusive**, like "the die shows 1" and "the die shows 2"), then \( P(A \cap B) = 0 \) and the rule shrinks to plain addition, \( P(A \cup B) = P(A) + P(B) \).

### Rule 3: multiplication for independent events ("and")

Two events are **independent** if knowing that one happened tells you nothing about the other. Two separate dice are independent: the first die has no way to influence the second. For independent events:

\[
P(A \cap B) = P(A) \times P(B)
\]

\( P(\text{both dice are 6}) = \tfrac{1}{6} \times \tfrac{1}{6} = \tfrac{1}{36} \), exactly one cell. And \( P(\text{no six on either}) = \tfrac{5}{6} \times \tfrac{5}{6} = \tfrac{25}{36} \), the 25 blue cells.

### All three rules, checked by counting

Build all 36 outcomes as two arrays (`np.meshgrid` pairs every value of one range with every value of another, and `ravel` flattens the grids), and let Python's `fractions.Fraction` keep the probabilities as exact fractions instead of long decimals:

```python
from fractions import Fraction

d1, d2 = np.meshgrid(np.arange(1, 7), np.arange(1, 7), indexing="ij")
d1, d2 = d1.ravel(), d2.ravel()          # all 36 (first, second) pairs, equally likely
print(len(d1), d1[:8], d2[:8])

def P(event):                            # equally likely outcomes: count / total, as an exact fraction
    return Fraction(int(event.sum()), event.size)

A = d1 == 6                              # event A: first die is a six
B = d2 == 6                              # event B: second die is a six
print(P(A), P(B))
print(P(A & B), P(A) * P(B))                     # multiplication rule (independent)
print(P(A | B), P(A) + P(B) - P(A & B))          # addition rule
print(1 - P(~A & ~B))                            # complement of "no six at all"
print(P(d1 + d2 == 7))                           # the sum-is-7 event from the picture
```

```text
36 [1 1 1 1 1 1 2 2] [1 2 3 4 5 6 1 2]
1/6 1/6
1/36 1/36
11/36 11/36
11/36
1/6
```

On masks, `&` is "and", `|` is "or" and `~` is "not" (the same operators as pandas filters on Day 16). Every rule matches the direct count. And the same answer by simulation, which is what you'll do when there's no neat list of outcomes to count:

```python
rng = np.random.default_rng(3)
two = rng.integers(1, 7, size=(200_000, 2))         # 200,000 throws of two dice
at_least_one_six = (two == 6).any(axis=1)
print(at_least_one_six.mean(), 11 / 36)
```

```text
0.30473 0.3055555555555556
```

### A real use: crash-free users vs crash-free sessions

Here's where the rules earn their keep. Your sessions are 99.8% crash-free. A typical user opens the app 20 times a week. What fraction of **users** get through the week without a single crash?

If sessions crash independently, "no crash in 20 sessions" is 20 independent "no crash" events, so multiply: \( (1 - 0.002)^{20} \). Then the complement gives "at least one crash":

\[
P(\text{user sees at least one crash}) = 1 - (1 - 0.002)^{20} = 1 - 0.998^{20} \approx 1 - 0.9608 = 0.0392
\]

```python
p_crash = 0.002
sessions = 20                            # sessions per user per week
exact = 1 - (1 - p_crash) ** sessions    # P(at least one crash) = 1 - P(no crash in all 20)
rng = np.random.default_rng(4)
users = rng.random((100_000, sessions)) < p_crash    # 100,000 users x 20 sessions
simulated = users.any(axis=1).mean()
print(f"crash-free sessions: {1 - p_crash:.2%}")
print(f"crash-free users:    {1 - exact:.2%} exact, {1 - simulated:.2%} simulated")
```

```text
crash-free sessions: 99.80%
crash-free users:    96.08% exact, 96.07% simulated
```

A crash rate that sounds tiny per session hits almost **1 user in 25** every week. Small probabilities compound over many tries, and "at least one" questions are always \( 1 - (1 - p)^n \).

!!! warning "Common mistake: multiplying events that aren't independent"
    The multiplication rule needs independence, and real events often aren't independent. If a user's sessions crash because their *device* has a bad GPU driver, a crash in session 1 makes a crash in session 2 *more* likely, and \( 0.998^{20} \) is wrong (here the true crash-free-users number would be higher, because the crashes pile up on the same unlucky users). Another classic: adding \( P(A) + P(B) \) without subtracting the overlap. A probability bigger than 1 is the giveaway; a quietly wrong 0.8 isn't. Day 38 (conditional probability) gives you the tools to handle dependence properly.

## Probabilities from data

So far the probabilities came from a model of the world ("the die is fair", "each session crashes with probability 0.002"). Often you only have data. Then the frequency *in the data* is your estimate. Pick a random app from the Play Store dataset: how likely is it to be rated 4.5 or higher? (Load it as on Day 17; [download the CSV from the Datasets page](../../datasets.md) to work offline and replace `BASE + ...` with your local path. This lesson was tested with NumPy 2.5 and pandas 3.0.)

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv").drop_duplicates()
rated = apps.dropna(subset=["rating"])                 # 462 rated apps

ads = (rated["has_ads"] == "Yes").to_numpy()
top = (rated["rating"] >= 4.5).to_numpy()
print(len(rated), ads.mean().round(4), top.mean().round(4))

rng = np.random.default_rng(6)
picks = rng.integers(0, len(rated), size=50_000)       # pick an app at random, 50,000 times
print(top[picks].mean().round(4))                      # long-run frequency of "rated 4.5+"

print((ads & top).mean().round(4), (ads.mean() * top.mean()).round(4))
```

```text
462 0.6299 0.2879
0.2876
0.1797 0.1813
```

- 63% of rated apps show ads, and 28.8% are rated 4.5 or more. Picking apps at random 50,000 times gives a running frequency of 0.2876, settling on the dataset's fraction, exactly like the coin settled on 0.5.
- The last line is a first peek at **independence in data**: if "has ads" and "rated 4.5+" were independent, \( P(\text{both}) \) would equal the product. It's 0.1797 against 0.1813: close, but is that difference real or noise? Day 38 builds the 2×2 table to look closer, and Week 7's hypothesis tests give a proper answer.

## Random variables and their PMF

Often we don't care about the outcome itself but about a **number computed from it**: the *sum* of two dice, the *number* of crashes in a week, the *predicted label* of a model. That number is a **random variable**.

A random variable, usually written with a capital letter like \( X \), is a rule that assigns a number to every outcome. For two dice, \( X = \) first die \( + \) second die: the outcome (2, 5) gives \( X = 7 \), the outcome (6, 6) gives \( X = 12 \). "Random" because the outcome is random; "variable" is a historical name, it's really a function. When its possible values can be listed (2, 3, ..., 12), it's a **discrete** random variable. (Continuous ones, like a height or a model's raw score, come on Day 37.)

The **probability mass function** (PMF) lists the probability of each value:

\[
p(k) = P(X = k)
\]

read "the probability that \( X \) equals \( k \)". **Worked example:** \( X = 7 \) happens on 6 of the 36 cells (the yellow diagonal: 1+6, 2+5, 3+4, 4+3, 5+2, 6+1), so \( p(7) = 6/36 \). \( X = 2 \) happens only for (1, 1): \( p(2) = 1/36 \).

Two facts hold for every PMF: each \( p(k) \) is between 0 and 1, and they add up to 1, because \( X \) always takes *some* value: \( \sum_k p(k) = 1 \).

Compute the whole PMF exactly from the 36 outcomes. `np.bincount` counts how many cells give each sum:

```python
sums = d1 + d2                           # X(outcome) for all 36 outcomes
pmf = np.bincount(sums, minlength=13) / 36           # index k holds P(X = k)
for k in range(2, 13):
    print(f"P(X = {k:>2}) = {pmf[k]:.4f}  {'#' * int(round(pmf[k] * 36))}")
print("sum of all probabilities:", pmf.sum())
```

```text
P(X =  2) = 0.0278  #
P(X =  3) = 0.0556  ##
P(X =  4) = 0.0833  ###
P(X =  5) = 0.1111  ####
P(X =  6) = 0.1389  #####
P(X =  7) = 0.1667  ######
P(X =  8) = 0.1389  #####
P(X =  9) = 0.1111  ####
P(X = 10) = 0.0833  ###
P(X = 11) = 0.0556  ##
P(X = 12) = 0.0278  #
sum of all probabilities: 1.0000000000000002
```

The text bars make the triangle shape visible: 7 is the most likely sum because the most cells add up to it. (The final `...0002` is floating-point rounding of eleven fractions; the exact total is 1.)

### The histogram converges to the PMF

Now simulate the random variable and compare. A histogram of simulated values, divided by the number of throws, is an *estimated* PMF. With more throws, it should approach the exact one:

```python
rng = np.random.default_rng(5)
for throws in [100, 1_000, 100_000]:
    x = rng.integers(1, 7, size=(throws, 2)).sum(axis=1)   # the random variable, sampled
    est = np.bincount(x, minlength=13) / throws
    print(f"{throws:>7,} throws: P(X=7) = {est[7]:.4f}, worst error over all k = {np.abs(est - pmf).max():.4f}")
```

```text
    100 throws: P(X=7) = 0.2300, worst error over all k = 0.0633
  1,000 throws: P(X=7) = 0.1640, worst error over all k = 0.0119
100,000 throws: P(X=7) = 0.1678, worst error over all k = 0.0019
```

The same simulation as a picture (same seed, so the same numbers):

```python
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})

rng = np.random.default_rng(5)
ks = np.arange(2, 13)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, throws in zip(axes, [100, 1_000, 100_000]):
    x = rng.integers(1, 7, size=(throws, 2)).sum(axis=1)
    est = np.bincount(x, minlength=13) / throws
    ax.bar(ks, est[ks], color="#9fa8da", edgecolor="#3949ab", label="simulated")
    ax.plot(ks, pmf[ks], "o", color="#e53935", label="exact (k/36)")
    ax.set_title(f"{throws:,} throws")
    ax.set_xticks(ks)
    ax.set_xlabel("sum of two dice")
axes[0].set_ylabel("probability")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("day36-pmf-converge.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Three bar charts of the simulated distribution of the sum of two dice, with red dots for the exact probabilities k over 36. With 100 throws the bars are ragged (sum 7 at 0.23, sum 4 at only 0.02); with 1,000 throws they roughly follow the dots; with 100,000 throws every bar top sits on its dot, forming a clean triangle peaking at 7](../../assets/img/phase-1/day36-pmf-converge.svg)
<figcaption>A real simulation: as the number of throws grows, the histogram (bars) converges to the PMF (dots).</figcaption>
</figure>

With 100 throws, sum 7 looks like it has probability 0.23 and sum 4 looks rare. With 100,000 throws, every bar lands on its dot. This is the frequency definition again, now for every value of \( X \) at once.

## The birthday problem

A famous puzzle that shows why simulation beats intuition. **How many people must be in a room before there's a 50% chance that at least two of them share a birthday?** (Ignore 29 February, and assume all 365 birthdays are equally likely.)

Most people guess around 180. The answer is **23**.

### By simulation

Simulate many "parties". Each row of a `(trials, people)` array is one party, and each entry a birthday from 0 to 364. To spot a shared birthday, sort each row: equal birthdays end up next to each other, so a zero difference between neighbours means a match.

```python
def birthday_sim(people, trials=100_000, seed=0):
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 365, size=(trials, people))     # one row per party
    days.sort(axis=1)                                      # equal birthdays become neighbours
    shared = (np.diff(days, axis=1) == 0).any(axis=1)      # any two neighbours equal?
    return shared.mean()

def birthday_exact(people):
    k = np.arange(people)                                  # 0, 1, ..., people-1
    p_all_different = np.prod((365 - k) / 365)
    return 1 - p_all_different

for people in [5, 10, 20, 23, 30, 50, 70]:
    print(f"{people:>2} people: simulated {birthday_sim(people):.4f}   exact {birthday_exact(people):.4f}")
```

```text
 5 people: simulated 0.0270   exact 0.0271
10 people: simulated 0.1157   exact 0.1169
20 people: simulated 0.4119   exact 0.4114
23 people: simulated 0.5070   exact 0.5073
30 people: simulated 0.7079   exact 0.7063
50 people: simulated 0.9697   exact 0.9704
70 people: simulated 0.9992   exact 0.9992
```

### Exactly, with the complement and multiplication rules

"At least two share" is an "at least one" question, so use the complement: compute \( P(\text{all birthdays different}) \) and subtract it from 1. Line the people up one by one:

- Person 1 can have any birthday: \( 365/365 \).
- Person 2 must avoid person 1's day: \( 364/365 \).
- Person 3 must avoid two taken days: \( 363/365 \).
- ...and person \( n \) must avoid \( n - 1 \) days: \( (365 - n + 1)/365 \).

Each person's birthday is independent of the others', so multiply:

\[
P(\text{all different}) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \times \dots \times \frac{365 - n + 1}{365}
\qquad
P(\text{a shared birthday}) = 1 - P(\text{all different})
\]

**Worked example** with \( n = 3 \): \( P(\text{all different}) = 1 \times \tfrac{364}{365} \times \tfrac{363}{365} = 0.9918 \), so \( P(\text{shared}) = 0.0082 \). That is what `birthday_exact` computes: `(365 - k) / 365` for `k = 0, 1, ..., n-1` gives the factors, and `np.prod` multiplies them. For 23 people the product drops just below one half, and the simulation (0.5070) agrees with the formula (0.5073) to within its noise.

```python
people = np.arange(1, 81)
exact = [birthday_exact(n) for n in people]
sim_n = np.arange(5, 81, 5)
sim = [birthday_sim(n, trials=20_000, seed=int(n)) for n in sim_n]

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(people, exact, color="#3949ab", lw=2, label="exact: 1 − (365/365)(364/365)…")
ax.plot(sim_n, sim, "o", color="#ef6c00", label="simulated (20,000 parties each)")
ax.axhline(0.5, color="#999", lw=1, ls="--")
ax.axvline(23, color="#999", lw=1, ls="--")
ax.annotate("23 people: 50.7%", xy=(23, birthday_exact(23)), xytext=(32, 0.32),
            arrowprops={"arrowstyle": "->", "color": "#555"}, color="#333")
ax.set_xlabel("number of people in the room")
ax.set_ylabel("P(at least two share a birthday)")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("day36-birthday.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![An S-shaped curve of the probability that at least two people share a birthday against the number of people from 1 to 80. Orange simulated points lie on the blue exact curve. Dashed lines mark 50% and 23 people, where the curve crosses one half; by 50 people it is near 0.97 and by 70 almost 1](../../assets/img/phase-1/day36-birthday.svg)
<figcaption>Simulation (dots) and the exact formula (line) agree: 23 people give a 50.7% chance of a shared birthday, and 50 people about 97%.</figcaption>
</figure>

**Why intuition fails.** You imagine *your own* birthday being matched, which really does need about 253 other people (Exercise 6). But the question is about *any pair*, and 23 people form \( 23 \times 22 / 2 = 253 \) pairs. Lots of pairs, each with a small chance, add up fast.

**Why an engineer should care.** Swap birthdays for **hash values** and you get the *birthday bound*: hash \( n \) items into \( N \) buckets and collisions become likely once \( n \) is around \( \sqrt{N} \), far sooner than \( N \). Random 32-bit IDs (about 4.3 billion values) are likely to collide after only about 77,000 items, which is why UUIDs use 122 random bits.

## Where this shows up in ML

Probability is not a side topic in ML; it's the language models speak:

- **Model outputs are PMFs.** A classifier's softmax layer outputs one probability per class, and they add up to 1: exactly a PMF over the labels. Cross-entropy loss (Day 54) scores how much probability the model put on the right answer.
- **Generating text is sampling from a PMF.** An LLM outputs a PMF over its vocabulary for the next token, and the app draws one token from it, just like rolling a weighted die.
- **Training uses randomness on purpose.** Random weight initialization (Day 37), shuffling the data every epoch and drawing mini-batches (Day 52), random train/test splits, data augmentation, and **dropout**, which switches each unit off with probability \( p \) during training.

Both of the last two are one line each with what you learned today:

```python
rng = np.random.default_rng(8)
words = np.array(["cat", "dog", "bird"])
probs = [0.7, 0.2, 0.1]                                # a model's output: a PMF over 3 labels
samples = rng.choice(words, size=10_000, p=probs)      # how an LLM "samples" its next token
values, counts = np.unique(samples, return_counts=True)
print(dict(zip(values.tolist(), (counts / counts.sum()).round(3).tolist())))

keep = rng.random((2, 8)) >= 0.25                      # dropout with p = 0.25: each unit dropped with prob 0.25
print(keep.astype(int))
```

```text
{'bird': 0.097, 'cat': 0.698, 'dog': 0.204}
[[0 1 1 0 0 0 1 0]
 [1 1 1 1 1 1 1 0]]
```

`rng.choice(..., p=...)` draws from any PMF you give it, and the frequencies come back close to 0.7, 0.2, 0.1. The dropout mask keeps each unit with probability 0.75; any single row can look lopsided (the first kept only 3 of 8), but over thousands of units the fraction kept is very close to 75%. And that's why seeds matter in ML: fix the seed and the "random" training run becomes repeatable, which is how you tell a real improvement from luck.

!!! android "Android bridge: crash-free rates and staged rollouts are probabilities"
    **Crash-free statistics.** Firebase Crashlytics reports what percentage of your users (and sessions) were crash-free over a period. That's today's long-run frequency, and the 99.8% vs 96.1% calculation above is why the two numbers differ so much: a user who opens the app often has many chances to hit a rare crash. Read a crash-free percentage from a few hundred sessions with suspicion; it's a noisy estimate.

    **Staged rollouts.** When you release to "5% of users" in a Play Console staged rollout, or gate a feature on a percentage with Remote Config, each user is effectively assigned a probability of 0.05 of being in. You do the same thing in your own feature flags. Two details matter: the decision must be **stable** for a user (the same answer on every launch), and it must be **unrelated** to anything else about them (so the 5% is a fair sample). Hashing the user ID into 100 buckets gives both:

    ```kotlin
    // Is this user in the rollout? Stable across launches; ~percent% of all users say yes.
    fun inRollout(userId: String, featureKey: String, percent: Int): Boolean {
        val bucket = Math.floorMod("$featureKey:$userId".hashCode(), 100)   // 0..99
        return bucket < percent
    }
    ```

    `String.hashCode()` is fully specified on the JVM, so the bucket is the same on every device and every launch. `Math.floorMod` keeps the bucket in 0..99 even when the hash is negative (`abs(hashCode()) % 100` breaks for `Int.MIN_VALUE`, whose absolute value overflows). Including `featureKey` gives each feature an independent 5%, so the same users aren't always the guinea pigs. And for genuinely random draws in Kotlin, `kotlin.random.Random(seed)` plays the role of `default_rng(seed)`: seed it in tests to make "random" behaviour reproducible.

---

## Exercises

1. By counting cells, find \( P(\text{sum of two dice} \ge 10) \). Check it with the `P` function from the lesson and with a 100,000-throw simulation (`rng = np.random.default_rng(10)`).
2. The Chevalier de Méré's 17th-century puzzle, which helped start probability theory: which is more likely, **at least one six in 4 rolls** of one die, or **at least one double six in 24 throws** of two dice? Answer exactly with the complement and multiplication rules, then simulate both (`rng = np.random.default_rng(11)`, 100,000 repeats each).
3. With a per-session crash probability of 0.002, what fraction of users are crash-free in a week if they open the app 50 times? Then work backwards: what per-session crash probability would you need for **99%** of 20-session users to be crash-free?
4. A colleague writes \( P(\text{even or} \ge 5) = 1/2 + 1/3 = 5/6 \) for one die. Find the mistake, compute the right answer with masks, and confirm with the addition rule.
5. Let \( X \) be the **larger** of two dice (the maximum). Compute its exact PMF from the 36 outcomes, spot the pattern, and compare with a simulation (`rng = np.random.default_rng(12)`, 100,000 throws).
6. Using `birthday_exact`, find the smallest group with at least a 90% chance of a shared birthday. Then find how many *other* people you need for a 50% chance that someone shares **your** birthday (each person independently matches you with probability 1/365). Compare with the number of pairs among 23 people.

??? success "Solutions"
    ```python
    # 1: the cells (4,6), (5,5), (6,4), (5,6), (6,5), (6,6) -> 6/36 = 1/6
    print(P(d1 + d2 >= 10))                                   # 6/36 -> printed as 1/6
    rng = np.random.default_rng(10)
    x = rng.integers(1, 7, size=(100_000, 2)).sum(axis=1)
    print((x >= 10).mean())
    # 1/6
    # 0.16556

    # 2
    print(1 - (5 / 6) ** 4, 1 - (35 / 36) ** 24)
    rng = np.random.default_rng(11)
    four_rolls = rng.integers(1, 7, size=(100_000, 4))
    print((four_rolls == 6).any(axis=1).mean())
    pairs = rng.integers(1, 7, size=(100_000, 24, 2))         # 24 throws of two dice, 100,000 times
    double_six = (pairs == 6).all(axis=2)                     # (100000, 24): was this throw a double six?
    print(double_six.any(axis=1).mean())
    # 0.5177469135802468 0.4914038761309034
    # 0.51847
    # 0.49289
    ```
    One six in four rolls (51.8%) is the better bet; a double six in 24 throws (49.1%) is just under even. A double six has probability \( 1/36 \) per throw, so "no double six in 24" is \( (35/36)^{24} \). De Méré lost money on the second bet even though a naive "4/6 vs 24/36, both 2/3" argument says they're equal: that argument adds probabilities of overlapping events, the mistake from Rule 2.

    ```python
    # 3
    print(f"{(1 - 0.002) ** 50:.2%}")                         # crash-free users at 50 sessions a week
    p_needed = 1 - 0.99 ** (1 / 20)
    print(f"{p_needed:.6f}", f"{(1 - p_needed) ** 20:.4f}")   # per-session target, and a check
    # 90.47%
    # 0.000502 0.9900

    # 4
    omega = np.arange(1, 7)
    even, big = omega % 2 == 0, omega >= 5
    print(omega[even | big], (even | big).mean())             # [2 4 5 6] -> 4/6
    print(even.mean() + big.mean() - (even & big).mean())     # the addition rule agrees
    # [2 4 5 6] 0.6666666666666666
    # 0.6666666666666666
    ```
    For 3: heavy users (50 sessions) are only 90.5% crash-free. To get 99% crash-free users at 20 sessions you need \( (1 - p)^{20} = 0.99 \), so \( p = 1 - 0.99^{1/20} \approx 0.0005 \): a session crash rate four times lower than 0.002, i.e. 99.95% crash-free sessions. For 4: the outcome 6 is both even and \( \ge 5 \), so it was counted twice. The right answer is \( 1/2 + 1/3 - 1/6 = 4/6 = 2/3 \).

    ```python
    # 5
    mx = np.maximum(d1, d2)                                   # X = the larger of the two dice
    pmf_max = np.bincount(mx, minlength=7)[1:] / 36           # P(X = 1), ..., P(X = 6)
    print((pmf_max * 36).astype(int))                         # [ 1  3  5  7  9 11] = 2k - 1
    rng = np.random.default_rng(12)
    sim = rng.integers(1, 7, size=(100_000, 2)).max(axis=1)
    print((np.bincount(sim, minlength=7)[1:] / 100_000).round(4))
    print(pmf_max.round(4))
    # [ 1  3  5  7  9 11]
    # [0.0268 0.0842 0.1409 0.1945 0.2502 0.3033]
    # [0.0278 0.0833 0.1389 0.1944 0.25   0.3056]
    ```
    \( P(X = k) = (2k - 1)/36 \). In the grid, the cells with maximum exactly \( k \) form an "L": \( k \) cells along row \( k \) plus \( k \) along column \( k \), sharing the corner \( (k, k) \), so \( k + k - 1 = 2k - 1 \). That's the addition rule again. The odd numbers 1 + 3 + 5 + 7 + 9 + 11 add up to 36, as a PMF must.

    ```python
    # 6
    n = 1
    while birthday_exact(n) < 0.9:
        n += 1
    print(n, round(birthday_exact(n), 4))                     # smallest group with P >= 0.9

    m = 1
    while 1 - (364 / 365) ** m < 0.5:
        m += 1
    print(m, round(1 - (364 / 365) ** m, 4))                  # others needed to share YOUR birthday
    print(23 * 22 // 2)                                       # pairs among 23 people
    # 41 0.9032
    # 253 0.5005
    # 253
    ```
    41 people give a 90% chance. To match *your* birthday you need 253 other people, because each comparison involves you. 23 people already make 253 *pairs*, and every pair is a chance at a match: that's the whole trick of the birthday problem.

## Resources

- [Seeing Theory: Basic Probability](https://seeing-theory.brown.edu/): Brown University's interactive visual introduction; its first chapter animates exactly the coin-frequency idea from this lesson.
- [Khan Academy: Probability](https://www.khanacademy.org/math/statistics-probability/probability-library): short videos and practice on events, complements, the addition rule and independence.
- [NumPy docs: Random Generator](https://numpy.org/doc/stable/reference/random/generator.html): `default_rng`, `integers`, `random`, `choice` and the other methods used today, and why the `Generator` API replaced `np.random.seed`.
- [Harvard Stat 110 (Joe Blitzstein)](https://projects.iq.harvard.edu/stat110): a complete free probability course with lecture videos; the birthday problem is in the first lectures.
- [Dive into Deep Learning: Probability and Statistics](https://d2l.ai/chapter_preliminaries/probability.html): the same ideas from an ML angle, with sampling code.
- [Mathematics for Machine Learning (free book), ch. 6 "Probability and Distributions"](https://mml-book.github.io/): the formal version, for when you want definitions with all the details.
- [Firebase Crashlytics documentation](https://firebase.google.com/docs/crashlytics): where the crash-free statistics in the Android bridge come from.

## Checklist

- [ ] I can define outcome, sample space, event and probability, and compute \( P(E) \) by counting equally likely outcomes.
- [ ] I can explain probability as a long-run frequency, and I know the error shrinks roughly like \( 1/\sqrt{n} \).
- [ ] I simulate yes/no events with `rng.random(size) < p` and dice with `rng.integers(1, 7)`, and I remember the upper bound is excluded.
- [ ] I can use the complement rule for "at least one" questions, and compute \( 1 - (1 - p)^n \).
- [ ] I subtract the overlap in the addition rule, and I only multiply probabilities for independent events.
- [ ] I can define a discrete random variable and its PMF, and show a simulated histogram converging to it.
- [ ] I solved the birthday problem by simulation and exactly, and can explain why 23 is enough.
- [ ] I can name three places where probability appears in training or using a model.

**Next:** Day 37: Common distributions: Bernoulli, binomial, uniform, normal (coming soon)
