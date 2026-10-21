# Day 39: Expectation, Variance and the Law of Large Numbers

**Phase 1 · Week 6 · ~90 minutes**

## Objectives

- [ ] Compute an expected value as a probability-weighted average, by hand and in NumPy, and explain it as a balance point.
- [ ] Use linearity of expectation, including the trick of counting with 0/1 "indicator" variables.
- [ ] Compute variance and standard deviation by hand, with both the definition and the shortcut, and say what units each is in.
- [ ] Add variances of independent random variables, and explain why standard deviations do **not** add.
- [ ] Derive why the average of \( n \) independent copies has standard deviation \( \sigma/\sqrt{n} \), and check it by simulation.
- [ ] Show the law of large numbers with a real plot of running means, and tell the true mean apart from a sample mean.
- [ ] Explain why a mini-batch gradient is a noisy but unbiased estimate of the full gradient, and how its noise shrinks with batch size.

**Prerequisites.** [Day 36](day-36-probability-basics.md) (random variables, probability as long-run frequency, simulating with `rng`), [Day 37](day-37-distributions.md) (Bernoulli and normal distributions), [Day 38](day-38-bayes-rule.md) (independence), [Day 33](day-33-vector-matrix-gradients.md) (the MSE gradient) and [Day 19](day-19-matplotlib.md) (plotting).

---

## The big idea

An insurance company has no idea which of its customers will crash a car this year. Yet it sets prices a year ahead and stays in business, because it doesn't need to know any single customer. It needs two numbers about the *average* customer: how much a policy costs it **on average**, and how much that cost **swings** from customer to customer. Then it relies on one law of nature: over a million customers, the swings cancel and the average becomes almost perfectly predictable.

Those are today's three ideas:

| Idea | The question it answers | Symbol |
|---|---|---|
| **Expected value** | What is the long-run average per try? | \( \mathbb{E}[X] \), \( \mu \) |
| **Variance** and **standard deviation** | How far from that average does a single try typically land? | \( \mathrm{Var}(X) \), \( \sigma^2 \); \( \sigma \) |
| **Law of large numbers** | Why does the average of many tries settle down to the expected value? | |

**Analogy for an app developer.** You cannot predict whether *this* user will buy the $4.99 gem pack. You can predict, quite precisely, how much 100,000 users will spend in total. Every revenue forecast, every A/B test, and (it turns out) every step of training a neural network leans on that gap between "one try is unpredictable" and "the average of many is not".

## Expected value: the probability-weighted average

On Day 36 you met a **random variable** \( X \): a number produced by a chance process, like the face of a die. Some notation, one symbol at a time:

- \( x \) (lower case) is one value \( X \) can take, like 4.
- \( p(x) \) is the probability that \( X \) equals \( x \): for a fair die, \( p(4) = 1/6 \). Day 36 called this list of probabilities the **probability mass function** (pmf).
- \( \mathbb{E}[X] \) (read "the expected value of \( X \)", or "the expectation of \( X \)") is defined as

\[
\mathbb{E}[X] = \sum_{x} x \, p(x)
\]

In words: multiply each possible value by its probability, and add. It's an ordinary average where each value is weighted by how often it happens. The Greek letter \( \mu \) ("mu") is the usual short name for it: \( \mu = \mathbb{E}[X] \).

**Worked example: a fair die.** The faces are 1 to 6, each with probability \( 1/6 \):

\[
\mathbb{E}[X] = 1 \cdot \tfrac{1}{6} + 2 \cdot \tfrac{1}{6} + 3 \cdot \tfrac{1}{6} + 4 \cdot \tfrac{1}{6} + 5 \cdot \tfrac{1}{6} + 6 \cdot \tfrac{1}{6} = \frac{21}{6} = 3.5
\]

You will never roll a 3.5. The expected value is not "the most likely value" or "a value you'll see"; it is **what the average settles to if you roll forever**. Day 36 defined probability as a long-run frequency, and this is the same idea one level up: a long-run *average*.

```python
import numpy as np

faces = np.arange(1, 7)          # [1 2 3 4 5 6]
p = np.full(6, 1 / 6)            # each face has probability 1/6
mu = (faces * p).sum()           # multiply, then add: the definition
mu                               # 3.5

rng = np.random.default_rng(39)
rolls = rng.integers(1, 7, size=100_000)   # 100,000 rolls (the 7 is excluded)
rolls[:10]                       # array([6, 4, 2, 4, 5, 1, 2, 2, 5, 4])
rolls.mean()                     # 3.50077
```

The exact answer comes from the formula; the simulation lands within a thousandth of it. (In Jupyter, NumPy 2 shows a single number as `np.float64(3.5)`; the comments show the plain value, which is what `print()` gives you.)

### The balance point

Here is the picture to keep. Draw the pmf as weights sitting on a plank: a weight of \( p(x) \) at position \( x \). The expected value is the one spot where you can put the pivot and the plank balances. Below, \( X \) is the number of notifications a user opens in a day, with probabilities 0.20, 0.25, 0.20, 0.15, 0.10, 0.10 for 0 to 5 opens:

<figure class="diagram" markdown>
![Animation: six bars of heights 0.20, 0.25, 0.20, 0.15, 0.10 and 0.10 stand on a plank at positions 0 to 5; a triangular pivot slides under the plank; at position 1 the plank tips down to the right, at 3.5 it tips down to the left, and at 2.0 it rests level while a panel shows the weighted sum 0×0.20 + 1×0.25 + 2×0.20 + 3×0.15 + 4×0.10 + 5×0.10 = 2.0](../../assets/img/phase-1/day39-balance-point.svg)
<figcaption>Animated: the expected value is where the probability "weights" balance. Pivot too far left and the plank tips right, too far right and it tips left.</figcaption>
</figure>

\[
\mathbb{E}[X] = 0(0.20) + 1(0.25) + 2(0.20) + 3(0.15) + 4(0.10) + 5(0.10) = 2.0
\]

```python
opens = np.arange(6)
p_opens = np.array([0.20, 0.25, 0.20, 0.15, 0.10, 0.10])
p_opens.sum()                    # 1.0     a pmf must add up to 1
(opens * p_opens).sum()          # 2.0
```

The balance picture also tells you something the formula hides: one far-away value with a small probability can drag the balance point a long way, the way a light child at the very end of a seesaw can balance a heavy adult near the middle. Keep that in mind for the next example.

### A game's in-app purchases

A free game has four kinds of player. Each player's spend this month is a random variable \( S \):

| Spend \( s \) | Probability \( p(s) \) | \( s \cdot p(s) \) |
|---|---|---|
| $0.00 (spends nothing) | 0.950 | 0 |
| $0.99 (a small pack) | 0.030 | 0.0297 |
| $4.99 (a medium pack) | 0.015 | 0.07485 |
| $19.99 (a "whale") | 0.005 | 0.09995 |
| | **total 1** | **\( \mathbb{E}[S] = 0.2045 \)** |

So the game earns about 20 cents per player per month on average. That number has a name in the business: **ARPU**, average revenue per user. Notice that the rarest group, half a percent of players, contributes the largest share of it: that is the seesaw effect.

```python
spend = np.array([0.0, 0.99, 4.99, 19.99])
prob = np.array([0.95, 0.03, 0.015, 0.005])
arpu = (spend * prob).sum()
arpu                             # 0.20450000000000002   (0.2045 plus floating-point dust)

players = rng.choice(spend, size=200_000, p=prob)   # 200,000 simulated players
players[:8]                      # array([0., 0., 0., 0., 0., 0., 0., 0.])
players.mean()                   # 0.20457325
(players > 0).mean()             # 0.049675    about 5% pay anything
```

`rng.choice(values, size=..., p=...)` draws from any pmf you hand it. The first eight players spent nothing, which is what 95% of players do; the average is still about 20 cents.

### The expected value of a function

Often you care not about \( X \) itself but about something computed from it: a squared error, a cost, a reward. For any function \( g \), weight \( g(x) \) by the same probabilities:

\[
\mathbb{E}[g(X)] = \sum_{x} g(x)\, p(x)
\]

**Worked example.** For the die with \( g(x) = x^2 \):

\[
\mathbb{E}[X^2] = \frac{1 + 4 + 9 + 16 + 25 + 36}{6} = \frac{91}{6} \approx 15.17
\]

```python
(faces ** 2 * p).sum()           # 15.166666666666666   = 91/6
mu ** 2                          # 12.25                (E[X])² is a different number
(rolls ** 2).mean()              # 15.18513             the simulation agrees with 91/6
```

!!! warning "Common mistake: \( \mathbb{E}[g(X)] \ne g(\mathbb{E}[X]) \)"
    The average of the squares (15.17) is not the square of the average (12.25). You can only move \( \mathbb{E} \) inside a function when the function is a straight line (\( aX + b \), next section). In ML this bites when you compute "the loss of the average prediction" instead of "the average of the losses": they are different numbers, and only the second one is the loss.

This formula is how ML defines its most important number. A model's **loss** (Day 33's mean squared error) is \( \mathbb{E}[g(X)] \) where \( X \) is "a random example from the data" and \( g \) is "the error the model makes on it". Training minimizes an expected value.

## Linearity of expectation

Two rules, which hold for **every** pair of random variables, independent or not:

\[
\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b
\qquad\qquad
\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]
\]

Here \( a \) and \( b \) are ordinary fixed numbers. In words: scaling and shifting the values scales and shifts the average, and **the average of a sum is the sum of the averages**. Why the first one is true: \( \sum (a x + b)\,p(x) = a \sum x\,p(x) + b \sum p(x) = a\,\mathbb{E}[X] + b \), because the probabilities add up to 1.

**Worked examples.**

- Two dice: \( \mathbb{E}[X + Y] = 3.5 + 3.5 = 7 \). No need to list all 36 outcomes.
- \( 2X + 1 \) for one die: \( 2(3.5) + 1 = 8 \).
- 1,000 players of the game above: expected revenue \( 1000 \times 0.2045 = \$204.50 \).

```python
d1 = rng.integers(1, 7, size=100_000)    # die 1
d2 = rng.integers(1, 7, size=100_000)    # die 2, independent of die 1
(d1 + d2).mean()                         # 6.99119     ≈ 3.5 + 3.5
(2 * d1 + 1).mean()                      # 7.98502     ≈ 2(3.5) + 1

# Linearity does not need independence. Y = 7 - X depends completely on X:
(d1 + (7 - d1)).mean()                   # 7.0         always exactly 7, and 3.5 + 3.5 = 7
```

### Counting with indicators: the birthday problem again

Linearity gets powerful when combined with **indicator variables**: a random variable that is 1 if something happens and 0 if not (a Bernoulli variable from Day 37). Its expected value is just the probability of that something: \( \mathbb{E}[I] = 1 \cdot p + 0 \cdot (1 - p) = p \).

On Day 36 you found that 23 people have about a 50% chance that *some* two share a birthday. Now ask a different question: **on average, how many pairs share a birthday?** Counting the cases directly is a mess. With indicators it's two lines:

- There are \( \binom{23}{2} = \frac{23 \times 22}{2} = 253 \) pairs of people (read \( \binom{23}{2} \) as "23 choose 2": the number of ways to pick 2 people out of 23). Give each pair an indicator: 1 if those two share a birthday.
- For any one pair, the chance of a match is \( 1/365 \). So each indicator has expected value \( 1/365 \).
- The number of matching pairs is the sum of the 253 indicators, so by linearity its expected value is \( 253 / 365 \approx 0.693 \).

The pairs are not independent (if Ana matches Ben and Ben matches Cy, then Ana matches Cy), and it doesn't matter: linearity never needs independence.

```python
trials = 20_000
bdays = rng.integers(0, 365, size=(trials, 23))           # 20,000 rooms of 23 people
same = bdays[:, :, None] == bdays[:, None, :]             # (20000, 23, 23) via broadcasting (Day 9)
pairs = (same.sum(axis=(1, 2)) - 23) // 2                 # drop the diagonal, count each pair once
253 / 365                                                 # 0.6931506849315069
pairs.mean()                                              # 0.6966
(pairs > 0).mean()                                        # 0.50695   Day 36's answer, for free
```

## Variance and standard deviation

Two games can have the same ARPU and feel completely different: one where everybody pays 20 cents, and one where almost nobody pays and a few whales pay a lot. The expected value can't tell them apart. We need a number for **spread**.

The natural idea: measure how far each value is from the mean, \( x - \mu \) (the **deviation**), and average that. But positive and negative deviations cancel: the average deviation is always exactly 0. So square the deviations first, which makes them all positive and punishes big misses more:

\[
\mathrm{Var}(X) = \mathbb{E}\big[(X - \mu)^2\big] = \sum_x (x - \mu)^2\, p(x)
\qquad\qquad
\sigma = \sqrt{\mathrm{Var}(X)}
\]

- \( \mathrm{Var}(X) \), also written \( \sigma^2 \) ("sigma squared"), is the **variance**: the expected squared distance from the mean.
- \( \sigma \) ("sigma") is the **standard deviation**: the square root of the variance. Variance is in squared units (dollars², dice-points²), which nobody can picture; taking the root brings it back to the original units. Read \( \sigma \) as "a typical distance from the mean".

**Worked example: the die.** With \( \mu = 3.5 \), the deviations are \( -2.5, -1.5, -0.5, 0.5, 1.5, 2.5 \), and their squares are \( 6.25, 2.25, 0.25, 0.25, 2.25, 6.25 \). Each has probability \( 1/6 \):

\[
\mathrm{Var}(X) = \frac{6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25}{6} = \frac{17.5}{6} = \frac{35}{12} \approx 2.917
\qquad
\sigma = \sqrt{2.917} \approx 1.708
\]

<figure class="diagram" markdown>
![Animation: the faces 1 to 6 sit on a number line with the mean 3.5 marked in red; an arrow from the mean to each face shows its deviation, and a square with that deviation as its side grows under each face, labelled with its area 6.25, 2.25, 0.25, 0.25, 2.25, 6.25; then a highlighted average square of area 2.92 appears, labelled variance, with its side of 1.71 labelled standard deviation](../../assets/img/phase-1/day39-variance-squares.svg)
<figcaption>Animated: variance is the average area of the squares built on each deviation; the standard deviation is the side of that average square.</figcaption>
</figure>

There is a shortcut that is often easier by hand, "the mean of the square minus the square of the mean":

\[
\mathrm{Var}(X) = \mathbb{E}[X^2] - \mu^2 = \frac{91}{6} - 3.5^2 = 15.1\overline{6} - 12.25 = 2.91\overline{6}
\]

It comes from expanding the square, \( (X - \mu)^2 = X^2 - 2\mu X + \mu^2 \), and taking the expectation of each term with linearity: \( \mathbb{E}[X^2] - 2\mu \cdot \mu + \mu^2 \). And it explains the warning box above: \( \mathbb{E}[X^2] \) and \( \mu^2 \) differ by exactly the variance, so they are equal only when nothing varies.

```python
dev = faces - mu
dev                                   # array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])
var = (dev ** 2 * p).sum()
var                                   # 2.9166666666666665   = 35/12
(faces ** 2 * p).sum() - mu ** 2      # 2.916666666666666    the shortcut (last digit: rounding)
sigma = np.sqrt(var)
sigma                                 # 1.707825127659933

rolls.var(), rolls.std()              # (2.9297394071000005, 1.7116481551709162)   from the 100,000 rolls
```

**Same mean, different spread.** A trick die with faces 1, 1, 1, 6, 6, 6 also has mean 3.5. Every roll is 2.5 away from the mean, so its variance is \( 2.5^2 = 6.25 \) and \( \sigma = 2.5 \): the same average, much wilder. And the game:

```python
trick = np.array([1, 1, 1, 6, 6, 6])        # six equally likely faces
trick.mean(), trick.var(), trick.std()      # (3.5, 6.25, 2.5)

var_spend = ((spend - arpu) ** 2 * prob).sum()
sd_spend = np.sqrt(var_spend)
var_spend, sd_spend                          # (2.3590847499999996, 1.5359312321845662)
players.std()                                # 1.540754150695184
```

A typical player is \$1.54 away from an average of \$0.20: the spread is seven and a half times the mean. That one fact will decide how many players you need to measure ARPU, later in this lesson.

**Scaling and shifting.** Shifting every value by \( b \) moves the mean but not the spread; scaling by \( a \) scales every deviation by \( a \), so the squared deviations by \( a^2 \):

\[
\mathrm{Var}(aX + b) = a^2\, \mathrm{Var}(X)
\qquad
\sigma_{aX + b} = |a|\, \sigma_X
\]

Example: converting the game's spend from dollars to cents (\( a = 100 \)) multiplies \( \sigma \) by 100 and the variance by 10,000. One more you'll reuse: a Bernoulli variable (1 with probability \( p \), else 0) has mean \( p \) and, since \( X^2 = X \) for 0/1 values, variance \( p - p^2 = p(1-p) \). It is largest at \( p = 0.5 \), when the outcome is hardest to guess.

!!! warning "Common mistake: `np.var` divides by \( n \), not \( n - 1 \)"
    `np.var(data)` and `np.std(data)` treat the array as the *whole* population and divide by \( n \). pandas' `Series.var()` and `Series.std()` divide by \( n - 1 \) by default, so the same column gives slightly different numbers in the two libraries. Passing `ddof=1` to NumPy ("delta degrees of freedom") makes it divide by \( n - 1 \). Today we compute the variance of a known distribution, or of a huge simulation where the difference is invisible, so the default is fine; Day 43 explains when and why \( n - 1 \) is the right choice for a sample.

## The variance of a sum

Linearity says means always add. Variances add too, but **only when the variables are independent**:

\[
\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) \qquad \text{if } X \text{ and } Y \text{ are independent}
\]

**Worked example.** Two independent dice: \( \mathrm{Var}(X + Y) = 35/12 + 35/12 = 35/6 \approx 5.83 \), so \( \sigma \approx 2.42 \). Notice that the standard deviation is **not** \( 1.71 + 1.71 = 3.42 \). Standard deviations combine like the sides of a right triangle, \( \sqrt{1.71^2 + 1.71^2} \): independent wobbles partly cancel each other out.

Why does independence matter? Expand \( (X - \mu_X + Y - \mu_Y)^2 \) and you get the two squared terms plus a cross term \( 2(X - \mu_X)(Y - \mu_Y) \). Its expected value, divided by 2, is the **covariance** you used to build covariance matrices on Days 24 and 25. The full rule is

\[
\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)
\]

and independent variables have covariance 0. Two extreme dependent cases show the cross term at work:

```python
(d1 + d2).var()          # 5.9170123839   independent: ≈ 35/12 + 35/12 = 5.83
(d1 + d1).var()          # 11.6955355996  Y = X: the same wobble twice, Var(2X) = 4 Var(X) ≈ 11.67
(d1 + (7 - d1)).var()    # 0.0            Y = 7 - X: the wobbles cancel, the sum is always 7
```

Day 40 takes covariance and dependence further.

### The average of n independent copies

Now the result this whole lesson is building towards. Take \( n \) independent draws \( X_1, X_2, \dots, X_n \) of the same random variable, each with mean \( \mu \) and variance \( \sigma^2 \), and average them. Write the average as \( \bar{X} \) ("X bar"):

\[
\bar{X} = \frac{X_1 + X_2 + \dots + X_n}{n}
\]

Two steps, each using a rule from above:

1. **Its mean.** By linearity, \( \mathbb{E}[X_1 + \dots + X_n] = n\mu \), and dividing by \( n \) gives \( \mathbb{E}[\bar{X}] = \mu \). The average is aimed at the right target.
2. **Its variance.** By independence, \( \mathrm{Var}(X_1 + \dots + X_n) = n\sigma^2 \). Dividing by \( n \) is "scaling by \( a = 1/n \)", which divides the variance by \( n^2 \):

\[
\mathrm{Var}(\bar{X}) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}
\qquad\qquad
\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}
\]

**Worked example.** The average of 100 dice has mean 3.5 and standard deviation \( 1.708 / \sqrt{100} = 0.1708 \). The average of 10,000 dice: \( 1.708 / 100 = 0.017 \). To make the average 10 times steadier you need **100 times** as many samples. That square root is one of the most important facts in all of data science, and Day 43 gives \( \sigma/\sqrt{n} \) its official name, the *standard error*.

```python
for n in (1, 4, 16, 64, 100):
    means = rng.integers(1, 7, size=(10_000, n)).mean(axis=1)   # 10,000 averages of n dice
    print(f"n={n:4d}  std of the mean = {means.std():.4f}   sigma/sqrt(n) = {sigma / np.sqrt(n):.4f}")
# n=   1  std of the mean = 1.7033   sigma/sqrt(n) = 1.7078
# n=   4  std of the mean = 0.8616   sigma/sqrt(n) = 0.8539
# n=  16  std of the mean = 0.4289   sigma/sqrt(n) = 0.4270
# n=  64  std of the mean = 0.2135   sigma/sqrt(n) = 0.2135
# n= 100  std of the mean = 0.1701   sigma/sqrt(n) = 0.1708
```

## The law of large numbers

Put the two facts together: \( \bar{X} \) is aimed at \( \mu \), and its spread around \( \mu \) shrinks like \( \sigma / \sqrt{n} \), all the way to zero. So:

> **Law of large numbers.** The average of \( n \) independent draws of the same random variable gets closer and closer to its expected value as \( n \) grows.

This is what made Day 36's "probability is a long-run frequency" honest: a frequency is just the average of an indicator variable, so it settles to the indicator's expected value, the probability. And it is what lets the insurance company sleep.

Watch it happen. A **running mean** is the average of the first 1, 2, 3, ... draws; `np.cumsum` gives the running totals, and dividing by \( 1, 2, 3, \dots \) turns them into averages:

```python
N = 10_000
r = rng.integers(1, 7, size=N)
running = np.cumsum(r) / np.arange(1, N + 1)     # mean of the first 1, 2, ..., N rolls
running[[9, 99, 999, 9_999]]                      # array([3.6   , 3.51  , 3.528 , 3.5021])
```

Here are five independent runs for the die and five for the game's spend, each against a band of \( \mu \pm 2\sigma/\sqrt{n} \):

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

def running_means(draws):
    """Row-wise running mean: column j holds the mean of the first j+1 draws."""
    return np.cumsum(draws, axis=1) / np.arange(1, draws.shape[1] + 1)

N = 20_000
n_axis = np.arange(1, N + 1)
keep = np.unique(np.geomspace(1, N, 800).astype(int)) - 1   # ~500 log-spaced points keep the file small
dice_runs = running_means(rng.integers(1, 7, size=(5, N)))
spend_runs = running_means(rng.choice(spend, size=(5, N), p=prob))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
panels = [(axes[0], dice_runs, mu, sigma, "Rolling a die", "running mean of the rolls", (1.5, 5.5)),
          (axes[1], spend_runs, arpu, sd_spend, "Players' monthly spend ($)", "running ARPU ($)", (-0.05, 1.0))]
colors = ["#3949ab", "#00897b", "#ef6c00", "#8e24aa", "#6d4c41"]
for ax, runs, m, s, title, ylabel, ylim in panels:
    band = 2 * s / np.sqrt(n_axis[keep])
    ax.fill_between(n_axis[keep], m - band, m + band, color="#fff59d", label=r"$\mu \pm 2\sigma/\sqrt{n}$")
    for run, c in zip(runs, colors):
        ax.plot(n_axis[keep], run[keep], color=c, lw=1.1)
    ax.axhline(m, color="#e53935", lw=1.5, ls="--", label=rf"true mean $\mu$ = {m:.4g}")
    ax.set_xscale("log")
    ax.set_xlabel("number of draws n (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("day39-running-means.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Two panels of running means on a log-scaled x axis from 1 to 20,000 draws. Left: five runs of die rolls start scattered between 1 and 6 and all close in on the dashed red line at 3.5, inside a yellow band that narrows like 2 sigma over root n. Right: five runs of players' spend start at zero, jump up whenever a big spender appears, and only slowly close in on the dashed line at 0.2045 dollars](../../assets/img/phase-1/day39-running-means.svg)
<figcaption>The law of large numbers, five runs each. The die settles fast; the game's spend, with its rare $19.99 whales, jumps around for thousands of players before it settles.</figcaption>
</figure>

Both converge, but at very different speeds, and the band tells you why: its width is set by \( \sigma \). The spend has \( \sigma \) about 7.5 times its mean, so each whale who shows up yanks the running ARPU upward and then it slowly sags back. How many players do you need before your measured ARPU has a standard deviation of just 1 cent? Solve \( \sigma / \sqrt{n} = 0.01 \) for \( n \):

\[
n = \left(\frac{\sigma}{0.01}\right)^2 = \left(\frac{1.536}{0.01}\right)^2 \approx 23{,}591 \text{ players}
\]

```python
(sd_spend / 0.01) ** 2           # 23590.847499999996
```

!!! warning "Common mistake: the gambler's fallacy"
    The law of large numbers does **not** say that a run of low rolls must be "made up for" by high rolls. Dice have no memory: the next roll is still 1 to 6, equally likely. The average converges because the early bad luck gets **diluted** by a growing pile of ordinary rolls, not corrected. In fact the *total* gap from the expected total usually grows, roughly like \( \sqrt{n} \), while the *average* gap shrinks like \( 1/\sqrt{n} \):

    ```python
    for n in (10, 100, 1_000, 10_000):
        totals = rng.integers(1, 7, size=(1_000, n), dtype=np.int8).sum(axis=1)   # 1,000 runs of n rolls
        gap = totals - 3.5 * n                   # total minus expected total, per run
        print(f"n={n:6d}  typical total gap = {gap.std():6.1f}   typical average gap = {gap.std() / n:.4f}")
    # n=    10  typical total gap =    5.5   typical average gap = 0.5467
    # n=   100  typical total gap =   16.3   typical average gap = 0.1628
    # n=  1000  typical total gap =   53.7   typical average gap = 0.0537
    # n= 10000  typical total gap =  165.3   typical average gap = 0.0165
    ```

    The typical total gap is about \( \sigma\sqrt{n} \) (\( 1.708 \times 100 \approx 171 \) at \( n = 10{,}000 \)) and keeps growing; only its share of \( n \) shrinks.

!!! android "Android bridge: averaging sensor readings"
    You've probably smoothed an accelerometer or a light sensor by averaging the last few readings. That is the law of large numbers at work: if each reading is the true value plus independent noise with standard deviation \( \sigma \), the average of \( N \) readings has noise \( \sigma/\sqrt{N} \).

    ```kotlin
    class AveragingListener(private val n: Int) : SensorEventListener {
        private var sum = 0.0
        private var count = 0
        override fun onSensorChanged(event: SensorEvent) {
            sum += event.values[2]                      // z-axis, m/s² for TYPE_ACCELEROMETER
            if (++count == n) {
                Log.d("Avg", "mean of $n readings = ${sum / n}")   // noise: σ / √n
                sum = 0.0; count = 0
            }
        }
        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
    }
    ```

    Two honest limits, both straight from today's math. Averaging removes *random* noise, but a sensor's fixed offset (bias) is part of the mean, so the average converges to "true value plus bias". And \( \sigma/\sqrt{N} \) assumes independent readings; consecutive samples from a hardware-filtered sensor are correlated, so in practice the gain is smaller. The same arithmetic sets the size of an A/B test (Day 45): to see a 1-cent change in ARPU, you need tens of thousands of users per variant, because one whale's purchase is louder than the effect you're looking for.

### Sample mean versus true mean

Two numbers that are easy to blur together:

- The **true mean** \( \mu \) is a fixed property of the random process. Nobody rolled anything to get 3.5; it came from the pmf.
- The **sample mean** \( \bar{x} \) is the average of the data you actually got. It is a random variable itself: collect a new sample and you get a new value. The law of large numbers says it lands near \( \mu \), within about \( \sigma/\sqrt{n} \).

In real work you almost never know \( \mu \); the sample mean is your *estimate* of it. Here's a real-data version: pretend the Play Store dataset's ratings are the whole "population" (so we know the true mean), and survey 20 random apps at a time (lesson tested on pandas 3.0):

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")    # or a local copy from the Datasets page
ratings = apps.drop_duplicates()["rating"].dropna().to_numpy()
len(ratings)                                        # 462
ratings.mean(), ratings.std()                       # (4.165584415584416, 0.4533731016292602)   the "true" mu and sigma

surveys = [rng.choice(ratings, size=20, replace=False).mean() for _ in range(5)]
np.round(surveys, 3)                                # array([4.24 , 4.245, 4.265, 4.145, 4.155])
ratings.std() / np.sqrt(20)                         # 0.10137730744129464   sigma / sqrt(20)
```

(You can download the CSV from the [Datasets](../../datasets.md) page and read it from a local path instead.) Each survey of 20 apps gives a different sample mean, scattered around the true mean by roughly \( \sigma/\sqrt{20} \). Day 43 turns this into the sampling distribution, and Day 44 into confidence intervals.

## Why averaging noisy gradients works: a preview of SGD

Here is where today's math runs every training job you'll ever start. On Day 33 you computed the gradient of the mean squared error for a model with parameter \( w \):

\[
L(w) = \frac{1}{n}\sum_{i=1}^{n} (w x_i - y_i)^2
\qquad\qquad
\frac{dL}{dw} = \frac{1}{n}\sum_{i=1}^{n} \underbrace{2\,(w x_i - y_i)\,x_i}_{g_i}
\]

Read the right-hand side with today's eyes: the full gradient is the **average** of \( n \) per-example gradients \( g_i \). Equivalently, it is the **expected value** of \( g_I \) where \( I \) is a randomly chosen example. Computing it exactly means touching all \( n \) examples, and with a million images that is one slow step.

So take a shortcut: pick a random **mini-batch** of \( B \) examples and average only their gradients. That mini-batch gradient is a sample mean of \( B \) draws of \( g_I \), so everything from this lesson applies:

1. **It is aimed at the right target** (unbiased): its expected value is the full gradient.
2. **Its noise is \( \sigma_g / \sqrt{B} \)**, where \( \sigma_g \) is the standard deviation of the per-example gradients.

Let's check both on 10,000 made-up points with \( y \approx 3x \), at the starting guess \( w = 0 \):

```python
n = 10_000
x = rng.normal(0, 1, size=n)
y = 3 * x + rng.normal(0, 1, size=n)     # the true slope is 3, plus noise

w = 0.0
g_each = 2 * (w * x - y) * x              # one gradient per example (shape (10000,))
g_full = g_each.mean()                    # the full-batch gradient: Day 33's formula
sigma_g = g_each.std()
g_full, sigma_g                           # (-6.074703797007767, 8.788615299186915)

for B in (1, 4, 16, 64, 256):
    idx = rng.integers(0, n, size=(5_000, B))     # 5,000 random mini-batches of size B
    est = g_each[idx].mean(axis=1)                # one gradient estimate per mini-batch
    print(f"B={B:4d}  mean of estimates = {est.mean():7.3f}   std = {est.std():.3f}"
          f"   sigma_g/sqrt(B) = {sigma_g / np.sqrt(B):.3f}")
# B=   1  mean of estimates =  -6.322   std = 9.047   sigma_g/sqrt(B) = 8.789
# B=   4  mean of estimates =  -6.036   std = 4.401   sigma_g/sqrt(B) = 4.394
# B=  16  mean of estimates =  -6.054   std = 2.208   sigma_g/sqrt(B) = 2.197
# B=  64  mean of estimates =  -6.074   std = 1.113   sigma_g/sqrt(B) = 1.099
# B= 256  mean of estimates =  -6.077   std = 0.547   sigma_g/sqrt(B) = 0.549
```

Every batch size averages out to the full gradient, about \( -6.07 \) (the single-example estimates wobble so much that even the average of 5,000 of them has a standard deviation of \( 9.05/\sqrt{5000} \approx 0.13 \), so \( -6.32 \) is an ordinary two-standard-deviation miss). And the spread follows \( \sigma_g/\sqrt{B} \) to within a couple of percent. A single example's gradient (\( B = 1 \)) is wild: its typical error, 9, is bigger than the gradient itself. At \( B = 256 \) it's about 9% of it.

(The mini-batches here are drawn *with replacement* so the math is exact; real training shuffles the data and takes batches without replacement, which makes the noise very slightly smaller.)

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for B, c in [(8, "#ef6c00"), (64, "#00897b"), (512, "#3949ab")]:
    est = g_each[rng.integers(0, n, size=(5_000, B))].mean(axis=1)
    ax1.hist(est, bins=80, range=(-18, 2), density=True, histtype="stepfilled",
             alpha=0.45, color=c, edgecolor=c, label=f"batch size B = {B}")
ax1.axvline(g_full, color="#e53935", ls="--", lw=1.5, label=f"full gradient = {g_full:.2f}")
ax1.set_xlabel("mini-batch gradient estimate")
ax1.set_ylabel("density")
ax1.set_title("Mini-batch gradients at w = 0")
ax1.legend(frameon=False, loc="upper left")

Bs = 2 ** np.arange(0, 11)                        # 1, 2, 4, ..., 1024
stds = [g_each[rng.integers(0, n, size=(2_000, B))].mean(axis=1).std() for B in Bs]
ax2.loglog(Bs, sigma_g / np.sqrt(Bs), color="#e53935", ls="--", label=r"$\sigma_g/\sqrt{B}$")
ax2.loglog(Bs, stds, "o", color="#3949ab", label="measured std")
ax2.set_xlabel("batch size B (log scale)")
ax2.set_ylabel("std of the estimate (log scale)")
ax2.set_title("Noise shrinks like 1/sqrt(B)")
ax2.legend(frameon=False)
fig.tight_layout()
fig.savefig("day39-minibatch-noise.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Left: three overlapping histograms of mini-batch gradient estimates for batch sizes 8, 64 and 512 around the dashed red line of the full gradient near minus 6; the batch-size-8 histogram is wide and lopsided with a long tail to the left, the 64 one narrower and the 512 one a narrow spike on the line. Right: a log-log plot of the estimate's standard deviation against batch size from 1 to 1024; the measured blue dots fall on a red dashed straight line of slope minus one half, sigma over root B](../../assets/img/phase-1/day39-minibatch-noise.svg)
<figcaption>Every batch size averages out to the full gradient (left; small batches are lopsided, because a few examples have huge gradients, but their mean is still right); bigger batches are only less noisy, by a factor of \( \sqrt{B} \) (right).</figcaption>
</figure>

This is the trade-off behind **stochastic gradient descent** (SGD), which Day 52 builds properly:

- A batch of 64 costs 1/156 of a full pass over these 10,000 examples, yet its gradient points the right way *on average*. Many cheap, slightly noisy steps beat a few expensive, exact ones.
- Quadrupling the batch size quadruples the cost of a step but only **halves** the noise. That diminishing return is why typical batch sizes are in the tens to hundreds, not millions.
- Over many steps the noise averages out, by the same law of large numbers that settled the dice.

## Where this shows up in ML

| Today's idea | Where you'll meet it |
|---|---|
| Expected value | The loss is the expected error over the data; a classifier's expected cost when you choose its decision threshold |
| \( \mathbb{E}[g(X)] \ne g(\mathbb{E}[X]) \) | Average of losses vs loss of the average; Jensen's inequality in the theory papers |
| Variance | Feature standardization divides by \( \sigma \) (Day 22); weight initialization picks a variance so signals neither explode nor vanish (Day 37) |
| Variances of independent terms add | Why a sum of \( n \) inputs times random weights has variance proportional to \( n \), and initializers divide by \( n \) |
| \( \sigma/\sqrt{n} \) and the law of large numbers | Mini-batch gradients (Day 52), error bars on a test-set accuracy (Day 44), A/B test sizes (Day 45) |

---

## Exercises

1. A loaded die shows 6 with probability 0.5 and each of 1 to 5 with probability 0.1. By hand, find \( \mathbb{E}[X] \), \( \mathbb{E}[X^2] \), the variance and the standard deviation. Check with NumPy, both from the formula and by simulating 100,000 rolls.
2. You shuffle 10 cards numbered 1 to 10. What is the expected number of cards that end up in their own position (card 1 first, card 2 second, and so on)? Use indicators and linearity, then simulate with `rng.permutation`. What happens for 52 cards?
3. An app's cold start has three independent stages: loading the process (mean 120 ms, standard deviation 30 ms), inflating the UI (mean 300 ms, sd 40 ms) and the first network call (mean 80 ms, sd 20 ms). What are the mean and standard deviation of the total start time? Why is the sd not 90 ms? Check by simulating each stage as a normal distribution (Day 37).
4. How many die rolls do you need before the standard deviation of their average is at most 0.05? Verify by simulating 5,000 averages at that \( n \).
5. In the SGD example, what is the smallest batch size \( B \) whose gradient estimate has a standard deviation below 10% of the size of the full gradient \( |g_\text{full}| \)? Check it by simulation.
6. For a fair die, compute \( \mathbb{E}[1/X] \) and compare it with \( 1/\mathbb{E}[X] \). Which is bigger? Explain in a sentence why the warning box says you can't move \( \mathbb{E} \) inside most functions.

??? success "Solutions"
    ```python
    # 1: E[X] = 0.1(1+2+3+4+5) + 0.5(6) = 1.5 + 3 = 4.5
    #    E[X²] = 0.1(1+4+9+16+25) + 0.5(36) = 5.5 + 18 = 23.5
    #    Var = 23.5 - 4.5² = 3.25,  sd = √3.25 ≈ 1.803
    p_load = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    m1 = (faces * p_load).sum()
    v1 = (faces ** 2 * p_load).sum() - m1 ** 2
    m1, v1, np.sqrt(v1)          # (4.5, 3.25, 1.8027756377319946)
    sim1 = rng.choice(faces, size=100_000, p=p_load)
    sim1.mean(), sim1.var()      # (4.49122, 3.2656029116000003)
    ```

    ```python
    # 2: indicator I_k = 1 if card k lands in position k; P = 1/10 for each card,
    #    so E[matches] = 10 × 1/10 = 1. For 52 cards: 52 × 1/52 = 1. Always 1!
    matches10 = np.array([(rng.permutation(10) == np.arange(10)).sum() for _ in range(20_000)])
    matches52 = np.array([(rng.permutation(52) == np.arange(52)).sum() for _ in range(20_000)])
    matches10.mean(), matches52.mean()      # (0.98855, 1.0066)
    ```
    The indicators are dependent (if 9 cards are in place, the 10th must be too), but linearity doesn't care. More cards means each one is less likely to match, and there are more of them, and the two effects cancel exactly.

    ```python
    # 3: means add: 120 + 300 + 80 = 500 ms
    #    variances add (independent): 30² + 40² + 20² = 900 + 1600 + 400 = 2900, sd = √2900 ≈ 53.9 ms
    m = 200_000
    total = (rng.normal(120, 30, m) + rng.normal(300, 40, m) + rng.normal(80, 20, m))
    total.mean(), total.std(), np.sqrt(2900)
    # (500.0922678624046, 53.88211180243282, 53.85164807134504)
    ```
    Standard deviations don't add, variances do. The sd would only be 90 ms if all three stages were perfectly in step (a slow load always meaning a slow inflate and a slow network call); independent stages partly cancel.

    ```python
    # 4: sigma / √n ≤ 0.05  ⇒  n ≥ (1.7078 / 0.05)² ≈ 1166.7  ⇒  n = 1167
    n4 = int(np.ceil((sigma / 0.05) ** 2))
    n4                                                        # 1167
    rng.integers(1, 7, size=(5_000, n4)).mean(axis=1).std()   # 0.049658252769727185
    ```

    ```python
    # 5: sigma_g / √B < 0.1 |g_full|  ⇒  B > (sigma_g / (0.1 |g_full|))²
    B5 = int(np.floor((sigma_g / (0.1 * abs(g_full))) ** 2)) + 1
    B5                                   # 210     (8.789 / 0.6075)² ≈ 209.3, rounded up
    est5 = g_each[rng.integers(0, n, size=(5_000, B5))].mean(axis=1)
    est5.std(), 0.1 * abs(g_full)        # (0.6031291484917282, 0.6074703797007768)
    ```

    ```python
    # 6: E[1/X] = (1 + 1/2 + 1/3 + 1/4 + 1/5 + 1/6) / 6 = 2.45 / 6 ≈ 0.408
    (1 / faces * p).sum(), 1 / mu        # (0.4083333333333334, 0.2857142857142857)
    ```
    \( \mathbb{E}[1/X] \approx 0.408 \) is bigger than \( 1/\mathbb{E}[X] \approx 0.286 \). The curve \( 1/x \) bends: small values like 1 blow up to 1 while large ones only shrink a little, so averaging *after* applying the function weights the small values heavily. Only for straight lines \( aX + b \) does the order not matter.

## Resources

- [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/): the "Basic Probability" chapter has interactive expectation, variance and law-of-large-numbers visualizations you can play with.
- [Khan Academy: Statistics and probability](https://www.khanacademy.org/math/statistics-probability): the "Random variables" unit covers expected value, variance and combining random variables, with practice.
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/): chapter 6.4 covers means, variances and covariances, including sums of random variables.
- [Dive into Deep Learning: Probability and Statistics](https://d2l.ai/chapter_preliminaries/probability.html): the same ideas written for deep learning, with expectations and variances in code.
- [OpenIntro Statistics (free book)](https://www.openintro.org/book/os/): a gentle textbook with a section on random variables, expectation and combining variances.
- [NumPy docs: `Generator.choice`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html), [`numpy.var`](https://numpy.org/doc/stable/reference/generated/numpy.var.html) (the `ddof` argument) and [`numpy.cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html).
- [Android reference: `SensorEvent`](https://developer.android.com/reference/android/hardware/SensorEvent): what `values[0..2]` mean for each sensor type.

## Checklist

- [ ] I can compute \( \mathbb{E}[X] = \sum x\,p(x) \) by hand and explain it as the balance point of the pmf.
- [ ] I know that \( \mathbb{E}[g(X)] \) is not \( g(\mathbb{E}[X]) \) unless \( g \) is a straight line.
- [ ] I can use linearity of expectation with indicator variables, and I know it never needs independence.
- [ ] I can compute a variance with the definition and with \( \mathbb{E}[X^2] - \mu^2 \), and say why we take the square root.
- [ ] I know that variances of independent variables add and standard deviations don't.
- [ ] I can derive \( \sigma/\sqrt{n} \) for an average, and use it to say how many samples I need.
- [ ] I plotted running means and can explain the law of large numbers without the gambler's fallacy.
- [ ] I can explain why a mini-batch gradient is unbiased, and why a 4× bigger batch only halves its noise.

**Next:** Day 40: Joint distributions, independence and the central limit theorem (coming soon)
