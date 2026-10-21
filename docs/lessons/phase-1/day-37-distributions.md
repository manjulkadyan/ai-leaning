# Day 37: Common Distributions: Bernoulli, Binomial, Uniform, Normal

**Phase 1 · Week 6 · ~120 minutes**

## Objectives

- [ ] Say what a probability distribution is, and tell a discrete distribution from a continuous one.
- [ ] Use the Bernoulli distribution for a single yes/no event, and simulate it with `rng.random() < p`.
- [ ] Compute binomial probabilities by hand with \( \binom{n}{k} p^k (1-p)^{n-k} \), and check them against a simulation.
- [ ] Explain why a continuous variable's probability is an **area** under a density curve, not a height.
- [ ] Use the uniform and normal distributions, the 68–95–99.7 rule and z-scores.
- [ ] Read and compute a CDF, and use it to get the probability of a range or a percentile.
- [ ] Sample each distribution with NumPy and compare the histogram to the formula.
- [ ] Point to where these distributions show up in ML: weight initialization, dropout, noise and model outputs.

**Prerequisites.** [Day 36](day-36-probability-basics.md) (outcomes, events, random variables, the PMF, simulating with `rng`), [Day 19](day-19-matplotlib.md) (histograms), [Day 22](day-22-norms-distances.md) (standardizing a column) and [Day 12](day-12-layer-is-matmul.md) (a layer is `x @ W`, ReLU).

---

## The big idea

Yesterday you simulated random things one at a time: a coin, a die, a crash. Today you meet the **shapes** that keep turning up when you do that. A **probability distribution** is the complete answer to "which values can this random variable take, and how likely is each one?" Write it down once and you can answer any question about the variable without running another experiment.

**Analogy.** A random variable is a vending machine that drops a random snack each time you press the button. The distribution is the label on the side: "40% crisps, 35% chocolate, 25% gum". You can't predict the next press, but the label tells you exactly what a thousand presses will look like. Four labels cover a surprising amount of the world:

| Distribution | Models | Parameters | Example | NumPy |
|---|---|---|---|---|
| **Bernoulli** | one yes/no event | \( p \), the chance of "yes" | does this user tap the notification? | `rng.random() < p` |
| **Binomial** | how many "yes" in \( n \) independent tries | \( n \), \( p \) | how many of 1,000 sessions crash? | `rng.binomial(n, p)` |
| **Uniform** | any value in a range, all equally likely | low \( a \), high \( b \) | a random delay between 0 and 10 minutes | `rng.uniform(a, b)` |
| **Normal** | values piled around an average, thinning out symmetrically | mean \( \mu \), spread \( \sigma \) | the cold-start time of your app | `rng.normal(mu, sigma)` |

The first two are **discrete**: the values can be listed (0 or 1; 0, 1, 2, ...). The last two are **continuous**: the value can be any real number in a range, and that changes how probability works, as you'll see.

Every block on this page shares one namespace, so run them top to bottom. One generator with a fixed seed makes every number reproducible:

```python
import math
import numpy as np

rng = np.random.default_rng(37)
```

(In Jupyter, NumPy 2 shows a single number as `np.float64(0.15)`; the comments show the plain value, which is what `print()` gives you.)

## Bernoulli: one yes/no trial

A **Bernoulli** random variable \( X \) takes the value 1 ("success", "yes") with probability \( p \) and 0 ("failure", "no") with probability \( 1 - p \). That's the whole distribution: one number, \( p \), describes it.

\[
P(X = 1) = p
\qquad
P(X = 0) = 1 - p
\]

- \( X \) is the random variable: the result before you know it.
- \( P(X = 1) \) reads "the probability that \( X \) comes out as 1".
- "Success" is just a label for whichever outcome you code as 1. A crash can be a "success" in this sense.

**Worked example.** A push notification gets tapped by 10% of users: \( p = 0.1 \). For one user, \( P(X = 1) = 0.1 \) and \( P(X = 0) = 0.9 \). The two add up to 1, as the probabilities of all possible outcomes must (Day 36).

To simulate one: draw a uniform number between 0 and 1 and check whether it is below \( p \). A number in \( [0, 1) \) lands below 0.1 exactly 10% of the time, so the comparison is True with probability \( p \).

```python
p = 0.1                              # chance that one user taps the notification
taps = rng.random(20) < p            # 20 Bernoulli trials: True means "tapped"
taps.astype(int)
# array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
taps.mean()                          # 0.15   3 taps out of 20
```

Twenty users gave 15%, not 10%: small samples wobble. With 100,000 users the frequency settles near \( p \), exactly as Day 36's running frequency did:

```python
many = rng.random(100_000) < p       # 100,000 users
many.mean()                          # 0.09985
many.var()                           # 0.08987997749999999
p * (1 - p)                          # 0.09000000000000001
```

Because the values are 0 and 1, the **mean of a Bernoulli variable is \( p \)**: averaging 0s and 1s gives the fraction of 1s. Its **variance** (the average squared distance from the mean, which Day 39 covers properly) is \( p(1 - p) \). It is largest at \( p = 0.5 \), where the outcome is hardest to guess, and zero at \( p = 0 \) or \( 1 \), where nothing is random at all.

## Binomial: counting successes

Send the notification to \( n = 5 \) users, each tapping independently with \( p = 0.2 \). How many taps will you get? The count \( K \) can be 0, 1, 2, 3, 4 or 5, and its distribution is the **binomial** distribution with parameters \( n \) and \( p \). A binomial count is a sum of \( n \) independent Bernoulli variables.

The Galton board below is a physical version. Each peg is a Bernoulli trial (bounce right with probability 0.5); after 4 pegs, the bin a ball lands in is the number of right bounces.

<figure class="diagram" markdown>
![Animation: a Galton board with four rows of blue pegs; four orange balls fall one at a time, bouncing left or right at each peg, with their paths R L R R, L L R L, R L L R and L R R L listed on the right as k = 3, 1, 2 and 2, and teal bars growing in the bins below; then the bars jump to the expected counts out of 16 balls, 1, 4, 6, 4, 1, with the note that C(4, k) paths each have chance 1/16 and P(k = 2) = 6/16 = 0.375](../../assets/img/phase-1/day37-galton.svg)
<figcaption>Animated: each ball makes four 50/50 choices, and the number of right bounces is a binomial count. The middle bins fill fastest because more paths lead there.</figcaption>
</figure>

### The formula, one piece at a time

Back to 5 users with \( p = 0.2 \). What is \( P(K = 2) \), the chance of exactly 2 taps?

**Step 1: one specific arrangement.** Take "users 1 and 2 tap, users 3, 4 and 5 don't", written TTNNN. The users are independent, so (Day 36's multiplication rule) multiply their chances:

\[
0.2 \times 0.2 \times 0.8 \times 0.8 \times 0.8 = 0.2^2 \times 0.8^3 = 0.04 \times 0.512 = 0.02048
\]

**Step 2: count the arrangements.** TNTNN, NNNTT and the others have the same chance, because each still has two 0.2s and three 0.8s. The number of ways to choose which 2 of the 5 users tap is written \( \binom{5}{2} \), read "5 choose 2":

```python
from itertools import product

ways = [w for w in product("TN", repeat=5) if w.count("T") == 2]
len(ways)                            # 10
["".join(w) for w in ways]
# ['TTNNN', 'TNTNN', 'TNNTN', 'TNNNT', 'NTTNN', 'NTNTN', 'NTNNT', 'NNTTN', 'NNTNT', 'NNNTT']
```

Listing gets slow for big \( n \), so there's a formula. \( n! \) ("n factorial") is \( n \times (n-1) \times \dots \times 2 \times 1 \), with \( 0! = 1 \) by convention:

\[
\binom{n}{k} = \frac{n!}{k!\,(n-k)!}
\qquad
\binom{5}{2} = \frac{5!}{2!\,3!} = \frac{120}{2 \times 6} = 10
\]

(Why: there are \( 5! \) orders to line the users up; take the first 2 as tappers; then the \( 2! \) orders of the tappers and the \( 3! \) orders of the rest don't change who tapped, so divide them out.)

```python
math.comb(5, 2)                                          # 10
math.factorial(5) // (math.factorial(2) * math.factorial(3))   # 10
```

**Step 3: add up the arrangements.** They can't happen at the same time, so their probabilities add (Day 36's addition rule): \( 10 \times 0.02048 = 0.2048 \). In general:

\[
P(K = k) = \binom{n}{k}\, p^k\, (1 - p)^{n - k}
\]

- \( p^k \): the \( k \) chosen users all succeed.
- \( (1 - p)^{n-k} \): the other \( n - k \) all fail.
- \( \binom{n}{k} \): how many ways there are to choose which ones succeed.

This list of probabilities, one per possible count, is the binomial's **probability mass function** (PMF) from Day 36, now as a formula instead of a tally:

```python
def binom_pmf(k, n, p):
    """P(exactly k successes in n independent trials, each with success chance p)."""
    return math.comb(n, k) * p**k * (1 - p)**(n - k)

binom_pmf(2, 5, 0.2)                 # 0.2048000000000001   the 10 x 0.02048 from above
pmf = np.array([binom_pmf(k, 5, 0.2) for k in range(6)])
pmf.round(4)                         # array([3.277e-01, 4.096e-01, 2.048e-01, 5.120e-02, 6.400e-03, 3.000e-04])
pmf.sum()                            # 1.0000000000000002   the counts 0..5 cover every possibility
```

(`0.2048000000000001` and `1.0000000000000002` are ordinary floating-point rounding, the same tiny errors as Day 29's numerical derivatives. NumPy switches to `e` notation, 3.000e-04 meaning 0.0003, when an array mixes big and very small numbers.)

### Simulation vs formula

`rng.binomial(n, p, size=...)` draws binomial counts directly. Run "send 5 notifications" 100,000 times and compare the frequencies with the formula:

```python
counts = rng.binomial(n=5, p=0.2, size=100_000)   # 100,000 runs of "send 5 notifications"
counts[:12]                          # array([1, 0, 1, 3, 0, 2, 1, 0, 1, 3, 1, 1])
freq = np.bincount(counts, minlength=6) / len(counts)   # fraction of runs with 0, 1, ..., 5 taps
freq.round(4)                        # array([3.273e-01, 4.110e-01, 2.031e-01, 5.170e-02, 6.700e-03, 2.000e-04])
np.abs(freq - pmf).max().round(4)    # 0.0017   simulation and formula agree to about 0.002
counts.mean(), 5 * 0.2               # (1.00033, 1.0)
```

`np.bincount` counts how many times each whole number appears. And the average count is \( np = 1 \): 5 users times a 0.2 chance each. The **binomial mean is \( np \)** and its **variance is \( np(1-p) \)**, which is \( n \) times the Bernoulli's, because the \( n \) trials are independent (Day 39 shows why variances of independent things add).

You can also build the same counts from raw Bernoulli trials, which is what a binomial *is*:

```python
same = (rng.random((100_000, 5)) < 0.2).sum(axis=1)   # 5 Bernoulli trials per row, summed
(np.bincount(same, minlength=6) / 100_000).round(4)
# array([3.273e-01, 4.095e-01, 2.051e-01, 5.110e-02, 6.600e-03, 4.000e-04])
```

!!! warning "Common mistake: forgetting the \( \binom{n}{k} \)"
    \( 0.2^2 \times 0.8^3 = 0.0205 \) is the chance of **one particular** arrangement, such as "the first two users tap". "Exactly two taps, any two users" is 10 times bigger, 0.2048. If your probabilities for \( k = 0..n \) don't sum to 1, a missing \( \binom{n}{k} \) is the usual culprit.

!!! warning "Common mistake: using the binomial when trials aren't independent"
    The formula assumes every trial has the same \( p \) and doesn't affect the others. Ten sessions from the **same** user on the same buggy device are not independent: if one crashes, the others probably do too. Count users, not sessions, or expect far more spread than \( np(1-p) \) predicts.

!!! android "Android bridge: what zero crashes in a staged rollout tells you"
    Play Console's staged rollouts ship a release to a percentage of users first. Suppose the new build crashes in 0.2% of sessions, and the first stage gets 1,000 sessions. The number of crashes is binomial with \( n = 1000 \), \( p = 0.002 \), and the chance you see **none at all** is \( (1 - 0.002)^{1000} \):

    ```python
    crash_rate = 0.002          # 0.2% of sessions crash (made-up numbers)
    sessions = 1000             # sessions in the first rollout stage
    binom_pmf(0, sessions, crash_rate)       # 0.13506452244668338
    sims = rng.binomial(sessions, crash_rate, size=100_000)
    (sims == 0).mean()                       # 0.13364   the simulation agrees
    np.bincount(sims)[:6] / 100_000          # array([0.13364, 0.26978, 0.27066, 0.18165, 0.09106, 0.03661])
    ```

    A 13.5% chance of a perfectly clean dashboard, for a build that crashes twice per thousand sessions on average. "Zero crashes in stage one" is weak evidence when the stage is small; Crashlytics' crash-free percentage only becomes trustworthy once \( np \), the number of crashes you'd *expect*, is comfortably above a handful. Week 7 turns this into confidence intervals.

## Continuous variables and probability density

A die has six faces you can list. A waiting time, an app's cold-start time or a model weight can be **any real number** in a range. Here's the surprise: for a continuous variable, the probability of hitting one exact value is **zero**.

The simplest continuous distribution shows why. The **uniform distribution** on \( [a, b] \) says every value between \( a \) and \( b \) is equally likely. A delay picked uniformly between 0 and 10 minutes:

```python
u = rng.uniform(0, 10, size=100_000)       # 100,000 delays, anywhere in [0, 10) minutes
u[:5].round(3)                             # array([5.848, 7.607, 1.402, 6.063, 0.942])
(u == 3.0).sum()                           # 0   not one of 100,000 is exactly 3.0
((u >= 2) & (u <= 5)).mean()               # 0.30004
(5 - 2) / (10 - 0)                         # 0.3
```

There are infinitely many values between 0 and 10, so any single one gets a share of zero. What *does* have a probability is a **range**: "between 2 and 5 minutes" covers 3 of the 10 minutes, so its probability is \( 3/10 = 0.3 \).

### Density: probability per unit of \( x \)

So we describe a continuous variable with a **probability density function** (PDF), written \( f(x) \). Its height is not a probability; it's **probability per unit of \( x \)**, like population density is people per square kilometre. To get a probability you multiply by a width, and for a curve that means taking the **area** under it:

\[
P(a \le X \le b) = \text{area under } f(x) \text{ between } a \text{ and } b
\qquad
\text{total area} = 1
\]

(In calculus notation that area is the integral \( \int_a^b f(x)\,dx \); you don't need integrals today, only the idea of area.)

For the uniform distribution the density is flat, a rectangle of width \( b - a \). Its area must be 1, so its height must be

\[
f(x) = \frac{1}{b - a} \quad \text{for } a \le x \le b, \qquad 0 \text{ elsewhere}
\]

**Worked example.** Uniform on \( [0, 10] \): \( f(x) = 1/10 = 0.1 \) everywhere in the range. \( P(2 \le X \le 5) \) is a rectangle of width 3 and height 0.1: area 0.3, matching the simulation. Its mean is the middle, \( (a + b)/2 = 5 \), and its variance is \( (b - a)^2/12 \) (a standard result; here 100/12 ≈ 8.33):

```python
u.mean().round(3), u.var().round(3), (10 - 0)**2 / 12   # (5.003, 8.353, 8.333333333333334)
```

The animation shows where densities come from. Make a histogram where each bar's height is (fraction of values in the bar) / (bar width); then each bar's **area** is the fraction in it. As the bars get narrower, their tops trace out the density curve.

<figure class="diagram" markdown>
![Animation: on axes from -3 to 3, a histogram of the standard normal with 4 wide blue bars, then 8, then 16 narrower bars whose tops approach a red bell curve; finally the bars vanish and the area under the red curve between -1 and 1 is shaded yellow and labelled 0.683, while the side panel explains that bar height is the chance of landing in the bar divided by its width, that the height at 0, 0.399, is a density not a probability, and that P(X = exactly 0.5) is 0 because a line has no width](../../assets/img/phase-1/day37-density-area.svg)
<figcaption>Animated: bars of probability-per-width become a density curve as they narrow. The probability of a range is the area under the curve.</figcaption>
</figure>

`np.histogram(..., density=True)` computes exactly those heights. A uniform on \( [0, 0.5] \) shows the punchline: its density is \( 1/0.5 = 2 \), **higher than 1**, which is fine, because it's not a probability:

```python
small = rng.uniform(0, 0.5, size=100_000)
heights, edges = np.histogram(small, bins=10, density=True)
heights.round(2)        # array([1.98, 1.99, 2.  , 2.03, 1.97, 2.02, 1.99, 2.  , 1.99, 2.04])
edges.round(2)          # array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ])
(heights * np.diff(edges)).sum()          # 1.0   the AREAS add up to 1, not the heights
```

!!! warning "Common mistake: reading a density's height as a probability"
    "The density at 0.2 is 2.0" does not mean a 200% chance. Heights can be any positive number; only **areas** are probabilities, and the total area is always 1. The narrower the range a variable lives in, the taller its density has to be. Neural-network libraries report log-densities of continuous outputs that can be positive for the same reason.

## The normal distribution

Measure your app's cold-start time on one device a thousand times. Most runs land near the average, a few are noticeably slower or faster, and very few are far off in either direction. Plot them and you get the **bell curve**: the **normal** (or **Gaussian**) distribution. It shows up whenever a value is the sum of many small independent effects (thread scheduling, disk caches, CPU frequency, ...); Day 40's central limit theorem explains why.

It has two parameters:

- \( \mu \) (Greek "mu"): the **mean**, the centre of the bell. Changing it slides the curve left or right.
- \( \sigma \) (Greek "sigma"): the **standard deviation**, the typical distance from the mean. Changing it makes the bell wider and flatter or narrower and taller. \( \sigma^2 \) is the variance.

Its density is

\[
f(x) = \frac{1}{\sigma\sqrt{2\pi}}\; \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
\]

Reading it from the inside out:

- \( (x - \mu)^2 \): the squared distance from the centre. It is 0 at the centre and grows in both directions equally, which makes the curve symmetric.
- Dividing by \( 2\sigma^2 \) measures that distance in units of the spread: being 50 ms away matters a lot if \( \sigma = 10 \) ms and hardly at all if \( \sigma = 500 \) ms.
- \( \exp(-\,\cdot\,) \) is \( e^{-(\cdot)} \) (Day 30): 1 when the distance is 0, then falling quickly towards 0 the further you go. That's the bell shape.
- \( \frac{1}{\sigma\sqrt{2\pi}} \) is the constant that makes the total area exactly 1. (\( \pi \approx 3.14159 \); why it appears is a lovely story, but not needed today.)

The **standard normal** is the one with \( \mu = 0 \), \( \sigma = 1 \).

**Worked example.** Standard normal at \( x = 0 \): the exponent is 0, \( e^0 = 1 \), so \( f(0) = 1/\sqrt{2\pi} = 1/2.5066 \approx 0.3989 \). At \( x = 1 \): \( e^{-1/2} \approx 0.6065 \), so \( f(1) \approx 0.3989 \times 0.6065 \approx 0.2420 \). At \( x = 2 \): \( e^{-2} \approx 0.1353 \), so \( f(2) \approx 0.0540 \). For cold starts with \( \mu = 800 \) ms and \( \sigma = 50 \) ms the peak is \( 0.3989/50 \approx 0.00798 \) per millisecond: wider bell, lower peak, same area.

```python
def normal_pdf(x, mu=0.0, sigma=1.0):
    """Height of the normal (Gaussian) density at x."""
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

normal_pdf(0.0)                     # 0.3989422804014327
normal_pdf(1.0)                     # 0.24197072451914337
normal_pdf(2.0)                     # 0.05399096651318806
normal_pdf(800, mu=800, sigma=50)   # 0.007978845608028654

xs = np.linspace(-8, 8, 160_001)    # a fine grid of x values
dx = xs[1] - xs[0]
(normal_pdf(xs) * dx).sum()         # 0.9999999999976683   thin rectangles: total area is 1
```

That last line is the density idea in code: 160,000 rectangles of width `dx` under the curve, areas summed.

### The 68–95–99.7 rule

For **every** normal distribution, whatever \( \mu \) and \( \sigma \) are:

- about **68%** of values lie within 1 \( \sigma \) of the mean,
- about **95%** within 2 \( \sigma \),
- about **99.7%** within 3 \( \sigma \).

<figure class="diagram" markdown>
![Animation: a red bell curve over an axis marked mu - 3 sigma to mu + 3 sigma and z = -3 to 3; first the area between mu - sigma and mu + sigma is shaded indigo with a bracket labelled within 1 sigma: 68.3%, then the band out to 2 sigma is added in teal labelled 95.4%, then out to 3 sigma in orange labelled 99.7%, and a note says outside 2 sigma is about 1 in 22 and outside 3 sigma about 1 in 370; the corner reminds that with mu = 800 ms and sigma = 50 ms, 68% of cold starts land in 750 to 850 ms](../../assets/img/phase-1/day37-68-95-997.svg)
<figcaption>Animated: the shares within 1, 2 and 3 standard deviations are the same for every normal distribution.</figcaption>
</figure>

Check it on simulated cold starts:

```python
mu, sigma = 800, 50                        # cold-start time in ms (made-up numbers)
starts = rng.normal(mu, sigma, size=100_000)
starts[:5].round(1)                        # array([789.9, 770.5, 708.9, 727.1, 784.2])
starts.mean().round(2), starts.std().round(2)   # (799.96, 49.93)
for k in (1, 2, 3):
    inside = (np.abs(starts - mu) <= k * sigma).mean()
    print(f"within {k} sigma ({mu - k*sigma}..{mu + k*sigma} ms): {inside:.4f}")
# within 1 sigma (750..850 ms): 0.6828
# within 2 sigma (700..900 ms): 0.9550
# within 3 sigma (650..950 ms): 0.9972
```

### z-scores: "how many σ from the mean?"

A cold start of 920 ms: is that bad? The **z-score** answers in units of standard deviations:

\[
z = \frac{x - \mu}{\sigma}
\qquad
z = \frac{920 - 800}{50} = 2.4
\]

2.4 \( \sigma \) above the mean is outside the 95% band: unusual. This is exactly Day 22's **standardization** (subtract the mean, divide by the standard deviation). Standardizing a normal variable gives a standard normal, so one table, one rule and one CDF serve every normal distribution:

```python
z = (920 - mu) / sigma
z                                          # 2.4
zs = (starts - mu) / sigma                 # standardize every sample
zs.mean().round(3), zs.std().round(3)      # (-0.001, 0.999)   mean 0, standard deviation 1
```

!!! warning "Common mistake: assuming everything is normal"
    The 68–95–99.7 rule is a fact about normal distributions, not about data. App installs, review counts and latencies with rare long stalls are **skewed**: a long tail on one side. Ratings are squashed against a hard ceiling of 5. Check with a histogram before you lean on the rule; the real-data section below does exactly that.

## The CDF: "how much is at or below \( x \)?"

The **cumulative distribution function** (CDF) is the running total of probability from the left:

\[
F(x) = P(X \le x)
\]

- For a discrete variable it's a **staircase**: it jumps up by \( P(X = k) \) at every possible value \( k \).
- For a continuous variable it's a smooth **S-curve**: the area under the density to the left of \( x \).
- It always climbs from 0 to 1 and never goes down.

It turns every range question into one subtraction:

\[
P(a < X \le b) = F(b) - F(a)
\]

The normal CDF has no formula made of ordinary functions, but Python's `math` module has the **error function** `math.erf`, which computes exactly this kind of area. For the standard normal, \( \Phi(z) = \frac{1}{2}\left(1 + \operatorname{erf}(z/\sqrt{2})\right) \), where \( \Phi \) (Greek capital "phi") is the usual name for the standard normal's CDF:

```python
def normal_cdf(x, mu=0.0, sigma=1.0):
    """P(X <= x) for a normal variable, using the error function from the math module."""
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return 0.5 * (1 + np.vectorize(math.erf)(z / np.sqrt(2)))

normal_cdf(0.0)                            # 0.5   half the bell is left of the centre
normal_cdf(1.0) - normal_cdf(-1.0)         # 0.6826894921370859
normal_cdf(2.0) - normal_cdf(-2.0)         # 0.9544997361036416
normal_cdf(3.0) - normal_cdf(-3.0)         # 0.9973002039367398
1 - normal_cdf(920, mu=800, sigma=50)      # 0.008197535924596155   P(start slower than 920 ms)
(starts > 920).mean()                      # 0.00803
```

The exact 68.27%, 95.45% and 99.73% are where the rule's round numbers come from. (`np.vectorize` lets the scalar `math.erf` accept a whole array.)

For data you don't have a formula for, use the **empirical CDF**: the fraction of samples at or below \( x \). The inverse question, "below which value do 95% of starts fall?", is the **95th percentile**, or **quantile** 0.95. You know it as "p95 latency":

```python
def ecdf(samples, x):
    """Empirical CDF: the fraction of samples that are <= x."""
    return np.sort(samples).searchsorted(x, side="right") / len(samples)

ecdf(starts, 850), normal_cdf(850, 800, 50)   # (0.84137, 0.8413447460685429)
np.quantile(starts, 0.95).round(1)            # 881.8   p95 of the simulated starts
mu + 1.6449 * sigma                           # 882.245   for a normal, p95 = mu + 1.645 sigma
```

The figure puts the three views side by side: a staircase CDF for the binomial, and the density and CDF of the cold-start normal, where the shaded area on the left *is* the height of the dot on the right.

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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.6))
ks = np.arange(6)
ax1.step(np.r_[-1, ks, 6], np.r_[0, pmf.cumsum(), 1], where="post", color="#3949ab", lw=2)
ax1.scatter(ks, pmf.cumsum(), color="#3949ab", zorder=3)
ax1.set_title("Binomial(5, 0.2) CDF: a staircase")
ax1.set_xlabel("k taps")
ax1.set_ylabel("P(K ≤ k)")

t = np.linspace(600, 1000, 400)
ax2.plot(t, normal_pdf(t, mu, sigma), color="#e53935", lw=2)
left = t <= 850
ax2.fill_between(t[left], normal_pdf(t[left], mu, sigma), color="#fff59d")
ax2.text(785, 0.0012, "area = 0.841", ha="center", color="#ef6c00", fontweight="bold")
ax2.set_title("Normal(800, 50) density")
ax2.set_xlabel("cold start (ms)")
ax2.set_ylabel("density (per ms)")

ax3.plot(t, normal_cdf(t, mu, sigma), color="#e53935", lw=2, label="formula")
ax3.step(np.sort(starts[:300]), np.arange(1, 301) / 300, where="post",
         color="#00897b", lw=1, label="empirical, 300 samples")
ax3.scatter([850], [normal_cdf(850, mu, sigma)], color="#ef6c00", zorder=3, s=40)
ax3.annotate("F(850) = 0.841", (850, 0.841), xytext=(870, 0.55), color="#ef6c00",
             arrowprops=dict(arrowstyle="-", color="#ef6c00"))
ax3.set_title("Normal(800, 50) CDF: an S-curve")
ax3.set_xlabel("cold start (ms)")
ax3.set_ylabel("P(X ≤ x)")
ax3.legend(frameon=False, loc="lower right", fontsize=9)
for ax in (ax1, ax2, ax3):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("day37-pdf-cdf.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Three plots: left, the CDF of a binomial with n = 5 and p = 0.2 as a blue staircase jumping at k = 0 to 5 from 0.33 up to 1; middle, the red bell-shaped density of a normal with mean 800 and standard deviation 50 ms, with the area left of 850 shaded yellow and labelled area = 0.841; right, the matching S-shaped red CDF with a thin teal staircase from 300 samples hugging it and an orange dot at F(850) = 0.841](../../assets/img/phase-1/day37-pdf-cdf.svg)
<figcaption>A discrete CDF is a staircase, a continuous one an S-curve. The yellow area in the middle panel equals the height of the orange dot on the right.</figcaption>
</figure>

## Sampling each distribution and checking the formula

Every distribution today has a NumPy sampler. Here is the habit worth building: **draw samples, draw the formula on top, and look**. For discrete distributions compare frequencies with the PMF; for continuous ones use `density=True` so the histogram is on the same scale as the PDF.

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))

ax = axes[0, 0]                                     # Bernoulli(0.1)
b = (rng.random(10_000) < 0.1).astype(int)
ax.bar([0, 1], np.bincount(b, minlength=2) / len(b), width=0.5, color="#c5cae9", label="10,000 samples")
ax.scatter([0, 1], [0.9, 0.1], color="#e53935", zorder=3, s=50, label="formula")
ax.set_xticks([0, 1], ["0: no tap", "1: tap"])
ax.set_title("Bernoulli(p = 0.1)")
ax.set_ylabel("probability")
ax.legend(frameon=False)

ax = axes[0, 1]                                     # Binomial(20, 0.3)
k20 = np.arange(21)
c20 = rng.binomial(20, 0.3, size=10_000)
ax.bar(k20, np.bincount(c20, minlength=21) / len(c20), color="#c5cae9", label="10,000 samples")
ax.plot(k20, [binom_pmf(k, 20, 0.3) for k in k20], "o-", color="#e53935", ms=4, label="formula")
ax.set_title("Binomial(n = 20, p = 0.3)")
ax.set_xlabel("successes k")
ax.legend(frameon=False)

ax = axes[1, 0]                                     # Uniform(0, 10)
ax.hist(rng.uniform(0, 10, size=10_000), bins=20, density=True, color="#b2dfdb", label="10,000 samples")
ax.plot([0, 0, 10, 10], [0, 0.1, 0.1, 0], color="#e53935", lw=2, label="f(x) = 1/10")
ax.set_ylim(0, 0.14)
ax.set_title("Uniform(0, 10)")
ax.set_xlabel("minutes")
ax.set_ylabel("density")
ax.legend(frameon=False, loc="upper center", ncol=2)

ax = axes[1, 1]                                     # Normal(800, 50)
ax.hist(rng.normal(800, 50, size=10_000), bins=40, density=True, color="#b2dfdb", label="10,000 samples")
ax.plot(t, normal_pdf(t, 800, 50), color="#e53935", lw=2, label="formula")
ax.set_title("Normal(μ = 800, σ = 50)")
ax.set_xlabel("cold start (ms)")
ax.legend(frameon=False)

for ax in axes.flat:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("day37-sample-vs-formula.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Four plots, each with light bars from 10,000 samples and the red formula on top: Bernoulli with p = 0.1 has bars of about 0.9 and 0.1 matching two red dots; binomial with n = 20 and p = 0.3 has a hump peaking at k = 6 traced by red dots; uniform on 0 to 10 has flat bars around height 0.1 under a red rectangle; normal with mean 800 and sigma 50 has a bell of bars under a red bell curve](../../assets/img/phase-1/day37-sample-vs-formula.svg)
<figcaption>Samples (bars) against formulas (red) for all four distributions. With 10,000 draws the two agree to within a few percent everywhere.</figcaption>
</figure>

Notice the binomial with \( n = 20 \) already looks like a bell. That is no accident: a sum of many independent pieces drifts towards a normal shape, the central limit theorem of Day 40.

## Real data: is it normal?

The Play Store dataset gives two honest examples. Load it with the short cleaning from Days 17–19:

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")
apps = apps.drop_duplicates().assign(
    category=lambda d: d["category"].str.strip().str.title(),
    installs=lambda d: d["installs"].str.replace(",", "").str.rstrip("+").astype(int),
)
apps.shape     # (480, 12)
```

No internet? Download the file from the [Datasets page](../../datasets.md) and pass the local path to `read_csv`. (Tested on pandas 3.0.)

Raw review counts are wildly skewed (Day 19), but their `log10` (3 means 1,000 reviews, 6 means a million) is a different story. Compare both columns against the 68–95–99.7 rule:

```python
log_reviews = np.log10(apps["reviews"])
ratings = apps["rating"].dropna()
for name, col in [("log10(reviews)", log_reviews), ("rating", ratings)]:
    m, s = col.mean(), col.std()
    shares = [((col - m).abs() <= k * s).mean() for k in (1, 2, 3)]
    print(f"{name:>15}: mean {m:.2f}, std {s:.2f}, within 1/2/3 std: "
          + ", ".join(f"{x:.3f}" for x in shares))
#  log10(reviews): mean 3.20, std 0.95, within 1/2/3 std: 0.685, 0.958, 0.994
#          rating: mean 4.17, std 0.45, within 1/2/3 std: 0.641, 0.981, 1.000
```

`log10(reviews)` matches the rule closely: 68.5%, 95.8%, 99.4%. Ratings don't: only 64% within one standard deviation, and **nothing** beyond three, because a normal with mean 4.17 and \( \sigma = 0.45 \) would put about 3.3% of apps above 5 stars, which is impossible:

```python
1 - normal_cdf(5.0, ratings.mean(), ratings.std())   # 0.032996718864315655   what a normal fit expects above 5
(ratings > 5).mean()                                 # 0.0      what the data has
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, col, label, bins in [(ax1, log_reviews, "log10(reviews)", 20), (ax2, ratings, "rating (stars)", 22)]:
    m, s = col.mean(), col.std()
    ax.hist(col, bins=bins, density=True, color="#c5cae9", edgecolor="white", label=f"{len(col)} apps")
    g = np.linspace(m - 4 * s, m + 4 * s, 300)
    ax.plot(g, normal_pdf(g, m, s), color="#e53935", lw=2, label="normal, same mean and std")
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)
ax1.set_title("log10(reviews): close to normal")
ax1.legend(frameon=False, fontsize=9, loc="upper left")
ax2.axvline(5, color="#555", ls="--", lw=1)
ax2.text(5.05, 0.75, "5-star\nceiling", color="#555", fontsize=9)
ax2.set_title("Rating: capped at 5 stars")
fig.tight_layout()
fig.savefig("day37-playstore-normal.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Two histograms with a red normal curve fitted to each: on the left, log10 of review counts for 480 apps forms a roughly symmetric hump that the red bell follows well; on the right, ratings of 462 apps pile up between 4 and 5 and stop at a dashed 5-star ceiling, while the red bell spills past 5 where there is no data](../../assets/img/phase-1/day37-playstore-normal.svg)
<figcaption>Log review counts are close to normal; ratings are not: the fitted bell puts probability above 5 stars, where no app can be.</figcaption>
</figure>

## Where distributions show up in ML

You'll meet these four in almost every model you train.

**Weight initialization.** Before training, every weight in a network is drawn at random, usually from a normal or a uniform distribution. The spread matters enormously. Push a batch through 10 ReLU layers of width 512 (Day 12), with weights drawn three ways:

```python
fan_in = 512                                     # inputs per neuron
X0 = rng.normal(size=(1000, fan_in))             # a batch of 1000 inputs
for name, std in [("std 1.0", 1.0), ("std 0.01", 0.01), ("He sqrt(2/512)", np.sqrt(2 / fan_in))]:
    h = X0
    for layer in range(10):                      # 10 ReLU layers
        W = rng.normal(0.0, std, size=(fan_in, fan_in))
        h = np.maximum(0, h @ W)
    print(f"{name:>15}: spread of layer-10 outputs = {h.std():.3g}")
#         std 1.0: spread of layer-10 outputs = 8.71e+11
#        std 0.01: spread of layer-10 outputs = 9.32e-09
#  He sqrt(2/512): spread of layer-10 outputs = 0.867
```

Too wide and the values explode (each layer multiplies the spread by about 16); too narrow and they vanish towards 0. Gradients suffer the same fate during training, so the network can't learn. **He initialization**, a normal with \( \sigma = \sqrt{2/\text{fan\_in}} \), is chosen so the spread stays roughly constant through ReLU layers. PyTorch's `nn.Linear` uses a uniform distribution instead: weights from \( U(-1/\sqrt{\text{fan\_in}},\ 1/\sqrt{\text{fan\_in}}) \) by default. Today's uniform variance formula, \( (b - a)^2/12 \), tells you its spread:

```python
bound = 1 / np.sqrt(fan_in)
W = rng.uniform(-bound, bound, size=(256, fan_in))     # an nn.Linear(512, 256)-style weight matrix
W.var(), (2 * bound)**2 / 12, 1 / (3 * fan_in)
# (0.0006515166991089825, 0.0006510416666666665, 0.0006510416666666666)
```

**Dropout.** During training, dropout switches off each neuron at random: a Bernoulli mask per value, with "keep" probability, say, 0.8. Kept values are divided by 0.8 so the expected total doesn't change.

```python
keep = 0.8
mask = rng.random((2, 10)) < keep        # one Bernoulli(0.8) per activation
mask.astype(int)
# array([[1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
#        [1, 1, 1, 0, 1, 1, 0, 1, 0, 1]])
mask.mean()                              # 0.7   14 of 20 kept this time; 0.8 on average
```

**Output probabilities.** A binary classifier (spam or not, Day 41) ends with a sigmoid (Day 30) that outputs a number like 0.83. That number *is* the parameter \( p \) of a Bernoulli distribution over the label: the model is saying "I think this is a 'yes' with probability 0.83". Training with cross-entropy (Days 47 and 54) means making the observed labels as likely as possible under those Bernoullis. A multi-class model's softmax outputs a **categorical** distribution, the Bernoulli generalized to more than two outcomes, and a language model generates text by sampling from one at every step.

```python
p_spam = 0.83                            # a classifier's output for one message
labels = rng.random(10_000) < p_spam     # what the model "expects" over many such messages
labels.mean()                            # 0.8261
```

**Noise.** Linear regression (Day 55) assumes each target is the true line plus **normal noise**; Day 47 shows that this assumption is exactly why the loss is the mean squared error. Data augmentation adds random noise to inputs, and diffusion models generate images by learning to remove normal noise step by step.

!!! android "Android bridge: Bernoulli and normal draws in Kotlin"
    The `rng.random() < p` trick is the standard way to sample a fraction of anything on device, such as logging detailed traces for 10% of sessions:

    ```kotlin
    import kotlin.random.Random

    val sampleThisSession = Random.nextDouble() < 0.10   // one Bernoulli(0.1) trial per session
    if (sampleThisSession) startDetailedTracing()
    ```

    Out of \( n \) sessions, the number traced is then binomial(\( n \), 0.1), so expect about \( 0.1n \), give or take \( \sqrt{n \cdot 0.1 \cdot 0.9} \). For a normal draw, `java.util.Random().nextGaussian()` returns a standard normal sample; `mu + sigma * z` turns it into any normal, the z-score formula run backwards. You won't initialize weights on the phone, though: a LiteRT (TensorFlow Lite) model ships with its trained weights already fixed, and dropout is a training-only operation that does nothing at inference.

---

## Exercises

1. By hand, compute \( P(K = 2) \) for a binomial with \( n = 4 \), \( p = 0.5 \) (the Galton board). Then compute all five probabilities with `binom_pmf` and check them with `rng.binomial`.
2. Roll a fair die 10 times. What is the probability of **at least one** six? Answer it three ways: with the complement rule (Day 36), by summing `binom_pmf` over \( k = 1..10 \), and by simulation.
3. \( X \) is uniform on \( [0, 0.25] \). What is its density? What is \( P(0.1 \le X \le 0.2) \)? Explain why a density of 4 isn't a 400% chance. Verify the probability by simulation.
4. A test score is normal with \( \mu = 100 \), \( \sigma = 15 \). What fraction scores above 130? Between 85 and 115? Use z-scores and `normal_cdf`, then check with 100,000 samples.
5. In the Play Store data, compute the z-score of `log10(reviews)` for the app with the most reviews. Using the normal approximation, how many of the 480 apps would you *expect* to be at least that far above the mean? (Use `apps["reviews"].idxmax()`.)
6. Glorot (Xavier) uniform initialization draws weights from \( U(-r, r) \) with \( r = \sqrt{6/(\text{fan\_in} + \text{fan\_out})} \). Show with the variance formula that the variance is \( 2/(\text{fan\_in} + \text{fan\_out}) \), then check by sampling a \( 512 \times 256 \) matrix.

??? success "Solutions"
    ```python
    # 1: C(4, 2) = 6 arrangements, each 0.5^2 * 0.5^2 = 1/16, so 6/16 = 0.375
    [binom_pmf(k, 4, 0.5) for k in range(5)]    # [0.0625, 0.25, 0.375, 0.25, 0.0625]
    g = rng.binomial(4, 0.5, size=100_000)
    (np.bincount(g, minlength=5) / len(g)).round(4)   # array([0.0626, 0.2517, 0.3739, 0.2482, 0.0635])
    ```
    The probabilities are 1, 4, 6, 4, 1 sixteenths, the bin counts in the Galton animation.

    ```python
    # 2
    1 - (5/6)**10                                       # 0.8384944171101543   complement: 1 - P(no six)
    sum(binom_pmf(k, 10, 1/6) for k in range(1, 11))    # 0.8384944171101545   same, up to rounding
    rolls = rng.integers(1, 7, size=(100_000, 10))      # 100,000 runs of 10 rolls
    (rolls == 6).any(axis=1).mean()                     # 0.83878
    ```
    The complement is the quick way: "at least one" = 1 − "none", and "none" is \( (5/6)^{10} \).

    ```python
    # 3: density = 1 / (0.25 - 0) = 4; P = width 0.1 x height 4 = 0.4
    x3 = rng.uniform(0, 0.25, size=100_000)
    ((x3 >= 0.1) & (x3 <= 0.2)).mean()                  # 0.4011
    ```
    The density is 4 per unit of \( x \), but the variable only lives on a range 0.25 wide, so the total area is \( 4 \times 0.25 = 1 \). Only areas are probabilities.

    ```python
    # 4: 130 is z = (130 - 100)/15 = 2; 85..115 is z = -1..1
    1 - normal_cdf(130, 100, 15)                            # 0.02275013194817921
    normal_cdf(115, 100, 15) - normal_cdf(85, 100, 15)      # 0.6826894921370859
    scores = rng.normal(100, 15, size=100_000)
    (scores > 130).mean(), ((scores > 85) & (scores < 115)).mean()   # (0.02282, 0.67891)
    ```
    About 2.3% above 130 (half of the 4.6% outside 2 σ), and the 68% band between 85 and 115.

    ```python
    # 5
    top = apps["reviews"].idxmax()
    print(apps.loc[top, "app_name"], apps.loc[top, "reviews"])   # Orbit Lens Lite 2525379
    z5 = (log_reviews[top] - log_reviews.mean()) / log_reviews.std()
    z5.round(2)                                         # 3.36
    (480 * (1 - normal_cdf(z5))).round(2)               # 0.19
    ```
    The normal approximation expects about 0.19 apps that far out: seeing one is a bit unusual, but in a dataset of 480 a single app at 3.4 σ is not shocking. (`log_reviews` keeps the same index labels as `apps`, so `log_reviews[top]` picks the same app.)

    ```python
    # 6: variance of U(-r, r) = (2r)^2 / 12 = r^2 / 3 = (6 / (in + out)) / 3 = 2 / (in + out)
    fan_out = 256
    r = np.sqrt(6 / (fan_in + fan_out))
    W6 = rng.uniform(-r, r, size=(fan_in, fan_out))
    W6.var(), 2 / (fan_in + fan_out)                    # (0.002613536423212279, 0.0026041666666666665)
    ```

## Resources

- [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/): interactive visual chapters on probability, random variables and distributions.
- [OpenIntro Statistics (free textbook)](https://www.openintro.org/book/os/): chapter 4 covers the normal, binomial and Bernoulli distributions with worked examples and exercises.
- [Khan Academy: Statistics and probability](https://www.khanacademy.org/math/statistics-probability): the "Random variables" and "Modeling data distributions" units cover the binomial, density curves, z-scores and the 68–95–99.7 rule, with practice.
- [Dive into Deep Learning: Probability and Statistics](https://d2l.ai/chapter_preliminaries/probability.html): the same ideas from an ML point of view, with code.
- [NumPy docs: random `Generator`](https://numpy.org/doc/stable/reference/random/generator.html): every distribution sampler used today (`random`, `binomial`, `uniform`, `normal`) and many more.
- [CS231n notes: weight initialization](https://cs231n.github.io/neural-networks-2/): why the initialization spread matters, with the "calibrate the variance with 1/sqrt(n)" argument.
- [PyTorch docs: `torch.nn.init`](https://pytorch.org/docs/stable/nn.init.html): the uniform and normal initializers (Xavier/Glorot, Kaiming/He) you'll call in Phase 3.

## Checklist

- [ ] I can explain the difference between a discrete distribution (PMF) and a continuous one (PDF).
- [ ] I can simulate a Bernoulli trial with `rng.random() < p` and know its mean is \( p \).
- [ ] I can compute a binomial probability by hand, including the \( \binom{n}{k} \), and confirm it by simulation.
- [ ] I know why \( P(X = \text{exact value}) = 0 \) for a continuous variable, and that probability is the area under the density.
- [ ] I can write the uniform and normal densities and explain what \( \mu \) and \( \sigma \) do.
- [ ] I can apply the 68–95–99.7 rule and compute a z-score, and I check a histogram before assuming data is normal.
- [ ] I can use a CDF to get \( P(a < X \le b) = F(b) - F(a) \) and a percentile such as p95.
- [ ] I can name where distributions appear in ML: initialization, dropout, noise and output probabilities.

**Next:** [Day 38: Conditional probability and Bayes' rule](day-38-bayes-rule.md)
