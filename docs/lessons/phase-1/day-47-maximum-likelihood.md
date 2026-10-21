# Day 47: Maximum Likelihood Estimation

**Phase 1 · Week 7 · ~120 minutes**

## Objectives

- [ ] Explain the difference between a probability (the knob is fixed, the data varies) and a likelihood (the data is fixed, the knob varies).
- [ ] Write the likelihood of independent data as a product, and explain why we always work with its logarithm instead.
- [ ] Derive the maximum likelihood estimate of a coin's \( p \) with a derivative, get \( \hat{p} = k/n \), and check it with a grid search.
- [ ] Derive the maximum likelihood estimates of a normal distribution's mean and standard deviation, and say why the variance estimate divides by \( n \).
- [ ] Show that minimizing the mean squared error **is** maximum likelihood when the noise is Gaussian, and that cross-entropy **is** maximum likelihood for a yes/no classifier.
- [ ] Find a maximum likelihood estimate numerically with a grid search when there is no formula.
- [ ] Explain MAP estimation, and show that a Gaussian prior on a weight turns the maximum likelihood objective into "MLE plus an L2 penalty".

**Prerequisites.** [Day 30](day-30-derivative-rules.md) (the derivative of \( \ln x \)), [Day 31](day-31-chain-rule.md) (the chain rule), [Day 33](day-33-vector-matrix-gradients.md) (the MSE of a line fit), [Day 37](day-37-distributions.md) (Bernoulli, binomial and normal distributions), [Day 38](day-38-bayes-rule.md) (prior, likelihood, posterior) and [Day 43](day-43-sampling.md) (dividing by \( n - 1 \)). Day 22's [preview of regularization](day-22-norms-distances.md) comes back at the end.

---

## The big idea

You find a coin on the street and flip it 10 times: 7 heads, 3 tails. What is the coin's chance of heads, \( p \)?

You can't know for sure. A fair coin (\( p = 0.5 \)) can throw 7 heads; so can a coin with \( p = 0.9 \). But some values of \( p \) make what you saw **less surprising** than others. With \( p = 0.1 \), seven heads would be a near-miracle. With \( p = 0.7 \), it's the single most typical outcome. **Maximum likelihood estimation** (MLE) is the rule "pick the parameter value under which the data you actually observed was most probable".

**Analogy for an app developer.** A bug report says "the app froze for 4 seconds on a Pixel 6 after rotating the screen". You have three theories: a main-thread database query, a big bitmap decode, a network call on the UI thread. You don't ask which theory you *like*; you ask which one **would most likely produce exactly this report**. The theory that makes the evidence least surprising wins. That is the whole method; today we make it precise and turn it into calculus.

It matters far more than estimating coins. **Almost every loss function you will train with is a maximum likelihood objective in disguise.** Mean squared error is maximum likelihood with Gaussian noise; cross-entropy is maximum likelihood for classification; and adding L2 regularization is the same thing with a prior belief on top. By the end of today, those three formulas stop being arbitrary recipes and become one idea.

## Probability versus likelihood

Day 37 gave you the binomial formula: the probability of \( k \) heads in \( n \) flips of a coin with heads-probability \( p \) is

\[
P(K = k \mid p) = \binom{n}{k}\, p^k\, (1 - p)^{n - k}
\]

The bar \( \mid \) reads "given" (Day 38): "the probability of \( k \) heads, given that the coin's heads-chance is \( p \)". That one formula has two inputs, the data \( k \) and the parameter \( p \), and you can use it to answer two different questions:

| | What is fixed | What varies | Question | Name |
|---|---|---|---|---|
| **Probability** | the parameter \( p \) | the data \( k \) | "If the coin is fair, how likely are 7 heads? 8? 2?" | the pmf, \( P(k \mid p) \) |
| **Likelihood** | the data \( k \) | the parameter \( p \) | "I *saw* 7 heads. How well does \( p = 0.3 \) explain that? \( p = 0.7 \)?" | the likelihood, \( L(p) \) |

So the **likelihood function** is the same expression read the other way round:

\[
L(p) = P(\text{the data I observed} \mid p)
\]

with the data held fixed and \( p \) as the variable. The animation below shows both readings at once. On the left, for each setting of the knob \( p \), the full pmf over \( k = 0, \dots, 10 \): those bars always add up to 1. On the right, only the height of the bar for the data we actually saw (\( k = 7 \)), plotted against \( p \): that is the likelihood curve.

<figure class="diagram" markdown>
![Animation: on the left, binomial probability bars for k = 0 to 10 heads in 10 flips, redrawn for p = 0.3, 0.5, 0.9 and 0.7, with the bar for k = 7 in orange; a dashed line carries the height of the orange bar (0.009, 0.117, 0.057, 0.267) across to a dot above that p on the right-hand axes; finally a red curve through the dots appears, peaking at p = 0.7 with the label highest at p-hat = 0.7 = 7/10](../../assets/img/phase-1/day47-likelihood-knob.svg)
<figcaption>Animated: for each knob setting p, read off the height of the bar for the data you saw (7 heads). Those heights, plotted against p, form the likelihood curve, highest at p = 0.7.</figcaption>
</figure>

```python
import numpy as np
from math import comb

def binom_pmf(k, n, p):
    """P(K = k | p) for a binomial count: Day 37's formula. Works for an array of p too."""
    return comb(n, k) * p**k * (1 - p)**(n - k)

# Probability: the knob is fixed at p = 0.5, and we ask about every possible outcome k.
probs = np.array([binom_pmf(k, 10, 0.5) for k in range(11)])
probs.round(4)
# array([0.001 , 0.0098, 0.0439, 0.1172, 0.2051, 0.2461, 0.2051, 0.1172,
#        0.0439, 0.0098, 0.001 ])
probs.sum()          # 1.0

# Likelihood: the outcome is fixed at k = 7, and we ask about different knob settings p.
for p in (0.3, 0.5, 0.7, 0.9):
    print(f"p = {p}:  L(p) = {binom_pmf(7, 10, p):.4f}")
# p = 0.3:  L(p) = 0.0090
# p = 0.5:  L(p) = 0.1172
# p = 0.7:  L(p) = 0.2668
# p = 0.9:  L(p) = 0.0574
```

The pmf column adds up to 1, as every probability distribution must. The likelihood values do not have to add up to anything. In fact the area under the whole likelihood curve, over all \( p \) from 0 to 1, is not 1:

```python
p_fine = np.linspace(0, 1, 100_001)
dp = p_fine[1] - p_fine[0]
(binom_pmf(7, 10, p_fine) * dp).sum()     # 0.09090909090909091   thin rectangles (Day 37): 1/11, not 1
```

!!! warning "Common mistake: reading a likelihood as the probability of the parameter"
    "\( L(0.7) = 0.267 \), so there is a 26.7% chance the coin's \( p \) is 0.7" is wrong. 0.267 is the probability of **the data** (7 heads) if \( p \) were 0.7. It says nothing directly about how probable \( p = 0.7 \) itself is; the curve's area isn't even 1. Turning "how well does \( p \) explain the data" into "how probable is \( p \)" needs a prior and Bayes' rule (Day 38); that is the MAP section at the end of this lesson. Likelihoods are only good for **comparing** parameter values: \( L(0.7) / L(0.5) = 0.267 / 0.117 \approx 2.3 \) says the data is 2.3 times more probable under \( p = 0.7 \) than under a fair coin.

## The maximum likelihood estimate

The **maximum likelihood estimate** (MLE) is the parameter value at the top of the likelihood curve. It gets a hat, \( \hat{p} \) ("p-hat"), the usual mark for "an estimate of \( p \) computed from data":

\[
\hat{p} = \operatorname*{arg\,max}_{p}\; L(p)
\]

\( \operatorname{arg\,max} \) reads "the argument that maximizes": not the highest value of \( L \), but **the \( p \) where** that highest value happens. NumPy has exactly that: `np.argmax` returns the *position* of the largest entry, not the entry.

### The likelihood of a sequence is a product

Before the calculus, one simplification. You didn't really observe "7 heads in some order"; you observed one particular sequence, say H H T H H H T H T H. Flips are independent, and Day 36's multiplication rule says the probability of independent events all happening is the product of their probabilities. Each head contributes a factor \( p \), each tail a factor \( 1 - p \):

\[
L(p) = p \cdot p \cdot (1-p) \cdot p \cdot p \cdot p \cdot (1-p) \cdot p \cdot (1-p) \cdot p = p^7 (1 - p)^3
\]

In general, for \( n \) independent data points \( x_1, \dots, x_n \), each with probability \( P(x_i \mid \theta) \) under a parameter \( \theta \) ("theta", the usual name for "whatever the model's parameters are"):

\[
L(\theta) = \prod_{i=1}^{n} P(x_i \mid \theta)
\]

The capital Greek letter \( \prod \) ("product") is the multiplying cousin of \( \sum \): multiply the terms for \( i = 1 \) to \( n \).

Compared with the binomial formula, the sequence version is missing the factor \( \binom{10}{7} = 120 \). That factor doesn't depend on \( p \), so it stretches the whole curve up by the same amount and **can't move the peak**. When maximizing a likelihood you can always drop factors that don't involve the parameter.

```python
def coin_likelihood(p, k, n):
    """Probability of one particular sequence of n flips containing k heads."""
    return p**k * (1 - p)**(n - k)

p_grid = np.linspace(0, 1, 1001)          # 0.000, 0.001, ..., 1.000
L = coin_likelihood(p_grid, 7, 10)
p_grid[np.argmax(L)]                      # 0.7000000000000001   the p where L is highest
L.max()                                   # 0.0022235660999999998
np.unique((binom_pmf(7, 10, p_grid[1:-1]) / L[1:-1]).round(6))   # array([120.])  the same ratio at every p
```

That is a **grid search**: try many parameter values, keep the best. It's crude but it always works for one or two parameters, and it is how you check every formula in this lesson.

### Logs: products become sums

The product has a practical problem. Flip the coin 2,000 times and each factor is 0.7 or 0.3; multiply 2,000 of them and the result is far smaller than the smallest number a 64-bit float can hold (about \( 10^{-308} \)). It **underflows** to exactly 0.0, and every \( p \) then looks equally bad:

```python
rng = np.random.default_rng(47)
flips = rng.random(2000) < 0.7            # 2,000 flips of a coin with true p = 0.7 (True = heads)
k_big = flips.sum()
k_big                                     # 1340
np.prod(np.where(flips, 0.7, 0.3))        # 0.0   the product of 2,000 factors underflows
```

The fix is the natural logarithm (Day 30). Two of its properties do all the work:

1. **\( \ln(ab) = \ln a + \ln b \)**: the log of a product is the sum of the logs. Sums of 2,000 moderate negative numbers are perfectly safe.
2. **\( \ln \) is increasing**: if \( a > b \) then \( \ln a > \ln b \). So whatever \( p \) maximizes \( L(p) \) also maximizes \( \ln L(p) \). Same peak position, different height.

The **log-likelihood** is written with a script \( \ell \) ("ell"):

\[
\ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln P(x_i \mid \theta)
\]

For the coin, \( k \) factors of \( p \) and \( n - k \) factors of \( 1 - p \) become

\[
\ell(p) = k \ln p + (n - k) \ln(1 - p)
\]

**Worked example.** For 7 heads in 10 flips at \( p = 0.7 \): \( \ell = 7 \ln 0.7 + 3 \ln 0.3 = 7(-0.3567) + 3(-1.2040) = -2.4967 - 3.6119 = -6.1086 \). Check: \( e^{-6.1086} \approx 0.00222 \), the `L.max()` above.

```python
def coin_loglik(p, k, n):
    """Log-likelihood of k heads in n flips: k ln p + (n - k) ln(1 - p)."""
    return k * np.log(p) + (n - k) * np.log(1 - p)

coin_loglik(0.7, 7, 10)                   # -6.108643020548935
coin_loglik(0.7, k_big, 2000)             # -1272.5664757330192   no underflow: big and negative is fine
np.log(np.where(flips, 0.7, 0.3)).sum()   # -1272.5664757330192   the same thing, flip by flip

p_inner = np.linspace(0.001, 0.999, 999)  # skip 0 and 1, where ln(0) = -inf
p_inner[np.argmax(coin_loglik(p_inner, k_big, 2000))]   # 0.67
k_big / 2000                                            # 0.67
```

The grid search over the log-likelihood lands on 0.67, which is exactly the fraction of heads in the 2,000 flips. That's not a coincidence.

!!! warning "Common mistake: multiplying probabilities in a loop"
    Any code that multiplies many probabilities together (a likelihood, the chance of a long sequence, a naive Bayes score) will eventually return 0.0 and silently break. Add log-probabilities instead. Every ML library does this: PyTorch's `nn.NLLLoss` expects log-probabilities, `nn.CrossEntropyLoss` computes them internally from raw scores, and Day 41's spam filter added log-likelihoods for the same reason.

### Deriving p̂ = k/n with a derivative

At the top of a smooth hill the slope is zero (Day 29). So set the derivative of \( \ell(p) \) to zero and solve. Using \( (\ln p)' = 1/p \) (Day 30) and, by the chain rule (Day 31), \( (\ln(1-p))' = \frac{1}{1-p} \cdot (-1) \):

\[
\frac{d\ell}{dp} = \frac{k}{p} - \frac{n - k}{1 - p}
\]

Set it to zero and multiply both sides by \( p(1-p) \):

\[
k(1 - p) - (n - k)p = 0
\quad\Longrightarrow\quad
k - kp - np + kp = 0
\quad\Longrightarrow\quad
\hat{p} = \frac{k}{n}
\]

The maximum likelihood estimate of a coin's heads-probability is **the fraction of heads you saw**. You'd have guessed it; now you know *why* it's the right guess, and the same recipe (write the log-likelihood, differentiate, set to zero) works where guessing doesn't.

Is it a peak and not a valley? The second derivative is \( -k/p^2 - (n-k)/(1-p)^2 \), negative everywhere, so the curve bends downward: it's a peak. Check the slope numerically with Day 29's central difference:

```python
def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

ell = lambda p: coin_loglik(p, 7, 10)
for p in (0.5, 0.7, 0.9):
    print(f"p = {p}:  numerical slope = {central_diff(ell, p):8.4f}   formula = {7 / p - 3 / (1 - p):8.4f}")
# p = 0.5:  numerical slope =   8.0000   formula =   8.0000
# p = 0.7:  numerical slope =  -0.0000   formula =   0.0000
# p = 0.9:  numerical slope = -22.2222   formula = -22.2222
```

(The `-0.0000` is a tiny negative rounding error, printed to four decimals.)

Uphill (positive slope) left of 0.7, flat at 0.7, downhill right of it.

**More data, sharper peak.** Seven heads in 10 flips and 70 heads in 100 flips have the same \( \hat{p} = 0.7 \), but they are not equally convincing. The figure plots both, each scaled so its peak is 1 (left), and as log-likelihoods shifted so the peak is 0 (right):

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for k, n, c in [(7, 10, "#3949ab"), (70, 100, "#ef6c00")]:
    ll = coin_loglik(p_inner, k, n)
    ax1.plot(p_inner, np.exp(ll - ll.max()), color=c, lw=2, label=f"{k} heads in {n} flips")
    ax2.plot(p_inner, ll - ll.max(), color=c, lw=2, label=f"{k} heads in {n} flips")
for ax in (ax1, ax2):
    ax.axvline(0.7, color="#e53935", ls="--", lw=1.2)
    ax.set_xlabel("p")
    ax.legend(frameon=False, loc="upper left")
ax1.set_ylabel("L(p) / L(p̂)")
ax1.set_title("Likelihood, scaled so the peak is 1")
ax2.set_ylabel("ℓ(p) − ℓ(p̂)")
ax2.set_title("Log-likelihood, shifted so the peak is 0")
ax2.set_ylim(-10, 0.5)
fig.tight_layout()
fig.savefig("day47-coin-likelihood.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Left: two bell-like likelihood curves over p from 0 to 1, both peaking at the dashed red line p = 0.7; the blue one for 7 heads in 10 flips is wide, spreading from about 0.3 to 0.95, while the orange one for 70 heads in 100 flips is a narrow spike from about 0.58 to 0.82. Right: the same two as log-likelihoods shifted to peak at 0; both are upside-down bowls with their top at 0.7, the orange one much steeper](../../assets/img/phase-1/day47-coin-likelihood.svg)
<figcaption>Same estimate, different certainty. Ten times the data makes the likelihood peak ten times more sharply curved.</figcaption>
</figure>

The steepness of the peak measures how sure the data makes you. The curvature (second derivative) at \( \hat{p} \) works out to \( -n / (\hat{p}(1 - \hat{p})) \), and the peak's width, one over the square root of that, is

\[
\sqrt{\frac{\hat{p}(1 - \hat{p})}{n}} = \sqrt{\frac{0.7 \times 0.3}{100}} \approx 0.046
\]

which is exactly the standard error of a proportion you met on Days 43 and 44. Sharp likelihood peaks are narrow confidence intervals.

### Real data: how many Play Store apps show ads?

Each app in the dataset either has ads or doesn't: a coin flip with unknown \( p \). The MLE is the fraction with ads (lesson tested on pandas 3.0):

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv").drop_duplicates()   # or a local copy
ads = (apps["has_ads"] == "Yes").to_numpy()
k_ads, n_apps = int(ads.sum()), len(ads)
k_ads, n_apps                                               # (301, 480)
k_ads / n_apps                                              # 0.6270833333333333   the formula
p_inner[np.argmax(coin_loglik(p_inner, k_ads, n_apps))]     # 0.627                the grid search agrees
```

(The CSV is also on the [Datasets](../../datasets.md) page if you want to read it from a local path.) The grid has steps of 0.001, so it lands on the grid point nearest \( 301/480 = 0.6271 \).

## MLE for a normal distribution

Now a continuous variable. You measure your app's cold start 8 times (milliseconds):

\[
812,\ 790,\ 845,\ 801,\ 768,\ 830,\ 795,\ 819
\]

You model them as draws from a normal distribution with unknown mean \( \mu \) and standard deviation \( \sigma \) (Day 37). Which \( \mu \) and \( \sigma \) make these 8 numbers most likely?

For a continuous variable a single value has probability 0 (Day 37), so the likelihood uses the **density** \( f(x \mid \mu, \sigma) \) instead: the height of the bell at each data point. It plays the same role, because a taller density means "more probability per millisecond around this value". The likelihood is the product of the 8 heights, and its log is a sum. Taking the log of Day 37's density:

\[
\ln f(x \mid \mu, \sigma) = -\ln\!\big(\sigma\sqrt{2\pi}\big) - \frac{(x - \mu)^2}{2\sigma^2}
\]

(The log undoes the \( \exp \), and the constant in front becomes a subtraction.) Summed over the data:

\[
\ell(\mu, \sigma) = -n \ln \sigma - n \ln\sqrt{2\pi} - \frac{1}{2\sigma^2}\sum_{i=1}^{n} (x_i - \mu)^2
\]

**The mean.** Only the last term involves \( \mu \). Its partial derivative with respect to \( \mu \) (Day 32: treat \( \sigma \) as a constant) is \( \frac{1}{\sigma^2} \sum (x_i - \mu) \). Setting it to zero gives \( \sum x_i = n\mu \):

\[
\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} x_i
\]

The MLE of the mean is the **sample mean**. Notice what maximizing \( \ell \) over \( \mu \) actually did: it *minimized* \( \sum (x_i - \mu)^2 \), the sum of squared distances. Remember that; it's the punchline of the next section.

**The standard deviation.** Now differentiate with respect to \( \sigma \), using \( (\ln \sigma)' = 1/\sigma \) and \( (\sigma^{-2})' = -2\sigma^{-3} \):

\[
\frac{\partial \ell}{\partial \sigma} = -\frac{n}{\sigma} + \frac{1}{\sigma^3}\sum_{i=1}^{n}(x_i - \hat{\mu})^2 = 0
\quad\Longrightarrow\quad
\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu})^2
\]

**Worked example.** The 8 times add up to 6460, so \( \hat{\mu} = 807.5 \) ms. The deviations are \( 4.5, -17.5, 37.5, -6.5, -39.5, 22.5, -12.5, 11.5 \); their squares add up to 4130; so \( \hat{\sigma}^2 = 4130 / 8 = 516.25 \) and \( \hat{\sigma} = \sqrt{516.25} \approx 22.72 \) ms.

```python
times = np.array([812., 790., 845., 801., 768., 830., 795., 819.])   # cold starts, ms
mu_hat = times.mean()
sigma_hat = np.sqrt(((times - mu_hat) ** 2).mean())
mu_hat, sigma_hat                         # (807.5, 22.721135535003526)
times.std(), times.std(ddof=1)            # (22.721135535003526, 24.289915602982237)
# NumPy's default (ddof=0) divides by n: that IS the MLE

def normal_loglik(x, mu, sigma):
    """Sum over the last axis of ln f(x | mu, sigma). mu and sigma may be arrays (broadcasting)."""
    return np.sum(-np.log(sigma * np.sqrt(2 * np.pi)) - (x - mu) ** 2 / (2 * sigma ** 2), axis=-1)

normal_loglik(times, mu_hat, sigma_hat)   # -36.33787284273835
normal_loglik(times, 800.0, sigma_hat)    # -36.77370819382794   any other mu scores lower
normal_loglik(times, mu_hat, 30.0)        # -36.85553176337907   so does any other sigma
```

With two parameters the grid becomes a 2-D table: every combination of a \( \mu \) and a \( \sigma \). Broadcasting (Day 9) builds it in one line: \( \mu \) along the rows, \( \sigma \) along the columns, the 8 data points along a third axis that `normal_loglik` sums away.

```python
mus = np.arange(770, 845.01, 0.25)                      # 301 candidate means
sigmas = np.arange(8, 60.01, 0.25)                      # 209 candidate standard deviations
LL = normal_loglik(times, mus[:, None, None], sigmas[None, :, None])
LL.shape                                                # (301, 209)
i, j = np.unravel_index(np.argmax(LL), LL.shape)        # row and column of the best cell
mus[i], sigmas[j]                                       # (807.5, 22.75)
```

`np.unravel_index` turns the flat position that `argmax` returns into a (row, column) pair. The grid's best cell is the grid point nearest the formula's answer, (807.5, 22.72).

Does fitting a normal make sense for real data? Fit the Play Store ratings the same way and draw the fitted bell over their histogram:

```python
from math import erf, sqrt

ratings = apps["rating"].dropna().to_numpy()
mu_r, sigma_r = ratings.mean(), ratings.std()         # the MLE: mean, and std dividing by n
len(ratings), mu_r, sigma_r                            # (462, 4.165584415584416, 0.4533731016292602)
z = (5 - mu_r) / sigma_r                               # how many sigmas above the mean 5 stars is
p_above_5 = 0.5 * (1 - erf(z / sqrt(2)))               # area of the fitted bell above 5 (normal CDF via erf)
z, p_above_5                                           # (1.840461159731343, 0.03285028075273039)
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
cs = ax1.contour(sigmas, mus, LL, levels=LL.max() - np.array([20, 10, 5, 2, 0.5]),
                 colors="#3949ab", linestyles="solid")
ax1.clabel(cs, fmt=lambda v: f"{v - LL.max():.1f}", fontsize=9)
ax1.plot(sigma_hat, mu_hat, "o", color="#e53935", ms=8)
ax1.annotate(f"MLE: μ = {mu_hat:.1f}, σ = {sigma_hat:.2f}", (sigma_hat, mu_hat), xytext=(31, 836),
             color="#e53935", arrowprops=dict(arrowstyle="->", color="#e53935"),
             bbox=dict(boxstyle="round", fc="white", ec="none"))
ax1.set_xlabel("σ (ms)")
ax1.set_ylabel("μ (ms)")
ax1.set_title("Cold starts: log-likelihood minus its maximum")

ax2.hist(ratings, bins=np.arange(1.0, 5.11, 0.1), density=True, color="#c5cae9", edgecolor="#3949ab")
r_axis = np.linspace(2.0, 6.0, 400)
ax2.plot(r_axis, np.exp(normal_loglik(r_axis[:, None], mu_r, sigma_r)), color="#e53935", lw=2,
         label=f"fitted normal: μ = {mu_r:.2f}, σ = {sigma_r:.2f}")
above = r_axis > 5
ax2.fill_between(r_axis[above], np.exp(normal_loglik(r_axis[above][:, None], mu_r, sigma_r)),
                 color="#fff59d", label=f"area above 5 stars: {p_above_5:.1%}")
ax2.set_xlabel("rating")
ax2.set_ylabel("density")
ax2.set_xlim(2.0, 6.0)
ax2.set_ylim(0, 1.2)
ax2.set_title("Play Store ratings with their MLE normal")
ax2.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("day47-normal-fit.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Left: contour lines of the cold-start log-likelihood over sigma on the x axis (8 to 60 ms) and mu on the y axis (770 to 845 ms), nested ovals labelled -0.5, -2, -5, -10 and -20 around a red dot at sigma 22.72, mu 807.5. Right: a histogram of Play Store ratings piled up between 3.5 and 4.9 with a tail down to about 2.8, and a red fitted normal curve centred at 4.17 with sigma 0.45 that spills past 5 stars into a small yellow shaded area](../../assets/img/phase-1/day47-normal-fit.svg)
<figcaption>Left: the 2-D likelihood is a hill with one top, at the formula's answer. Right: MLE always finds the best normal, even when the data isn't normal: the fitted bell puts a few percent of apps above the 5-star maximum.</figcaption>
</figure>

!!! warning "Common mistake: trusting the family you chose"
    MLE answers "**given** that the data comes from a normal (or a coin, or a line with Gaussian noise), which parameters fit best?". It cannot tell you that the family itself is wrong. Ratings are capped at 5 and skewed to the left, so the best normal still predicts about 3% of apps rated above 5 stars, which is impossible. Always plot the fitted model over the data.

!!! warning "Common mistake: the MLE variance is slightly too small"
    \( \hat{\sigma}^2 \) divides by \( n \). Day 43 showed by simulation that this underestimates the true variance on average, and that dividing by \( n - 1 \) fixes it (Bessel's correction). Both are reasonable: the MLE is "the best-fitting value", \( n - 1 \) gives "right on average". With 8 points the difference is a factor \( \sqrt{8/7} \approx 1.07 \); with 8,000 it's invisible. "Maximum likelihood" does not mean "unbiased".

## Why minimizing MSE is maximum likelihood

On Day 33 you fitted a line by minimizing the mean squared error. Why squared? Why not the absolute error, or the fourth power? Here is the answer.

Write down a **model of how the data was generated**: each target is the line plus random noise,

\[
y_i = w x_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
\]

- \( w \) is the weight (slope) we want to estimate. To keep it to one parameter, the line goes through the origin.
- \( \varepsilon_i \) ("epsilon") is the noise on point \( i \), and \( \sim \mathcal{N}(0, \sigma^2) \) reads "is drawn from a normal distribution with mean 0 and variance \( \sigma^2 \)". The noise terms are independent.

Equivalently: given \( x_i \), the target \( y_i \) is normal with mean \( w x_i \) and standard deviation \( \sigma \). Each point gets its own bell, centred on the line. The likelihood of \( w \) is the product of the bell heights at the actual points:

<figure class="diagram" markdown>
![Animation: six red data points rising from left to right; a straight line y = w x is drawn through them with slope 1.5, then 2.6, then 2.0. At each point a sideways bell curve is centred on the line, and an orange bar shows the bell's height at the point. For slope 1.5 and 2.6, points far from the line sit in the bells' thin tails with tiny bars (as low as 0.001), the side panel showing log-likelihood -17.30 and MSE 3.93, then -22.75 and 5.75. For slope 2.0 every point sits near a bell's peak, with log-likelihood -6.13 and MSE 0.21, labelled best w: highest likelihood = lowest MSE](../../assets/img/phase-1/day47-gaussian-bells.svg)
<figcaption>Animated: under Gaussian noise each point's likelihood factor is the height of a bell centred on the line. The slope that puts every point near the top of its bell is the one with the smallest squared errors.</figcaption>
</figure>

Now the log-likelihood. Plug the mean \( w x_i \) into the normal log-density from the previous section:

\[
\ell(w) = \sum_{i=1}^{n} \left[ -\ln\!\big(\sigma\sqrt{2\pi}\big) - \frac{(y_i - w x_i)^2}{2\sigma^2} \right]
= \underbrace{-n\ln\!\big(\sigma\sqrt{2\pi}\big)}_{\text{does not involve } w} \;-\; \frac{n}{2\sigma^2} \cdot \underbrace{\frac{1}{n}\sum_{i=1}^{n} (y_i - w x_i)^2}_{\text{MSE}(w)}
\]

Read it slowly. The first part is a constant. The second is the MSE times a positive number \( n/(2\sigma^2) \), with a minus sign. So **making \( \ell \) as large as possible is exactly making the MSE as small as possible**, for any value of \( \sigma \):

\[
\hat{w}_{\text{MLE}} = \operatorname*{arg\,max}_{w}\; \ell(w) = \operatorname*{arg\,min}_{w}\; \text{MSE}(w)
\]

Least squares isn't an arbitrary choice: it's what maximum likelihood says to do **if you believe the noise is Gaussian**. And setting the derivative of \( \sum (y_i - w x_i)^2 \) to zero, \( -2\sum x_i (y_i - w x_i) = 0 \), gives the 1-D formula \( \hat{w} = \sum x_i y_i / \sum x_i^2 \).

```python
x = np.array([1., 2., 3., 4., 5., 6.])
y = np.array([2.6, 3.5, 6.4, 8.3, 9.4, 12.1])        # roughly y = 2x plus noise

def mse(w):
    return np.mean((y - w * x) ** 2)

def gauss_loglik(w, sigma=1.0):
    return normal_loglik(y, w * x, sigma)

w_grid = np.linspace(0, 4, 4001)                      # steps of 0.001
mses = np.array([mse(w) for w in w_grid])
w_grid[np.argmin(mses)]                                           # 1.996   lowest MSE
w_grid[np.argmax([gauss_loglik(w) for w in w_grid])]              # 1.996   highest likelihood, sigma = 1
w_grid[np.argmax([gauss_loglik(w, sigma=3.0) for w in w_grid])]   # 1.996   sigma doesn't move the peak
(x * y).sum() / (x * x).sum()                                     # 1.9956043956043956   the formula
np.linalg.lstsq(x[:, None], y, rcond=None)[0]                     # array([1.9956044])   Day 25's least squares

# The identity above, checked at w = 2: loglik = -n ln(sigma sqrt(2 pi)) - n MSE / (2 sigma^2)
gauss_loglik(2.0), -6 * np.log(np.sqrt(2 * np.pi)) - 6 * mse(2.0) / 2
# (-6.128631199228037, -6.128631199228036)   equal, up to the last digit of rounding
```

Four routes, one answer. The same argument works for many weights and a bias: with Gaussian noise, maximizing the likelihood of a linear regression is minimizing Day 33's MSE, \( \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2 \).

This also tells you **when MSE is the wrong loss**. A Gaussian's tails fall off like \( e^{-r^2/2} \), so the model considers a big residual \( r \) astronomically unlikely and bends the line hard to avoid one: that's why MSE is so sensitive to outliers (Day 50). If you instead assume noise with fatter tails, the Laplace distribution with density proportional to \( e^{-|r|/b} \), the same derivation gives the mean **absolute** error. Choosing a loss is choosing a noise model.

## Why cross-entropy is maximum likelihood

Now a classifier. Each example has a label \( y_i \in \{0, 1\} \) (say "the user tapped the notification" or not), and the model outputs a probability \( p_i \) that \( y_i = 1 \), for example \( p_i = \sigma(w x_i + b) \) with Day 30's sigmoid. The data model: **each label is a coin flip with its own heads-probability \( p_i \)**, a Bernoulli variable (Day 37).

The likelihood factor for one example is \( p_i \) if \( y_i = 1 \) and \( 1 - p_i \) if \( y_i = 0 \). A neat way to write both cases in one formula:

\[
P(y_i \mid p_i) = p_i^{\,y_i}\,(1 - p_i)^{1 - y_i}
\]

(If \( y_i = 1 \) the second factor has power 0 and equals 1; if \( y_i = 0 \) the first does.) Take logs, add over the examples, flip the sign so it's something to minimize, and divide by \( n \) to get an average:

\[
-\frac{1}{n}\,\ell = -\frac{1}{n}\sum_{i=1}^{n} \Big[\, y_i \ln p_i + (1 - y_i)\ln(1 - p_i) \,\Big]
\]

That is **binary cross-entropy**, also called log loss, the standard loss for every yes/no classifier. Minimizing it **is** maximizing the likelihood of the labels. Minimizing "the negative log-likelihood" (NLL) is the general name; every MLE in this lesson can be written as one.

**Worked example.** Four examples with labels \( 1, 0, 1, 0 \) and model probabilities \( 0.9, 0.2, 0.6, 0.7 \). The likelihood factors are \( 0.9, 0.8, 0.6, 0.3 \) (for the 0-labels, take \( 1 - p \)). Their logs: \( -0.105, -0.223, -0.511, -1.204 \), summing to \( -2.043 \). Divide by 4 and flip the sign: the cross-entropy is \( 0.511 \). The fourth example, labelled 0 but given probability 0.7 of being 1, contributes more than the other three combined.

```python
y_cls = np.array([1, 0, 1, 0])
p_model = np.array([0.9, 0.2, 0.6, 0.7])

factors = np.where(y_cls == 1, p_model, 1 - p_model)     # each label's likelihood factor
factors                                                  # array([0.9, 0.8, 0.6, 0.3])
factors.prod()                                           # 0.12960000000000002   likelihood of all four labels
-np.log(factors).mean()                                  # 0.5108256237659907    negative log-likelihood, averaged
-np.mean(y_cls * np.log(p_model) + (1 - y_cls) * np.log(1 - p_model))   # 0.5108256237659907   cross-entropy
```

With more than two classes, each example's factor is the probability the model gave to the correct class, and the average negative log of it is the (categorical) cross-entropy that sits after every softmax layer. Day 50 uses these losses and Day 54 explains the name "cross-entropy" from information theory; today's point is that both are just "negative log-likelihood, averaged".

!!! android "Android bridge: you already ship maximum likelihood estimates"
    **Crash rates.** Firebase Crashlytics shows crash-free statistics: the percentage of users or sessions that didn't crash. Treat each session as a coin flip that crashes with probability \( p \): then "crashed sessions ÷ sessions" is precisely \( \hat{p} = k/n \), the maximum likelihood estimate. If you compute such a likelihood yourself (to compare two builds' crash rates over thousands of sessions, say), sum logs instead of multiplying `Double`s, for the same underflow reason as in Python:

    ```kotlin
    import kotlin.math.ln

    // Log-likelihood of crash rate p, given each session's outcome (true = crashed).
    fun logLikelihood(p: Double, crashed: List<Boolean>): Double =
        crashed.sumOf { if (it) ln(p) else ln(1 - p) }

    // Its maximum, from the derivative: the fraction that crashed.
    fun mleCrashRate(crashed: List<Boolean>): Double =
        crashed.count { it }.toDouble() / crashed.size
    ```

    The MLE has a weak spot with little data: a new build with 0 crashes in its first 3 sessions gets \( \hat{p} = 0 \), "it never crashes". The MAP section below fixes exactly that.

    **On-device classifiers.** The image or text classifier you run with LiteRT (TensorFlow Lite) was almost certainly trained by minimizing cross-entropy, so the probabilities in its output tensor are maximum likelihood fits to its *training* data. If your users' photos or messages look different from that data, those probabilities can be confidently wrong; the likelihood was only ever maximized on what the model saw.

## Numerical MLE when there is no formula

The coin, the normal and least squares all had clean formulas. Most models don't: there's no closed form for logistic regression's weights, let alone a neural network's. Then you maximize the log-likelihood numerically. Today's tool is the grid search; Week 8 replaces it with gradient descent, which walks uphill on \( \ell \) (or downhill on the NLL) instead of trying every point, and scales to millions of parameters.

Here's a case with no formula: a one-feature logistic model \( p_i = \sigma(w x_i) \) for whether a user taps a notification, with \( x_i \) = how many days since they last opened the app, centred so 0 is a typical user. Fit \( w \) by grid search on the log-likelihood:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

days = np.array([-3., -2., -2., -1., -1., 0., 0., 1., 1., 2., 2., 3.])  # days since last open, centred
tapped = np.array([1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0])                 # made-up labels

def logistic_loglik(w):
    p = sigmoid(w * days)
    return np.sum(tapped * np.log(p) + (1 - tapped) * np.log(1 - p))

w_cands = np.linspace(-5, 5, 10_001)                   # steps of 0.001
lls_logit = np.array([logistic_loglik(w) for w in w_cands])
w_best = w_cands[np.argmax(lls_logit)]
w_best                                                 # -1.104
sigmoid(w_best * np.array([-3., 0., 3.]))              # array([0.96483819, 0.5       , 0.03516181])
# fitted tap probability for a recent, a typical and a lapsed user
central_diff(logistic_loglik, w_best)                  # 0.0007894410014586127   slope ≈ 0 at the top
```

A negative \( w \): the longer since the last open, the lower the chance of a tap. Setting the derivative to zero here gives an equation with \( w \) trapped inside sigmoids that no algebra can untangle, but the numerical top of the hill is just as good.

## MAP estimation: adding a prior

Back to the weakness from the Android bridge. With 0 crashes in 3 sessions, the MLE says the crash rate is exactly 0. Nobody believes that. You have **prior knowledge**: most builds crash sometimes. Day 38's Bayes' rule is the tool for combining prior beliefs with evidence, and it works for parameters too:

\[
\underbrace{P(\theta \mid \text{data})}_{\text{posterior}} = \frac{\overbrace{P(\text{data} \mid \theta)}^{\text{likelihood } L(\theta)}\;\; \overbrace{P(\theta)}^{\text{prior}}}{P(\text{data})}
\]

The **MAP estimate** ("maximum a posteriori", Latin for "from the later" because it's computed after seeing the data) is the peak of the posterior instead of the peak of the likelihood. The denominator \( P(\text{data}) \) doesn't involve \( \theta \), so like the \( \binom{10}{7} \) earlier it can't move the peak, and taking logs turns the product into a sum:

\[
\hat{\theta}_{\text{MAP}} = \operatorname*{arg\,max}_{\theta}\; \Big[\, \underbrace{\ell(\theta)}_{\text{log-likelihood}} + \underbrace{\ln P(\theta)}_{\text{log-prior}} \,\Big]
\]

MLE is MAP with a flat prior (every \( \theta \) equally plausible beforehand, so \( \ln P(\theta) \) is a constant).

**The coin.** Take a prior proportional to \( p(1 - p) \): an upside-down bowl that is 0 at \( p = 0 \) and \( p = 1 \) and highest at 0.5, meaning "probably not a coin that never or always lands heads". The posterior is proportional to \( p^k(1-p)^{n-k} \cdot p(1-p) = p^{k+1}(1-p)^{n-k+1} \), which is the likelihood of \( k + 1 \) heads in \( n + 2 \) flips. So by the coin formula:

\[
\hat{p}_{\text{MAP}} = \frac{k + 1}{n + 2}
\]

The prior acts like one imaginary head and one imaginary tail added to the data. For 0 crashes in 3 sessions: \( 1/5 = 0.2 \) instead of 0; after 0 crashes in 3,000 sessions, \( 1/3002 \approx 0.0003 \): the data has overwhelmed the prior. It's the same "add one" idea as the Laplace smoothing in Day 41's spam filter, where every word gets one imaginary extra occurrence.

### A Gaussian prior on a weight is an L2 penalty

Now the result Day 38 promised. Go back to the line \( y_i = w x_i + \varepsilon_i \) with Gaussian noise of standard deviation \( \sigma \). Before seeing any data, believe that \( w \) is probably small: give it a normal prior with mean 0 and standard deviation \( \tau \) ("tau"),

\[
w \sim \mathcal{N}(0, \tau^2)
\qquad\Longrightarrow\qquad
\ln P(w) = -\ln\!\big(\tau\sqrt{2\pi}\big) - \frac{w^2}{2\tau^2}
\]

Add it to the log-likelihood from the MSE section and drop everything that doesn't involve \( w \):

\[
\ell(w) + \ln P(w) = \text{const} - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - w x_i)^2 - \frac{w^2}{2\tau^2}
\]

Maximizing that is minimizing its negative. Multiply by the positive constant \( 2\sigma^2 \) (which doesn't move the minimum either):

\[
\hat{w}_{\text{MAP}} = \operatorname*{arg\,min}_{w}\; \Bigg[\, \underbrace{\sum_{i=1}^{n}(y_i - w x_i)^2}_{\text{the MLE objective}} + \underbrace{\lambda\, w^2}_{\text{L2 penalty}} \,\Bigg],
\qquad \lambda = \frac{\sigma^2}{\tau^2}
\]

That is exactly Day 22's **L2 regularization** ("error \( + \lambda\|\mathbf{w}\|_2^2 \)", ridge regression, weight decay): the squared-error objective plus a penalty on the weight's size. Regularization isn't a hack bolted onto MLE. It **is** a prior belief that weights are small, and the knob \( \lambda \) has a meaning:

- **Strong belief** (small \( \tau \)): big \( \lambda \), weights pulled hard towards 0.
- **Weak belief** (large \( \tau \)): \( \lambda \to 0 \), and MAP becomes plain MLE.
- **Noisy data** (large \( \sigma \)): big \( \lambda \) too: the less you trust each measurement, the more the prior counts.

Setting the derivative to zero, \( -2\sum x_i(y_i - w x_i) + 2\lambda w = 0 \), gives

\[
\hat{w}_{\text{MAP}} = \frac{\sum x_i y_i}{\sum x_i^2 + \lambda}
\]

The \( \lambda \) in the denominator shrinks the estimate towards 0. **Worked example.** Use only the first two points, \( (1, 2.6) \) and \( (2, 3.5) \), with \( \sigma = 1 \) and \( \tau = 1 \), so \( \lambda = 1 \). Then \( \sum x_i y_i = 2.6 + 7.0 = 9.6 \) and \( \sum x_i^2 = 1 + 4 = 5 \): the MLE is \( 9.6 / 5 = 1.92 \), the MAP estimate \( 9.6 / 6 = 1.60 \). With all six points, \( \sum x_i y_i = 181.6 \) and \( \sum x_i^2 = 91 \): MLE \( 1.9956 \), MAP \( 181.6 / 92 \approx 1.974 \). The prior matters a lot with little data and hardly at all with more.

```python
def log_prior(w, tau):
    return -np.log(tau * np.sqrt(2 * np.pi)) - w ** 2 / (2 * tau ** 2)

sigma, tau = 1.0, 1.0
lam = sigma ** 2 / tau ** 2
for n_pts in (2, 6):
    xs, ys = x[:n_pts], y[:n_pts]
    w_mle = (xs * ys).sum() / (xs * xs).sum()
    w_map = (xs * ys).sum() / ((xs * xs).sum() + lam)
    log_post = np.array([normal_loglik(ys, w * xs, sigma) + log_prior(w, tau) for w in w_grid])
    penalized = np.array([((ys - w * xs) ** 2).sum() + lam * w ** 2 for w in w_grid])
    print(f"{n_pts} points: MLE = {w_mle:.4f}   MAP = {w_map:.4f}   "
          f"grid argmax log-posterior = {w_grid[np.argmax(log_post)]:.3f}   "
          f"grid argmin squared error + L2 = {w_grid[np.argmin(penalized)]:.3f}")
# 2 points: MLE = 1.9200   MAP = 1.6000   grid argmax log-posterior = 1.600   grid argmin squared error + L2 = 1.600
# 6 points: MLE = 1.9956   MAP = 1.9739   grid argmax log-posterior = 1.974   grid argmin squared error + L2 = 1.974
```

The formula, the log-posterior grid and the penalized-squared-error grid all agree. Here are the three curves for two points and for six:

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, n_pts in zip(axes, (2, 6)):
    xs, ys = x[:n_pts], y[:n_pts]
    ll_w = np.array([normal_loglik(ys, w * xs, sigma) for w in w_grid])
    lp_w = log_prior(w_grid, tau)
    post = ll_w + lp_w
    ax.plot(w_grid, ll_w - ll_w.max(), color="#3949ab", lw=2, label="log-likelihood")
    ax.plot(w_grid, lp_w - lp_w.max(), color="#00897b", lw=2, ls=":", label="log-prior, N(0, 1)")
    ax.plot(w_grid, post - post.max(), color="#e53935", lw=2.5, label="log-posterior = sum")
    ax.axvline(w_grid[np.argmax(ll_w)], color="#3949ab", ls="--", lw=1)
    ax.axvline(w_grid[np.argmax(post)], color="#e53935", ls="--", lw=1)
    ax.set_ylim(-8, 0.5)
    ax.set_xlabel("weight w")
    ax.set_title(f"{n_pts} data points: MLE {w_grid[np.argmax(ll_w)]:.2f}, MAP {w_grid[np.argmax(post)]:.2f}")
axes[0].set_ylabel("each curve shifted so its peak is 0")
axes[0].legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("day47-map-prior.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Two panels over the weight w from 0 to 4, each with three upside-down curves shifted to peak at 0: a dotted green log-prior peaking at w = 0, a blue log-likelihood and a red log-posterior. Left, with 2 data points, the blue curve is wide and peaks at 1.92 while the red posterior is pulled left to peak at 1.60. Right, with 6 data points, the blue curve is very narrow and the red curve almost coincides with it, peaks at 2.00 and 1.97](../../assets/img/phase-1/day47-map-prior.svg)
<figcaption>The posterior is the likelihood plus the prior. With 2 points the prior's pull towards 0 is strong (left); with 6 the much sharper likelihood dominates and MAP is almost MLE (right).</figcaption>
</figure>

Two more translations you will meet: a **Laplace prior**, density proportional to \( e^{-|w|/b} \), turns into an **L1 penalty** \( \lambda |w| \) (the lasso from Day 22, which pushes weights to exactly 0), and in deep learning "weight decay" is the L2 penalty applied by the optimizer.

## Where this shows up in ML

| Today's idea | Where you'll meet it |
|---|---|
| Likelihood vs probability | Every "the model assigns probability \( p \) to the data" statement; evaluating language models by the likelihood of real text |
| Negative log-likelihood | The general loss: PyTorch's `nn.NLLLoss`, and cross-entropy built on it (Days 50, 54) |
| MSE = MLE with Gaussian noise | Linear regression (Day 55's milestone), and why outliers wreck it (Day 50) |
| Cross-entropy = MLE for labels | Every classifier, from logistic regression to the softmax head of an image model |
| Grid search, then gradient descent | Numerical MLE; Week 8 swaps the grid for gradient descent (Day 51) |
| MAP = MLE + log-prior | L2 regularization / weight decay, L1 / lasso, Laplace smoothing |

---

## Exercises

1. A new build had 3 crashes in 400 sessions. Write the log-likelihood \( \ell(p) \), find \( \hat{p} \) by the formula and by a grid search over \( p \) in \( [0.0001, 0.05] \), and compute how many times more likely the data is under \( \hat{p} \) than under \( p = 0.02 \) (use a difference of log-likelihoods). What is the MAP estimate with the \( p(1-p) \) prior?
2. Among the rated Play Store apps (`apps["rating"].dropna()`), what is the MLE of the probability that an app is rated at least 4.5? Check it with a grid search on the coin log-likelihood.
3. Fit a normal by maximum likelihood to \( \log_{10} \) of the Play Store review counts (`apps["reviews"]`): get \( \hat{\mu} \) and \( \hat{\sigma} \) from the formulas, then from a 2-D grid search with `normal_loglik`, and confirm that `np.std` with its default `ddof` matches \( \hat{\sigma} \).
4. Fit the line with a bias, \( y_i = w x_i + b + \varepsilon_i \), to the six points `x`, `y` by maximizing the Gaussian log-likelihood (\( \sigma = 1 \)) with a 2-D grid search over \( w \) and \( b \). Compare with `np.linalg.lstsq` using a column of ones for \( b \).
5. Labels `[1, 1, 0, 1, 0, 1, 1, 0]`. A "model" with no features predicts the same probability \( q \) for every example. Grid-search the \( q \) that minimizes the binary cross-entropy. Which earlier formula does it match, and why?
6. For the six points with \( \sigma = 1 \), which \( \lambda \) makes \( \hat{w}_{\text{MAP}} = 1.5 \)? Which prior standard deviation \( \tau \) does that correspond to? Check by grid search on the penalized squared error.

??? success "Solutions"
    ```python
    # 1: l(p) = 3 ln p + 397 ln(1 - p);  p_hat = 3/400 = 0.0075
    p1 = np.linspace(0.0001, 0.05, 4991)          # steps of 0.00001
    p1[np.argmax(coin_loglik(p1, 3, 400))], 3 / 400   # (0.007500000000000001, 0.0075)
    diff = coin_loglik(3 / 400, 3, 400) - coin_loglik(0.02, 3, 400)
    diff, np.exp(diff)                            # (2.089265276965815, 8.078977172561189)   about 8 times likelier
    (3 + 1) / (400 + 2)                           # 0.009950248756218905   MAP with the p(1-p) prior
    ```

    ```python
    # 2: each rated app is a coin flip: "rated >= 4.5" or not
    k2, n2 = int((ratings >= 4.5).sum()), len(ratings)
    k2, n2, k2 / n2                                    # (133, 462, 0.2878787878787879)
    p_inner[np.argmax(coin_loglik(p_inner, k2, n2))]   # 0.28800000000000003   the nearest grid point
    ```

    ```python
    # 3: the MLE of a normal is the mean and the std dividing by n
    lr = np.log10(apps["reviews"].to_numpy())
    mu3, s3 = lr.mean(), np.sqrt(((lr - lr.mean()) ** 2).mean())
    mu3, s3, lr.std()             # (3.1996328950275355, 0.9525330265097233, 0.9525330265097233)
    m_grid = np.arange(2.0, 5.0, 0.005)
    s_grid = np.arange(0.5, 2.0, 0.005)
    LL3 = normal_loglik(lr, m_grid[:, None, None], s_grid[None, :, None])
    i3, j3 = np.unravel_index(np.argmax(LL3), LL3.shape)
    m_grid[i3], s_grid[j3]        # (3.1999999999999744, 0.9550000000000004)   within a grid step of the formula
    ```

    ```python
    # 4: two parameters, so a 2-D grid; the noise is Gaussian, so it's least squares
    w4 = np.arange(1.0, 3.0, 0.002)
    b4 = np.arange(-2.0, 2.0, 0.002)
    LL4 = normal_loglik(y, w4[:, None, None] * x + b4[None, :, None], 1.0)
    i4, j4 = np.unravel_index(np.argmax(LL4), LL4.shape)
    w4[i4], b4[j4]                                # (1.9180000000000008, 0.33800000000000185)
    X4 = np.column_stack([x, np.ones_like(x)])
    np.linalg.lstsq(X4, y, rcond=None)[0]         # array([1.91714286, 0.34      ])   the same, up to the grid step
    ```

    ```python
    # 5: a constant model is a coin; cross-entropy is its negative log-likelihood / n
    lab = np.array([1, 1, 0, 1, 0, 1, 1, 0])
    q = np.linspace(0.001, 0.999, 999)
    bce = -(lab.sum() * np.log(q) + (len(lab) - lab.sum()) * np.log(1 - q)) / len(lab)
    q[np.argmin(bce)], lab.mean()     # (0.625, 0.625)
    ```
    It's the coin MLE \( \hat{p} = k/n = 5/8 \): with no features, every label is a flip of the same coin, and minimizing cross-entropy is maximizing that coin's likelihood. A trained classifier's average predicted probability usually lands near the fraction of positives for the same reason.

    ```python
    # 6: 181.6 / (91 + lam) = 1.5  =>  lam = 181.6 / 1.5 - 91;  lam = sigma^2 / tau^2  =>  tau = 1 / sqrt(lam)
    lam6 = (x * y).sum() / 1.5 - (x * x).sum()
    lam6, 1 / np.sqrt(lam6)           # (30.066666666666663, 0.1823716631040194)
    pen6 = np.array([((y - w * x) ** 2).sum() + lam6 * w ** 2 for w in w_grid])
    w_grid[np.argmin(pen6)]           # 1.5
    ```
    It takes \( \lambda \approx 30 \), a prior standard deviation of about 0.18: a strong belief that \( w \) is near 0, strong enough to drag the estimate a quarter of the way from 2 towards 0 even with six points that clearly say 2.

## Resources

- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest): search the channel for "maximum likelihood"; his short videos on likelihood vs probability and MLE for the normal match today's order.
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/): chapter 8, "When Models Meet Data", covers maximum likelihood and MAP estimation, and chapter 9 derives linear regression both ways.
- [Dive into Deep Learning: Maximum Likelihood](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.html): the maximum likelihood principle, negative log-likelihood and its numerical stability, written for deep learning.
- [Stanford CS229 lecture notes](https://cs229.stanford.edu/): the main notes' "probabilistic interpretation" of least squares is today's MSE derivation, and the logistic regression section is today's cross-entropy one.
- [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/): the "Frequentist Inference" and "Bayesian Inference" chapters have interactive likelihood and prior/posterior visuals.
- [Wikipedia: Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation): a reference for the definitions and the worked examples (Bernoulli, normal) once you know the idea.
- [Firebase Crashlytics documentation](https://firebase.google.com/docs/crashlytics): where the crash-free statistics from the Android bridge come from.

## Checklist

- [ ] I can say, for one formula, which question is a probability and which is a likelihood.
- [ ] I know why a likelihood is a product over independent data points, and why I take its log before computing it.
- [ ] I can derive \( \hat{p} = k/n \) by setting the derivative of \( k \ln p + (n-k)\ln(1-p) \) to zero.
- [ ] I can derive the MLE of a normal's mean and variance, and I know the variance one divides by \( n \).
- [ ] I can explain why minimizing MSE is maximum likelihood under Gaussian noise, and what changes with a different noise model.
- [ ] I can show that binary cross-entropy is the averaged negative log-likelihood of Bernoulli labels.
- [ ] I can find an MLE numerically with a 1-D or 2-D grid search and `np.unravel_index`.
- [ ] I can explain MAP as "log-likelihood + log-prior", and derive that a Gaussian prior on a weight gives an L2 penalty with \( \lambda = \sigma^2/\tau^2 \).

**Next:** [Day 48: Week 7 review and mini-project](day-48-week7-review.md)
