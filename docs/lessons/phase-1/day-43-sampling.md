# Day 43: Samples, Populations and the Sampling Distribution

**Phase 1 · Week 7 · ~90 minutes**

## Objectives

- [ ] Tell a population from a sample, and a parameter (\( \mu, \sigma \)) from a statistic (\( \bar{x}, s \)).
- [ ] Recognize sampling bias (survivorship, self-selection, convenience) and show by simulation that a bigger biased sample does not fix it.
- [ ] Draw a simple random sample with `rng.choice(..., replace=False)` and with pandas' `DataFrame.sample`.
- [ ] Build the sampling distribution of the mean by simulation, and describe its centre, spread and shape.
- [ ] Use the standard error \( \sigma/\sqrt{n} \), estimate it from one sample as \( s/\sqrt{n} \), and plan a sample size with it.
- [ ] Explain why the sample variance divides by \( n - 1 \) (Bessel's correction), with a worked example and a simulation.
- [ ] Connect all of this to ML: test sets are samples, test accuracy has a standard error, and training data can be biased.

**Prerequisites.** [Day 39](day-39-expectation-variance.md) (expected value, variance, \( \sigma/\sqrt{n} \) for an average, sample mean vs true mean) and [Day 40](day-40-joint-clt.md) (the central limit theorem). The pandas is from [Day 16](day-16-pandas-selecting.md) and [Day 17](day-17-pandas-cleaning.md); the plotting from [Day 19](day-19-matplotlib.md).

---

## The big idea

A cook tastes one spoonful of soup and decides the whole pot needs salt. That's statistics in one gesture: look at a **small part**, draw a conclusion about the **whole**. It works for two reasons, and fails without either of them:

1. **Stir first.** If the salt sank to the bottom and you skim the top, the spoonful lies to you, and tasting ten spoonfuls from the top lies just as confidently. The spoonful must be a fair, random part of the pot.
2. **A spoonful is enough, but not a drop.** How sure you can be depends on how much you taste, and in a precise way: the error shrinks like \( 1/\sqrt{n} \). Interestingly, it barely depends on the size of the pot.

Week 6 ran in one direction: *given* the probabilities, what data will we see? Statistics, this week, runs the other way: *given* the data we saw, what can we say about the thing that produced it? Today is the vocabulary and the single most important picture for that, the **sampling distribution**.

| Word | Meaning | Example (Play Store ratings) |
|---|---|---|
| **Population** | Everything you want to know about | All rated apps in the store |
| **Sample** | The part you actually measured | 25 apps you looked at |
| **Parameter** | A number describing the population; fixed, usually unknown | The true mean rating \( \mu \) |
| **Statistic** | A number computed from the sample; changes with every sample | Your sample's mean \( \bar{x} \) |
| **Sampling distribution** | How a statistic would vary over all the samples you *could* have drawn | The spread of \( \bar{x} \) across many surveys of 25 apps |

**Analogy for an app developer.** You can't watch every user's session, so analytics tools look at some of them. Whether the dashboard's "average session length" is trustworthy is exactly today's two questions: was the sample fair (stirred), and was it big enough (a spoonful, not a drop)?

## Population and sample, parameter and statistic

To *see* the difference between a parameter and a statistic you need to know the truth, which in real life you never do. So we cheat: we declare that the Play Store dataset's 462 rated apps (after dropping the 6 duplicate rows and the 18 unrated apps) **are** the population. Now we know \( \mu \) and \( \sigma \) exactly, and can watch how well samples guess them. (This lesson was tested on pandas 3.0.)

```python
import numpy as np
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")    # or a local copy from the Datasets page
rated = apps.drop_duplicates().dropna(subset=["rating"])
population = rated["rating"].to_numpy()

N = len(population)
mu = population.mean()       # a parameter: the true mean of the whole population
sigma = population.std()     # ddof=0 is right here: this IS the whole population
N, mu, sigma                 # (462, 4.165584415584416, 0.4533731016292602)
```

(You can download the CSV from the [Datasets](../../datasets.md) page and read it from a local path instead.)

The notation, one symbol at a time. Greek letters for the population, Latin letters for the sample:

| | Population (parameter) | Sample (statistic) |
|---|---|---|
| Size | \( N \) | \( n \) |
| Mean | \( \mu \) ("mu") | \( \bar{x} \) ("x bar") |
| Standard deviation | \( \sigma \) ("sigma") | \( s \) |
| Proportion (e.g. share with ads) | \( p \) | \( \hat{p} \) ("p hat") |

A hat on a letter means "an estimate of": \( \hat{p} \) is our guess at \( p \), and you'll also see \( \hat{\mu} \) for \( \bar{x} \). The formulas for the sample, with data values \( x_1, x_2, \dots, x_n \):

\[
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
\qquad\qquad
s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2
\]

The \( n - 1 \) is not a typo; it gets its own section below. Now take one survey of 25 apps:

```python
rng = np.random.default_rng(43)
sample = rng.choice(population, size=25, replace=False)   # 25 different apps, no repeats
sample[:10]                 # array([4.1, 3.6, 4.2, 4.1, 3.9, 3. , 3.7, 4.2, 4.1, 3.5])
x_bar = sample.mean()           # a statistic: computed from the sample
s = sample.std(ddof=1)          # the sample standard deviation (n - 1, explained below)
x_bar, s                    # (4.172, 0.48607269689488497)
```

`replace=False` means **without replacement**: once an app is picked it can't be picked again, like dealing cards. The guess \( \bar{x} \) is close to \( \mu \) but not equal to it, and a different seed would give a different guess. That is the entire subject of this lesson: **a statistic is a random variable**, because the sample was random.

pandas can draw the sample for you, whole rows at a time, and accepts the same `rng`:

```python
survey = rated.sample(n=25, random_state=rng)   # 25 random rows (without replacement by default)
print(survey[["app_name", "category", "rating"]].head(3).to_string())
survey["rating"].mean()                         # 4.175999999999999   (4.176 plus floating-point dust)
```

```text
            app_name          category  rating
282       Calm Beats  Health & Fitness     4.8
238  Pixel Wallet Go             Games     4.4
56     Bright Runner           Finance     4.2
```

## Sampling bias: when the spoon skims the top

The cleverest maths can't rescue a sample that isn't representative. **Sampling bias** means the way you collected the data makes some members of the population more likely to end up in it than others, *in a way that's related to what you're measuring*. Three kinds you will meet constantly:

| Bias | What goes wrong | App-world example |
|---|---|---|
| **Self-selection** | People choose whether to be in the sample | Only users with strong feelings write a review |
| **Survivorship** | You only see the ones that "survived" a filter | Studying successful apps to learn what makes apps succeed, while the thousands that died are gone from the store |
| **Convenience** | You sample whatever is easy to reach | Surveying your beta testers, who love the app enough to have signed up |

Survivorship has a famous story. In World War II, analysts looked at where returning bombers had bullet holes and proposed armouring those spots. The statistician Abraham Wald pointed out that these were the planes that *came back*: the holes showed where a plane could be hit and survive. The armour belonged where the returning planes had *no* holes, because planes hit there never returned.

### Simulating self-selection: who writes a review?

Here is a made-up app with 100,000 users. Each user has a "true" star rating in their head, and the chance that they actually *write* a review depends on how they feel: furious users and delighted users write, the content majority mostly doesn't. (These numbers are invented for the illustration, but the J-shape is typical of real review sites.)

```python
stars = np.array([1, 2, 3, 4, 5])
p_feel = np.array([0.05, 0.07, 0.18, 0.40, 0.30])     # how 100,000 users really feel
users = rng.choice(stars, size=100_000, p=p_feel)

p_write = np.array([0.30, 0.10, 0.03, 0.04, 0.12])    # chance of writing a review, by feeling
writes = rng.random(100_000) < p_write[users - 1]     # True for users who write one
reviews = users[writes]

users.mean()                    # 3.82941    the true mean feeling (a parameter)
reviews.mean()                  # 3.651686807653575    what the review section shows
len(reviews)                    # 7944       about 8% of users wrote one
np.bincount(reviews, minlength=6)[1:] / len(reviews)   # share of 1..5 stars among reviews
# array([0.18668177, 0.08723565, 0.06558409, 0.20871098, 0.45178751])
```

`p_write[users - 1]` looks up each user's chance of writing (index 0 for 1 star, and so on); comparing a uniform random number with it is a coin flip with that probability, the Bernoulli trick from Day 37.

The true mean is about 3.8, but the review section shows something else, and the share of 1-star ratings is several times what it is among users. Nothing about the averaging was wrong. The *sample* was wrong.

Now the crucial point. Compare a small honest survey with an ever bigger pile of reviews:

```python
for n in (50, 500, 5_000):
    honest = rng.choice(users, size=n, replace=False).mean()     # ask n random users
    biased = rng.choice(reviews, size=n, replace=False).mean()   # read n random reviews
    print(f"n={n:5d}   random users: {honest:.3f}   reviews: {biased:.3f}   truth: {users.mean():.3f}")
# n=   50   random users: 3.940   reviews: 3.760   truth: 3.829
# n=  500   random users: 3.874   reviews: 3.564   truth: 3.829
# n= 5000   random users: 3.820   reviews: 3.656   truth: 3.829
```

The random sample's guess wanders around the truth and homes in on it. The review sample homes in too, but **on the wrong number**. Repeat each survey 2,000 times and plot the guesses:

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

def many_means(pool, n, reps=2_000):
    """Means of `reps` independent samples of size n, each drawn without replacement."""
    return np.array([rng.choice(pool, size=n, replace=False).mean() for _ in range(reps)])

fig, ax = plt.subplots(figsize=(10, 4.2))
bins = np.arange(2.99, 4.31, 0.02)   # means of whole stars move in steps of 1/n: keep bins aligned
for pool, n, color, alpha, label in [
        (users, 50, "#00897b", 0.35, "random users, n = 50"),
        (users, 500, "#00897b", 0.8, "random users, n = 500"),
        (reviews, 50, "#ef6c00", 0.35, "reviews only, n = 50"),
        (reviews, 500, "#ef6c00", 0.8, "reviews only, n = 500")]:
    ax.hist(many_means(pool, n), bins=bins, density=True, histtype="stepfilled",
            color=color, alpha=alpha, label=label)
ax.axvline(users.mean(), color="#e53935", ls="--", lw=1.5, label=f"true mean = {users.mean():.3f}")
ax.set_xlabel("the survey's estimate of the mean rating")
ax.set_ylabel("density")
ax.set_title("2,000 surveys each: more data shrinks the noise, not the bias")
ax.legend(frameon=False, loc="upper left")
fig.savefig("day43-bias-vs-noise.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Four histograms of survey estimates of the mean star rating. The two teal histograms, random users with n = 50 (wide) and n = 500 (narrow), are both centred on the dashed red true-mean line near 3.83. The two orange histograms, reviews only with n = 50 (wide) and n = 500 (narrow), are centred to the left, near 3.65, and the n = 500 one is narrow and nowhere near the truth](../../assets/img/phase-1/day43-bias-vs-noise.svg)
<figcaption>Two kinds of error. Noise (the width) shrinks as n grows; bias (how far the centre is from the truth) doesn't move at all.</figcaption>
</figure>

That picture is worth memorizing, because it separates the two ways an estimate goes wrong:

- **Noise** (also called **variance** of the estimator): the guesses scatter. Fix: more data.
- **Bias**: the guesses are centred on the wrong value. Fix: better data. More of the same data only makes you **more confidently wrong**.

The same thing happens in the real dataset if you only "survey" the category that's easiest for you, say you're a game developer who mostly looks at games:

```python
games = rated["category"].str.strip().str.title() == "Games"   # Day 17's category cleanup
population[games.to_numpy()].mean(), mu    # (4.283333333333333, 4.165584415584416)
```

Games are rated higher than the store as a whole, so a games-only sample overestimates the average app's rating no matter how many games you look at.

!!! warning "Common mistake: \"we have millions of rows, so bias doesn't matter\""
    Big data reduces noise, never bias. A model trained on millions of reviews learns the opinions of *people who write reviews*. A crash-rate estimate from users who opted in to sharing diagnostics describes *those* users, who may be on newer phones. Always ask "how did a row get into this table?" before asking "how many rows are there?"

!!! android "Android bridge: the Play In-App Review API forbids gating"
    Google Play's [in-app review API](https://developer.android.com/guide/playcore/in-app-review) lets you ask for a rating without leaving your app:

    ```kotlin
    val manager = ReviewManagerFactory.create(context)
    manager.requestReviewFlow().addOnCompleteListener { task ->
        if (task.isSuccessful) {
            manager.launchReviewFlow(activity, task.result)   // Play shows its own rating card
        }
    }
    ```

    Its guidelines tell you not to ask the user anything before the prompt, including opinion questions like "Do you like the app?". Through today's lens, that rule bans deliberate self-selection bias: an app that only prompts the users who said "yes" turns its public rating into the orange histogram above. The flip side is a bias you can't remove: the users who choose to rate when the card appears are still self-selected, which is one reason the store average and your own in-app survey (sent to *random* users) can disagree.

## The sampling distribution of the mean

Back to the Play Store population, where we know \( \mu \approx 4.166 \) and \( \sigma \approx 0.453 \). One survey of 25 apps gave one \( \bar{x} \). Imagine running the survey again and again, each time with 25 fresh random apps, and writing down every \( \bar{x} \). The histogram of all those \( \bar{x} \) values is the **sampling distribution of the mean**.

<figure class="diagram" markdown>
![Animation: on the left a grid of 40 dots stands for the 462 rated apps, with mu = 4.166; five dots light up in orange as one sample of n = 5, a panel lists their five ratings and their mean x-bar, and an orange block drops into a histogram on the right at that x-bar; this repeats for six samples, building a small histogram around the dashed red line at mu, and then a smooth teal bell curve of 10,000 sample means appears over it, labelled centre mu and width sigma over root n = 0.203](../../assets/img/phase-1/day43-sampling-machine.svg)
<figcaption>Animated: every sample gives one x̄; stack up the x̄ values of many samples and you get the sampling distribution.</figcaption>
</figure>

You can't do this in real life (you only get one survey), but in a simulation it's one loop:

```python
n, reps = 25, 10_000
means_25 = np.array([rng.choice(population, size=n, replace=False).mean() for _ in range(reps)])

means_25[:5]                           # array([4.088, 4.072, 4.22 , 4.252, 4.192])
means_25.mean(), mu                    # (4.165481199999999, 4.165584415584416)   centre: on the true mean
means_25.std(), sigma / np.sqrt(n)     # (0.08753485960781568, 0.09067462032585204)   spread: about sigma/sqrt(n) (a bit less: see below)
```

Three facts, each one a result from Week 6:

1. **Centre.** The sample means average out to \( \mu \). In symbols, \( \mathbb{E}[\bar{X}] = \mu \): the sample mean is **unbiased**. (Day 39, linearity of expectation.)
2. **Spread.** Their standard deviation is about \( \sigma/\sqrt{n} = 0.453/5 = 0.0907 \). (Day 39, the variance of an average.)
3. **Shape.** Their histogram is close to a bell curve, even though the ratings themselves are lumpy and capped at 5. (Day 40, the central limit theorem.)

The capital \( \bar{X} \) is the random variable "the mean of a sample I'm about to draw"; lower case \( \bar{x} \) is the number one particular sample gave. Same convention as Day 36's \( X \) and \( x \).

### The standard error

The spread of a statistic's sampling distribution is so important it has its own name: the **standard error** (SE). For the mean,

\[
\mathrm{SE}(\bar{x}) = \frac{\sigma}{\sqrt{n}}
\]

- \( \sigma \) is the population's standard deviation: how much *individual* apps differ.
- \( \sqrt{n} \) is how much averaging \( n \) of them calms that down.

Read the two names carefully, because they are different things. The **standard deviation** describes the spread of the *data* (one app's rating vs another's). The **standard error** describes the spread of an *estimate* (one survey's \( \bar{x} \) vs another's). The first stays the same however much data you collect; the second shrinks.

**Worked example.** With \( \sigma = 0.4534 \):

| \( n \) | \( \sqrt{n} \) | \( \mathrm{SE} = 0.4534/\sqrt{n} \) |
|---|---|---|
| 4 | 2 | 0.227 |
| 25 | 5 | 0.091 |
| 100 | 10 | 0.045 |
| 400 | 20 | 0.023 |

Each row quadruples \( n \) and only halves the SE. Precision is expensive: 10 times more precise costs 100 times more data.

Check the formula across sample sizes. One complication shows up as \( n \) gets close to \( N = 462 \): sampling **without** replacement from a finite population is a bit more precise than \( \sigma/\sqrt{n} \), because you can't pick the same app twice and in the extreme, \( n = N \), you'd have the whole population and zero error. The exact SE then gets a **finite population correction**:

\[
\mathrm{SE}_{\text{without replacement}} = \frac{\sigma}{\sqrt{n}}\sqrt{\frac{N - n}{N - 1}}
\]

```python
print("   n   with repl.   without   sigma/sqrt(n)   with correction")
for n in (5, 25, 100, 200, 400):
    with_r = rng.choice(population, size=(4_000, n)).mean(axis=1).std()       # replace=True is the default
    without = np.array([rng.choice(population, size=n, replace=False).mean() for _ in range(4_000)]).std()
    se = sigma / np.sqrt(n)
    fpc = se * np.sqrt((N - n) / (N - 1))
    print(f"{n:4d}   {with_r:9.4f}   {without:7.4f}   {se:13.4f}   {fpc:15.4f}")
#    n   with repl.   without   sigma/sqrt(n)   with correction
#    5      0.2013    0.2024          0.2028            0.2019
#   25      0.0917    0.0903          0.0907            0.0883
#  100      0.0456    0.0399          0.0453            0.0402
#  200      0.0322    0.0239          0.0321            0.0242
#  400      0.0225    0.0083          0.0227            0.0083
```

With replacement, the simulated spread matches \( \sigma/\sqrt{n} \); without replacement, it matches the corrected formula. When the sample is a small slice of the population (under about 5%, the usual rule of thumb), the correction is so close to 1 that everybody ignores it. That's why the size of the pot barely matters: a poll of 1,000 people is about as precise for a city of a million as for a country of 300 million.

Here are the sampling distributions for three sample sizes, and the standard error against \( n \):

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
for n, color in [(5, "#ef6c00"), (25, "#00897b"), (100, "#3949ab")]:
    m = rng.choice(population, size=(10_000, n)).mean(axis=1)
    ax1.hist(m, bins=np.arange(3.49, 4.81, 0.02), density=True, histtype="stepfilled",
             alpha=0.5, color=color, label=f"n = {n}:  SE = {sigma / np.sqrt(n):.3f}")
ax1.axvline(mu, color="#e53935", ls="--", lw=1.5, label=f"μ = {mu:.3f}")
ax1.set_xlabel("sample mean rating  x̄")
ax1.set_ylabel("density")
ax1.set_title("Sampling distribution of the mean (10,000 samples each)")
ax1.legend(frameon=False, loc="upper left")

ns = np.array([2, 4, 8, 16, 32, 64, 128, 256])
measured = [rng.choice(population, size=(4_000, n)).mean(axis=1).std() for n in ns]
ax2.loglog(ns, sigma / np.sqrt(ns), color="#e53935", ls="--", label=r"$\sigma/\sqrt{n}$")
ax2.loglog(ns, measured, "o", color="#3949ab", label="measured (with replacement)")
ax2.set_xlabel("sample size n (log scale)")
ax2.set_ylabel("standard error (log scale)")
ax2.set_title("4× the data, ½ the error")
ax2.legend(frameon=False)
fig.tight_layout()
fig.savefig("day43-standard-error.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![Left: three overlapping histograms of sample mean ratings, all centred on the dashed red line at mu = 4.166; the n = 5 histogram is wide (SE 0.203), the n = 25 one narrower (SE 0.091) and the n = 100 one a narrow peak (SE 0.045). Right: a log-log plot of standard error against sample size from 2 to 256; blue measured dots sit on a red dashed straight line of slope minus one half, sigma over root n](../../assets/img/phase-1/day43-standard-error.svg)
<figcaption>Every sample size is centred on μ; bigger samples are only narrower. On log-log axes, σ/√n is a straight line with slope −½.</figcaption>
</figure>

### Estimating the standard error from one sample

There's a catch in \( \sigma/\sqrt{n} \): it needs \( \sigma \), the population's standard deviation, which you don't know any more than you know \( \mu \). In practice you plug in the sample's own \( s \):

\[
\widehat{\mathrm{SE}} = \frac{s}{\sqrt{n}}
\]

So from **one** sample you get both a guess (\( \bar{x} \)) and a measure of how far off that guess typically is (\( \widehat{\mathrm{SE}} \)). Our first survey of 25 apps:

```python
se_hat = s / np.sqrt(len(sample))
x_bar, se_hat                  # (4.172, 0.09721453937897699)   the estimate and its estimated SE
sigma / np.sqrt(25)            # 0.09067462032585204   the true SE (we only know it because we cheated)
abs(x_bar - mu) / se_hat       # 0.06599408336003884   this survey missed by only 0.07 standard errors
```

A report would say "the mean rating is about 4.17, with a standard error of 0.10". (This particular survey happened to land very close to \( \mu \); a typical one misses by about one standard error.) Tomorrow, Day 44 turns that pair of numbers into a confidence interval: roughly, \( \bar{x} \pm 2\,\widehat{\mathrm{SE}} \) catches \( \mu \) in about 95% of surveys.

**Planning a sample size** runs the formula backwards. To get an SE of at most 0.05 on the mean rating, solve \( \sigma/\sqrt{n} \le 0.05 \):

\[
n \ge \left(\frac{\sigma}{0.05}\right)^2 = \left(\frac{0.4534}{0.05}\right)^2 \approx 82.2
\quad\Longrightarrow\quad n = 83
\]

You need a guess of \( \sigma \) for this, usually from a small pilot sample or last month's data.

!!! warning "Common mistake: shrinking the standard deviation instead of the standard error"
    "We collected more data, so the standard deviation went down" is a red flag. The sample standard deviation \( s \) estimates \( \sigma \), the spread of individual apps, and settles around 0.45 however big \( n \) gets. It is the **standard error** \( s/\sqrt{n} \) that goes down. If a chart's error bars shrink with \( n \), they are standard errors; if they don't, they are standard deviations. Labelling them is not optional.

## Why the sample variance divides by \( n - 1 \)

On Day 39 a warning box promised this explanation: `np.var` divides by \( n \), pandas' `.var()` by \( n - 1 \), and for a *sample* the \( n - 1 \) is right. Here's why.

The population variance measures squared distances from the **true mean** \( \mu \). But from a sample you don't know \( \mu \), so you measure from the sample's own mean \( \bar{x} \) instead. And \( \bar{x} \) is not a neutral stand-in: it is computed *from these very points*, so it sits right in the middle of them. In fact \( \bar{x} \) is the single number that makes the sum of squared distances to the points **as small as possible** (it's the least-squares centre, the 1-D version of Day 25's projection). Measured from \( \bar{x} \), any sample looks a little less spread out than it really is.

**Worked example.** Roll a fair die three times and get \( 1, 2, 6 \). The true mean is \( \mu = 3.5 \) and the true variance \( \sigma^2 = 35/12 \approx 2.917 \) (Day 39). The sample mean is \( \bar{x} = 9/3 = 3 \).

| Centre | Deviations | Squares | Sum of squares |
|---|---|---|---|
| \( \mu = 3.5 \) (the truth, unknown in practice) | \( -2.5, -1.5, 2.5 \) | \( 6.25, 2.25, 6.25 \) | **14.75** |
| \( \bar{x} = 3 \) (what we can compute) | \( -2, -1, 3 \) | \( 4, 1, 9 \) | **14** |

The sum around \( \bar{x} \) is smaller, as promised. It always is, and by an exact amount:

\[
\sum_{i=1}^{n} (x_i - \mu)^2 = \sum_{i=1}^{n} (x_i - \bar{x})^2 + n\,(\bar{x} - \mu)^2
\]

Check: \( 14.75 = 14 + 3 \times (3 - 3.5)^2 = 14 + 0.75 \). In words: the true spread equals the spread around \( \bar{x} \) **plus** a penalty for \( \bar{x} \) itself being off target.

<figure class="diagram" markdown>
![Animation: three die rolls 1, 2 and 6 sit on a number line; an orange centre marker c slides to 1.5, 2.5, 3, 3.5 and 4.5, with an orange segment from each point to it; on the right a parabola shows the sum of squared distances against c, with a dot moving along it showing 20.75, 14.75, 14, 14.75 and 20.75; the lowest point, 14, is at c = x-bar = 3 and is marked in teal, while the true mean mu = 3.5 gives 14.75, so measuring from x-bar always under-counts the spread](../../assets/img/phase-1/day43-bessel-centre.svg)
<figcaption>Animated: the sum of squared distances is smallest at the sample's own mean x̄, so measuring spread from x̄ under-counts it compared with the true mean μ.</figcaption>
</figure>

### How big is the shortfall?

Take the expected value of both sides of that identity, using two facts from Day 39:

- On the left, each \( (x_i - \mu)^2 \) has expected value \( \sigma^2 \) (that's the definition of variance), so the sum has expected value \( n\sigma^2 \).
- On the right, \( (\bar{x} - \mu)^2 \) is the squared miss of the sample mean, whose expected value is \( \mathrm{Var}(\bar{X}) = \sigma^2/n \). Times \( n \) gives exactly \( \sigma^2 \).

So

\[
n\sigma^2 = \mathbb{E}\Big[\sum (x_i - \bar{x})^2\Big] + \sigma^2
\quad\Longrightarrow\quad
\mathbb{E}\Big[\sum (x_i - \bar{x})^2\Big] = (n - 1)\,\sigma^2
\]

The sum of squares around \( \bar{x} \) is, on average, worth \( n - 1 \) copies of \( \sigma^2 \), not \( n \). Dividing it by \( n \) gives an estimate that is too small on average, by a factor \( (n-1)/n \). Dividing by \( n - 1 \) makes it right on average. That's **Bessel's correction**, and

\[
s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2
\]

is called the **unbiased** sample variance. For our three rolls: \( 14/3 \approx 4.67 \) dividing by \( n \), and \( 14/2 = 7 \) dividing by \( n - 1 \). (Both are far from 2.917; three rolls is a tiny sample. "Unbiased" is a promise about the average over many samples, not about any one.)

The \( n - 1 \) is also called the **degrees of freedom**: once you know \( \bar{x} \), only \( n - 1 \) of the deviations are free, because they must add up to zero (here \( -2 - 1 + 3 = 0 \); knowing two of them tells you the third).

```python
rolls = np.array([1, 2, 6])
rolls.var(), rolls.var(ddof=1)            # (4.666666666666667, 7.0)   divide by n, and by n - 1
((rolls - 3.5) ** 2).sum(), ((rolls - rolls.mean()) ** 2).sum()   # (14.75, 14.0)   around mu vs around x-bar
```

`ddof` stands for "delta degrees of freedom": NumPy divides by \( n - \text{ddof} \). Now the simulation: draw 200,000 samples of size \( n \) from a fair die and average each variance estimate.

```python
sigma2_die = 35 / 12
for n in (2, 3, 5, 10, 30):
    samples = rng.integers(1, 7, size=(200_000, n))     # 200,000 samples of n rolls each
    v0 = samples.var(axis=1).mean()                     # divide by n, averaged over samples
    v1 = samples.var(axis=1, ddof=1).mean()             # divide by n - 1
    print(f"n={n:3d}   ÷n: {v0:.4f}   (n-1)/n·σ²: {(n - 1) / n * sigma2_die:.4f}   ÷(n-1): {v1:.4f}   σ²: {sigma2_die:.4f}")
# n=  2   ÷n: 1.4597   (n-1)/n·σ²: 1.4583   ÷(n-1): 2.9193   σ²: 2.9167
# n=  3   ÷n: 1.9430   (n-1)/n·σ²: 1.9444   ÷(n-1): 2.9144   σ²: 2.9167
# n=  5   ÷n: 2.3306   (n-1)/n·σ²: 2.3333   ÷(n-1): 2.9132   σ²: 2.9167
# n= 10   ÷n: 2.6257   (n-1)/n·σ²: 2.6250   ÷(n-1): 2.9174   σ²: 2.9167
# n= 30   ÷n: 2.8197   (n-1)/n·σ²: 2.8194   ÷(n-1): 2.9169   σ²: 2.9167
```

Dividing by \( n \) lands exactly on \( \frac{n-1}{n}\sigma^2 \): half the true variance for pairs, and still 3% short at \( n = 30 \). Dividing by \( n - 1 \) hits \( \sigma^2 \) at every \( n \).

```python
ns = np.arange(2, 31)
avg0, avg1 = [], []
for n in ns:
    samples = rng.integers(1, 7, size=(50_000, n))
    avg0.append(samples.var(axis=1).mean())
    avg1.append(samples.var(axis=1, ddof=1).mean())

fig, ax = plt.subplots(figsize=(9, 4))
ax.axhline(sigma2_die, color="#e53935", ls="--", lw=1.5, label="true variance σ² = 35/12")
ax.plot(ns, (ns - 1) / ns * sigma2_die, color="#9e9e9e", lw=1.2, label="(n − 1)/n · σ²")
ax.plot(ns, avg0, "o", color="#ef6c00", ms=5, label="average of var( ) dividing by n")
ax.plot(ns, avg1, "s", color="#00897b", ms=5, label="average of var(ddof=1), dividing by n − 1")
ax.set_xlabel("sample size n")
ax.set_ylabel("average variance estimate")
ax.set_title("Bessel's correction: 50,000 samples of die rolls per n")
ax.set_ylim(1.2, 3.2)
ax.legend(frameon=False, loc="lower right")
fig.savefig("day43-bessel.svg", format="svg", bbox_inches="tight")
```

<figure class="diagram" markdown>
![A plot of the average variance estimate against sample size n from 2 to 30 for die rolls. The teal squares, dividing by n minus 1, sit on the dashed red line of the true variance 2.917 at every n. The orange dots, dividing by n, start at about 1.46 for n = 2 and climb towards the red line along a grey curve (n - 1)/n times sigma squared, still visibly below it at n = 30](../../assets/img/phase-1/day43-bessel.svg)
<figcaption>Dividing by n under-estimates the variance by the factor (n − 1)/n; dividing by n − 1 is right on average at every sample size.</figcaption>
</figure>

One honest footnote: \( s \), the square root of \( s^2 \), is still slightly **too small** on average, because the square root bends (Day 39's \( \mathbb{E}[g(X)] \ne g(\mathbb{E}[X]) \)). Nobody corrects for it in everyday work; the bias is tiny once \( n \) is more than a handful.

```python
s_small = rng.integers(1, 7, size=(200_000, 5)).std(axis=1, ddof=1)
s_small.mean(), np.sqrt(sigma2_die)       # (1.646637111592452, 1.707825127659933)   average s for n = 5 vs true sigma
```

!!! warning "Common mistake: mixing `ddof` conventions"
    `np.std(x)` divides by \( n \); `pd.Series.std()` divides by \( n - 1 \). The same column gives two different numbers, and for small samples the difference is real. Rule: describing a full population you actually have (as with `sigma` at the top of this page), use `ddof=0`; estimating a population's spread from a sample, use `ddof=1`. When the two disagree by an amount that matters, your sample is too small to trust either one much.

## Where this shows up in machine learning

Every dataset you train or test on is a sample of the world your model will face. Today's ideas, in ML clothes:

**A test set is a sample, so test accuracy has a standard error.** A model gets each test example right (1) or wrong (0), so accuracy is the mean of 0/1 values: a proportion \( \hat{p} \). For a 0/1 variable, \( \sigma = \sqrt{p(1 - p)} \) (the Bernoulli variance from Day 39), so

\[
\mathrm{SE}(\hat{p}) = \sqrt{\frac{p(1 - p)}{n}}
\]

**Worked example.** A model that is truly 90% accurate, tested on 200 images: \( \sqrt{0.9 \times 0.1 / 200} = \sqrt{0.00045} \approx 0.021 \). Its measured accuracy will typically be off by about 2 percentage points, just from *which* 200 images landed in the test set.

```python
p_true, n_test = 0.90, 200
acc = (rng.random((10_000, n_test)) < p_true).mean(axis=1)   # 10,000 test sets of 200 examples
acc[:5]                       # array([0.9  , 0.905, 0.89 , 0.885, 0.895])
acc.std(), np.sqrt(p_true * (1 - p_true) / n_test)
# (0.021426857999949516, 0.021213203435596423)   simulated vs formula
```

So if model A scores 91% and model B 89% on the same 200 images, you have not shown that A is better; the gap is about one standard error. Day 45 turns this into a proper test. To tell apart models that differ by 1 point, you need test sets in the thousands.

**Biased training data gives biased models.** A speech model trained mostly on one accent, a photo classifier trained on photos from high-end phone cameras, a recommender trained only on what the old recommender chose to show: each is the review-section histogram above, and no amount of extra data *of the same kind* fixes it. When the data a model sees in production differs from its training sample, that's called **distribution shift**, and it starts as a sampling question.

**Mini-batches are samples.** Day 39 showed a mini-batch gradient is a sample mean of per-example gradients with standard error \( \sigma_g/\sqrt{B} \). That's why training data is **shuffled** before batching: an unshuffled file sorted by class or date gives batches that are biased samples, not random ones.

**Bessel's correction in the wild.** PyTorch's `BatchNorm` layer normalizes each batch with the variance computed by dividing by \( n \), but the running variance it keeps for inference uses the \( n - 1 \) version; its documentation states both. Small details like this are why the same model can give slightly different numbers in two libraries.

!!! android "Android bridge: sample users, not events"
    Analytics and performance traces are often sampled on device to save battery and bandwidth, for example keeping 1% of sessions. Two lessons from today apply directly:

    ```kotlin
    // Decide once per user (stable), not with a fresh coin flip per event.
    fun inSample(userId: String, percent: Int): Boolean =
        Math.floorMod(userId.hashCode(), 100) < percent
    ```

    - **Pick the sampling unit on purpose.** Flipping a coin for every *event* gives heavy users far more rows than light users, so "average user behaviour" computed from those events is really "average heavy-user behaviour", a self-selection bias. Deciding per user (a stable hash of the user ID, so the same user is always in or always out) makes the sample a random sample of *users*. (`String.hashCode()` is stable across runs; for real IDs, a hash with better mixing, such as one based on a UUID's bits, spreads users more evenly.)
    - **Know your standard error before you read the dashboard.** 1% of 50,000 daily users is 500 users. A crash-free rate of 99% measured on 500 users has a standard error of \( \sqrt{0.99 \times 0.01/500} \approx 0.0045 \): day-to-day wobbles of half a percentage point are just noise.

---

## Exercises

1. **By hand.** A sample of five app ratings is \( 4.1, 3.8, 4.6, 4.5, 4.0 \). Compute \( \bar{x} \), the sum of squared deviations from \( \bar{x} \), the variance dividing by \( n \) and by \( n - 1 \), \( s \), and the estimated standard error \( s/\sqrt{n} \). Check with NumPy.
2. **Sample size.** Using the population's \( \sigma \), find the smallest \( n \) whose standard error on the mean rating is at most 0.03. Verify by simulating 5,000 samples of that size **with** replacement. Then say why the answer is awkward for this particular population (hint: compare with \( N \)), and compute the without-replacement SE at that \( n \) with the finite population correction.
3. **Sampling distribution of a proportion.** Let \( p \) be the share of rated apps in the population with `has_ads == "Yes"`. Simulate 10,000 samples of \( n = 40 \) apps (without replacement) and compute \( \hat{p} \) for each. Compare the mean of the \( \hat{p} \) values with \( p \), and their standard deviation with \( \sqrt{p(1-p)/n} \) and with its finite-population-corrected version.
4. **Spot the bias.** For each study, name the kind of bias and say which way it probably pushes the result: (a) a fitness app estimates how often people exercise from users who opened the app this week; (b) a blog post studies the 50 top-grossing games to find "what makes a game succeed"; (c) a team estimates app startup time from devices that sent a performance trace, but traces are only uploaded on Wi-Fi.
5. **Is every statistic unbiased?** Simulate 50,000 samples of \( n = 5 \) die rolls. Is the sample mean an unbiased estimate of \( \mu = 3.5 \)? Is the sample *maximum* an unbiased estimate of the population maximum, 6? Compute the average of each and explain the maximum's result in one sentence.
6. **Two kinds of error, on real data.** Draw 2,000 samples of \( n = 30 \) from the games-only ratings and 2,000 from the whole population. For each, report the average estimate minus \( \mu \) (the bias) and the standard deviation of the estimates (the standard error). Which error would more data fix?

??? success "Solutions"
    ```python
    # 1: x-bar = 21.0 / 5 = 4.2
    #    deviations: -0.1, -0.4, 0.4, 0.3, -0.2   squares: 0.01, 0.16, 0.16, 0.09, 0.04   sum = 0.46
    #    var (÷n) = 0.46 / 5 = 0.092    s² (÷(n-1)) = 0.46 / 4 = 0.115
    #    s = √0.115 ≈ 0.339    SE = 0.339 / √5 ≈ 0.152
    x1 = np.array([4.1, 3.8, 4.6, 4.5, 4.0])
    x1.mean(), ((x1 - x1.mean()) ** 2).sum()            # (4.2, 0.45999999999999996)
    x1.var(), x1.var(ddof=1)                            # (0.092, 0.11499999999999999)
    x1.std(ddof=1), x1.std(ddof=1) / np.sqrt(len(x1))   # (0.3391164991562634, 0.151657508881031)
    ```

    ```python
    # 2: sigma / √n ≤ 0.03  ⇒  n ≥ (0.4534 / 0.03)² ≈ 228.4  ⇒  n = 229
    n2 = int(np.ceil((sigma / 0.03) ** 2))
    n2                                                            # 229
    rng.choice(population, size=(5_000, n2)).mean(axis=1).std()   # 0.03030490312428603   with replacement: about 0.03
    sigma / np.sqrt(n2) * np.sqrt((N - n2) / (N - 1))             # 0.021299308638539016  without replacement, corrected
    ```
    229 apps is about half the population of 462, far beyond the 5% rule of thumb, so sampling without replacement is much more precise than \( \sigma/\sqrt{n} \) says: the corrected SE is about 0.021. For this tiny population you'd reach 0.03 with fewer apps. For a real store with millions of apps the correction would be invisible.

    ```python
    # 3
    has_ads = (rated["has_ads"] == "Yes").to_numpy()   # True/False per rated app
    p = has_ads.mean()
    p                                    # 0.6298701298701299
    p_hats = np.array([rng.choice(has_ads, size=40, replace=False).mean() for _ in range(10_000)])
    p_hats.mean(), p_hats.std()          # (0.6289775, 0.0722771886126598)
    se_p = np.sqrt(p * (1 - p) / 40)
    se_p, se_p * np.sqrt((N - 40) / (N - 1))   # (0.07634359000065356, 0.07304295612837988)
    ```
    The \( \hat{p} \) values average out to \( p \) (unbiased), and their spread matches the corrected formula; 40 apps is about 9% of the population, so the correction is noticeable but small.

    **4.** (a) Self-selection / convenience: people who opened a fitness app this week are the motivated ones, so exercise frequency is overestimated. (b) Survivorship: the games that did the same things and failed aren't in the top 50, so any "success factor" found may be just as common among the failures. (c) Self-selection by connectivity: Wi-Fi users skew towards home use and possibly newer, better-connected devices, so startup time is probably underestimated for users on slow networks or old phones. In all three, a bigger sample of the same kind doesn't help.

    ```python
    # 5
    rolls5 = rng.integers(1, 7, size=(50_000, 5))
    rolls5.mean(axis=1).mean()      # 3.499068   about 3.5: unbiased
    rolls5.max(axis=1).mean()       # 5.43358    well below 6: biased low
    ```
    The sample maximum can never be *above* the true maximum, only equal to it or below, so its average must fall short: it is biased low. (With 5 rolls, the chance that no 6 appears is \( (5/6)^5 \approx 0.40 \).) Not every statistic is unbiased; the sample mean is special.

    ```python
    # 6
    games_pop = population[games.to_numpy()]
    est_games = np.array([rng.choice(games_pop, size=30, replace=False).mean() for _ in range(2_000)])
    est_all = np.array([rng.choice(population, size=30, replace=False).mean() for _ in range(2_000)])
    est_games.mean() - mu, est_games.std()     # (0.1170172510822507, 0.0669078404100247)     games only
    est_all.mean() - mu, est_all.std()         # (-0.0011644155844159343, 0.07821080658493856) whole population
    ```
    The games-only survey has a bias of about +0.12. Its standard error is a little smaller than the whole-store one, because 30 games is a quarter of the 114-game population (the finite population correction again). More data (bigger \( n \)) shrinks the standard error of both, but the games-only bias stays about +0.12 forever; only sampling from the whole store fixes it.

## Resources

- [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/): the "Frequentist Inference" chapter animates point estimates and sampling distributions you can play with.
- [OpenIntro Statistics (free book)](https://www.openintro.org/book/os/): chapter 1 covers sampling principles and bias; chapter 5 covers point estimates, sampling variability and standard error.
- [Khan Academy: Statistics and probability](https://www.khanacademy.org/math/statistics-probability): the "Study design" unit covers sampling methods and bias, and "Sampling distributions" covers the sampling distribution of the mean and of a proportion.
- [Dive into Deep Learning: Probability and Statistics](https://d2l.ai/chapter_preliminaries/probability.html): estimators, bias and variance, written for ML.
- [NumPy docs: `Generator.choice`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html) (the `replace` argument) and [`numpy.var`](https://numpy.org/doc/stable/reference/generated/numpy.var.html) (the `ddof` argument).
- [pandas docs: `DataFrame.sample`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html).
- [Android developers: Google Play In-App Review API](https://developer.android.com/guide/playcore/in-app-review): the API and its design guidelines on when (and how not) to ask for a review.

## Checklist

- [ ] I can say which numbers are parameters and which are statistics, and use \( \mu, \sigma, p \) vs \( \bar{x}, s, \hat{p} \) correctly.
- [ ] I can name self-selection, survivorship and convenience bias, with an app example of each.
- [ ] I simulated a biased sample and saw that more data shrinks the noise but not the bias.
- [ ] I can draw a random sample with `rng.choice(..., replace=False)` and with `DataFrame.sample`.
- [ ] I built a sampling distribution by simulation and can state its centre, spread and shape.
- [ ] I know the difference between a standard deviation and a standard error, and can compute \( s/\sqrt{n} \) from one sample.
- [ ] I can plan a sample size from a target standard error, and know when the finite population correction matters.
- [ ] I can explain Bessel's correction with the "sum of squares is smallest at \( \bar{x} \)" argument, and I know what `ddof` does.
- [ ] I can compute the standard error of a test-set accuracy and say whether a 2-point gap between two models means anything.

**Next:** Day 44: Confidence intervals and the bootstrap (coming soon)
