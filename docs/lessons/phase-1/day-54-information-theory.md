# Day 54: Entropy, Cross-Entropy and KL Divergence

**Phase 1 · Week 8 · ~150 minutes**

## Objectives

- [ ] Compute the surprise of an event, \( \log_2(1/p) \), and explain why a logarithm is the right way to measure information.
- [ ] Compute entropy as the average surprise of a distribution, in bits and in nats, and say which distributions have the most and the least.
- [ ] Explain entropy as the best possible average code length, with a tiny Huffman-style code.
- [ ] Compute cross-entropy \( H(p, q) \) and KL divergence \( D_{\mathrm{KL}}(p \,\|\, q) \), and show that KL is not symmetric.
- [ ] Explain why cross-entropy is *the* classification loss, and why minimizing it is the same as maximum likelihood.
- [ ] Work a 3-class softmax + cross-entropy example by hand, derive its gradient \( \mathbf{q} - \mathbf{y} \), check it numerically, and use it to train a small classifier.
- [ ] Compute softmax and its loss without overflow, and sanity-check a fresh \( K \)-class classifier against a loss of \( \ln K \).
- [ ] Recognise the classic bugs: log of 0, bits vs nats, counts instead of probabilities, and softmax applied twice.

**Prerequisites.**

- [Day 12](day-12-layer-is-matmul.md): a layer is \( XW + \mathbf{b} \); today its outputs become logits.
- [Days 17–19](day-17-pandas-cleaning.md): the short Play Store cleaning (strip and title-case `category`, drop duplicates).
- [Day 29](day-29-derivatives.md): `central_diff(f, x, h=1e-5)` for checking derivatives.
- [Day 30](day-30-derivative-rules.md): the derivatives of \( e^x \) and \( \ln x \).
- [Day 31](day-31-chain-rule.md): the chain rule.
- [Day 32](day-32-partial-derivatives-gradient.md) and [Day 33](day-33-vector-matrix-gradients.md): partial derivatives, and the shape rule (the gradient has the parameter's shape).
- [Day 36](day-36-probability-basics.md) and [Day 37](day-37-distributions.md): distributions, and that probabilities of independent events multiply.
- [Day 39](day-39-expectation-variance.md): expected value and the law of large numbers.
- [Day 47](day-47-maximum-likelihood.md): maximum likelihood, and the coin with 7 heads in 10 flips.
- [Day 50](day-50-loss-functions.md): binary cross-entropy (log loss), and why accuracy has no gradient.
- [Day 51](day-51-gradient-descent.md): gradient descent, \( w \leftarrow w - \eta \nabla L \).
- [Day 53](day-53-convexity.md): convex functions have one minimum.

---

## The big idea

Your weather app pushes a notification: *"Sunny in the Atacama Desert today."* You learn almost nothing: it is sunny there nearly every day, and you would have guessed it. Then: *"Snow in the Atacama Desert."* That one is news. It was unlikely, so it surprised you, and it told you a lot.

Information theory, invented by Claude Shannon in 1948 to send messages down telephone wires, turns that feeling into a number. Four ideas build on each other today, each taught in its own section below:

| Idea | The question it answers | Symbol |
|---|---|---|
| **Surprise** | How much does seeing *one* event tell me? | \( \log_2 \frac{1}{p} \) |
| **Entropy** | How surprised am I *on average*, if I know the true odds? | \( H(p) \) |
| **Cross-entropy** | How surprised am I on average, if reality is \( p \) but I *believe* \( q \)? | \( H(p, q) \) |
| **KL divergence** | How much *extra* surprise does believing the wrong thing cost me? | \( D_{\mathrm{KL}}(p \,\|\, q) \) |

**Why an ML engineer cares.** A classifier is a weather app for labels: for every photo it announces "70% cat, 20% dog, 10% bird". Then the truth arrives. The loss used to train almost every classifier in existence, from logistic regression to the image model in your app to a large language model predicting the next token, is **how surprised the model was by the truth**. That is cross-entropy, and by the end of today you will have derived it, its gradient, and trained a classifier with it.

## Surprise: the information in one event

**What it is.** The **surprise** (also called *self-information*) of an event is a number that says how much you learn when that event happens. **Why it exists:** every quantity in this lesson is an average of surprises, and the training loss of a classifier is the surprise of the true label.

**Intuition first.** We want a function "surprise of an event with probability \( p \)" that behaves like the Atacama example. Three wishes:

1. **A certain event is no news.** If \( p = 1 \), the surprise is 0.
2. **Rarer means more surprising.** As \( p \) shrinks, the surprise grows.
3. **Independent surprises add.** Two independent coin flips both landing heads (probability \( \frac12 \cdot \frac12 = \frac14 \)) should be exactly twice as surprising as one head.

Wish 3 is the clue. Probabilities of independent events **multiply** (Day 36), but we want surprises to **add**. The function that turns multiplication into addition is the logarithm: \( \log(a \cdot b) = \log a + \log b \). So the surprise of an event with probability \( p \) is

\[
s = \log_2 \frac{1}{p} = -\log_2 p
\]

- \( p \) is the probability of the event that happened.
- \( \log_2 \) is the logarithm base 2: \( \log_2 8 = 3 \) because \( 2^3 = 8 \). Base 2 measures surprise in **bits**.
- \( \frac{1}{p} \) makes the number positive and growing as \( p \) shrinks (\( p \) is at most 1, so \( \log_2 p \le 0 \), and \( \log_2 \frac1p \ge 0 \)).

**Worked example.** One fair coin lands heads: \( p = \frac12 \), \( s = \log_2 2 = 1 \) bit. Two heads in a row: \( p = \frac14 \), \( s = \log_2 4 = 2 \) bits \( = 1 + 1 \). A fair die shows a 6: \( p = \frac16 \), \( s = \log_2 6 = 2.585 \) bits (to 3 decimals). The sun rising: \( p = 1 \), \( s = \log_2 1 = 0 \).

One **bit** is the information in one fair yes/no answer. What does a fractional bit like 2.585 mean? You can't ask 2.585 questions about one roll; a single roll needs 3 yes/no questions in the worst case. But if you ask cleverly about *batches* of rolls (say, guess five rolls at once: \( 6^5 = 7776 \) possibilities, which 13 yes/no questions can pin down, since \( 2^{13} = 8192 \)), the cost per roll falls to \( 13/5 = 2.6 \), and with bigger batches it approaches 2.585. A bit count is an *average cost per event*, so fractions are natural.

<figure class="diagram" markdown>
![Animation: the curve of surprise log2 of 1 over p falls from about 7 bits near p = 0 to 0 bits at p = 1; a red dot stops at p = 1 (0 bits, the sun rose), p = 1/2 (1 bit, a coin lands heads), p = 1/4 (2 bits, two heads), p = 1/8 (3 bits, three heads) and p = 1/64 (6 bits, six heads in a row), with a panel reminding that probabilities of independent events multiply while surprises add](../../assets/img/phase-1/day54-surprise-curve.svg)
<figcaption>Animated: each halving of the probability adds exactly one bit of surprise.</figcaption>
</figure>

```python
import numpy as np

def surprise_bits(p):
    """Information in an event of probability p, in bits."""
    return np.log2(1 / p)

surprise_bits(1.0)        # np.float64(0.0)      certain: no news
surprise_bits(0.5)        # np.float64(1.0)      one fair coin
surprise_bits(0.25)       # np.float64(2.0)      two fair coins
surprise_bits(1 / 6)      # np.float64(2.584962500721156)   one die roll
surprise_bits(1 / 1024)   # np.float64(10.0)     ten heads in a row
surprise_bits(0.999)      # np.float64(0.0014434168696688449)   almost certain: almost no news
```

**A note on how values display.** `np.log2` returns a **NumPy scalar** (a single number with a NumPy type, here `np.float64`, NumPy's 64-bit float, the same as Kotlin's `Double`). The ▶ Run button shows values with `repr()`, and the `repr` of a NumPy scalar is `np.float64(2.0)`, not `2.0`. A plain Python `float` shows as `2.0`. Both behave the same in arithmetic. Later, the `entropy` function wraps its result in `float(...)`, so it displays as a plain `2.0`; that difference is only cosmetic.

**Underneath.** `1 / p` divides; `np.log2` asks "2 to what power gives this?". For \( p = 1/1024 \), `1 / p` is 1024 and \( 2^{10} = 1024 \), so the answer is 10. Halving \( p \) doubles \( 1/p \), which adds exactly 1 to its base-2 log: that is the staircase in the animation.

### When surprise goes wrong: probability 0

What is the surprise of an event your model said was impossible? Try it:

```python
try:
    surprise_bits(0)                     # a Python int 0: plain Python division
except ZeroDivisionError as e:
    print(type(e).__name__ + ":", e)
# ZeroDivisionError: division by zero

surprise_bits(np.float64(0.0))           # a NumPy zero: no exception, but a warning
# np.float64(inf)
# (also prints: RuntimeWarning: divide by zero encountered in scalar divide)
```

`1 / 0` with a Python `int` raises `ZeroDivisionError`, like Kotlin's `1 / 0` on `Int` throws `ArithmeticException`. With a NumPy float, division by zero follows IEEE 754 (as Kotlin's `1.0 / 0.0` does): the answer is infinity, and NumPy adds a `RuntimeWarning`. Mathematically, infinity is the honest answer: an event you believed impossible that then happens is *infinitely* surprising. In a training loss that infinity turns into `nan` gradients, so later sections show how real code keeps probabilities away from exactly 0.

### Bits or nats

The base of the logarithm only sets the **unit**, like metres versus feet. Base 2 gives bits; the natural log \( \ln \) (base \( e \), about 2.718, Day 30) gives **nats**. Information theory books use bits; ML code uses nats, because `np.log` is the natural log and the derivative \( \frac{d}{dx}\ln x = \frac1x \) has no extra constant. Converting is one multiplication: \( \ln x = \ln 2 \cdot \log_2 x \), so 1 bit \( = \ln 2 = 0.693 \) nats (to 3 decimals).

```python
np.log(1 / 0.5)              # np.float64(0.6931471805599453)   one coin flip, in nats
np.log(1 / 0.5) / np.log(2)  # np.float64(1.0)                  ...converted back to bits
```

!!! warning "Common mistake: comparing bits with nats"
    A PyTorch training run prints `loss: 0.6931` for a binary classifier. A textbook says "a coin-flip guess costs 1 bit". It looks like the model beats a coin flip. It doesn't: the two numbers are the same thing in different units.

    ```python
    loss_nats = np.float64(0.6931471805599453)   # what PyTorch printed (it uses ln)
    loss_nats / np.log(2)                        # np.float64(1.0)   = exactly 1 bit: a coin-flip guess
    np.log(2), np.log2(2)                        # (np.float64(0.6931471805599453), np.float64(1.0))
    ```

    Recognise it by a loss suspiciously close to 0.693 (\( \ln 2 \)), 1.0986 (\( \ln 3 \)) or 2.303 (\( \ln 10 \)): those are "uniform guess" values in nats. Fix: convert one side, dividing nats by \( \ln 2 \) to get bits. Everything below says which unit it uses.

## Entropy: average surprise

**What it is.** **Entropy** is the average surprise of a whole distribution: before the event happens, how surprised do you expect to be? **Why it exists:** it measures how unpredictable something is, it is the best possible compression rate, and it is the floor that no classifier's loss can go below.

**Intuition first.** A coin that always lands heads never surprises you: entropy 0. A fair coin surprises you every time by 1 bit: entropy 1. A coin that lands heads 90% of the time is in between: mostly boring, occasionally a big surprise. Averaging those surprises is an expected value (Day 39): each outcome's surprise, weighted by its probability.

\[
H(p) = \sum_{x} p(x) \log_2 \frac{1}{p(x)}
\]

- \( p \) is now a whole distribution: a list of probabilities \( p(x) \), one per outcome \( x \), adding up to 1.
- \( \sum_x \) adds over every outcome.
- \( H \) is the letter Shannon chose; it echoes Boltzmann's H in physics.
- An outcome with \( p(x) = 0 \) contributes nothing: it never happens, so its surprise is never paid. (Mathematically \( 0 \cdot \log \frac10 \) is undefined, and the convention is to count it as 0.)

**Worked examples.**

- **Fair coin**, \( p = [\frac12, \frac12] \): \( H = \frac12 \cdot 1 + \frac12 \cdot 1 = 1 \) bit.
- **Biased coin**, \( p = [0.9, 0.1] \): \( H = 0.9 \log_2 \frac{1}{0.9} + 0.1 \log_2 \frac{1}{0.1} = 0.9 \cdot 0.152 + 0.1 \cdot 3.322 = 0.137 + 0.332 = 0.469 \) bits. Usually heads, so usually unsurprising.
- **Fair die**: six outcomes of \( \frac16 \), each with surprise 2.585, so \( H = 2.585 \) bits.
- **Which tab a user opens first** in an app with four tabs, measured from analytics: Home 50%, Search 25%, Library 12.5%, Settings 12.5%:

\[
H = 0.5 \cdot 1 + 0.25 \cdot 2 + 0.125 \cdot 3 + 0.125 \cdot 3 = 0.5 + 0.5 + 0.375 + 0.375 = 1.75 \text{ bits}
\]

```python
def entropy(p, base=2):
    """Entropy of a discrete distribution p (bits by default; base=np.e for nats)."""
    p = np.asarray(p, dtype=float)      # accept lists, arrays or pandas Series
    p = p[p > 0]                        # drop zero-probability outcomes: 0 · log(1/0) counts as 0
    return float(np.sum(p * np.log(1 / p)) / np.log(base))   # log in any base = ln / ln(base)

entropy([0.5, 0.5])                 # 1.0
entropy([0.9, 0.1])                 # 0.46899559358928133
entropy([0.99, 0.01])               # 0.08079313589591128
entropy([1.0, 0.0])                 # 0.0    no uncertainty at all
entropy(np.full(6, 1 / 6))          # 2.584962500721156
tabs = np.array([0.5, 0.25, 0.125, 0.125])
entropy(tabs)                       # 1.75
entropy(np.full(4, 0.25))           # 2.0    four equally likely tabs
entropy(tabs, base=np.e)            # 1.2130075659799042   the same thing in nats
```

Line by line: `np.asarray(p, dtype=float)` turns whatever you pass into a float array. `p[p > 0]` is a boolean mask (Day 8): `p > 0` is an array of `True`/`False`, and indexing with it keeps only the `True` positions. `np.log(1 / p)` is the surprise of each outcome in nats, `p * ...` weights it, `np.sum` adds, and dividing by `np.log(base)` converts to the requested unit (by default \( \ln 2 \), giving bits).

Two facts jump out:

- **Entropy is 0 when one outcome is certain**, and grows as outcomes become harder to guess.
- **Entropy is largest for the uniform distribution.** Four outcomes can have at most \( \log_2 4 = 2 \) bits, reached only when all four are equally likely. The tab distribution, being lopsided, has 1.75. In general, \( K \) outcomes have at most \( \log_2 K \) bits. **Why:** any lopsidedness makes some outcome easier to guess, and a guessable outcome surprises you less on average. With all outcomes equal there is nothing to lean on. The coin plot below shows this for \( K = 2 \): the peak is exactly at 50/50.

**Underneath: entropy really is an average.** Because entropy is an expected value, the law of large numbers (Day 39) says the average surprise of many simulated events settles to it:

```python
rng = np.random.default_rng(54)
opens = rng.choice(4, size=100_000, p=tabs)       # 100,000 simulated first taps: tab numbers 0..3
opens[:8]                                         # array([0, 1, 0, 1, 0, 0, 1, 0])
tabs[opens][:8]                                   # array([0.5 , 0.25, 0.5 , 0.25, 0.5 , 0.5 , 0.25, 0.5 ])
surprise_bits(tabs[opens]).mean()                 # np.float64(1.74719)   average surprise, close to H = 1.75
```

`rng.choice(4, size=..., p=tabs)` draws tab numbers 0–3 with the given probabilities. `tabs[opens]` is integer-array indexing (Day 8): for each simulated tap it looks up that tab's probability, so the result has one probability per tap. Their surprises averaged over 100,000 taps give 1.747, within a few thousandths of 1.75; more taps would get closer.

### When entropy goes wrong: counts instead of probabilities

The formula assumes its input sums to 1. Pass raw counts and you get nonsense, with no error:

```python
counts = np.array([5, 3, 2])              # e.g. 5 apps in Games, 3 in Tools, 2 in Finance
entropy(counts)                           # -18.364527976600282   a NEGATIVE entropy: impossible
entropy(counts / counts.sum())            # 1.4854752972273346    the fix: divide by the total first
```

With counts, `np.log(1 / 5)` is negative (because \( 1/5 < 1 \)), so every term is negative. **Recognise it** by an entropy below 0 or above \( \log_2 K \); both are impossible for a real distribution. In pandas, `value_counts(normalize=True)` gives probabilities directly.

The other classic bug is dropping the `p > 0` mask:

```python
def entropy_no_mask(p):
    p = np.asarray(p, dtype=float)
    return np.sum(p * np.log2(1 / p))     # no mask

entropy_no_mask([1.0, 0.0])               # np.float64(nan)   should be 0.0
# (also prints two RuntimeWarnings: divide by zero encountered in divide,
#  and invalid value encountered in multiply)
```

`1 / 0.0` is `inf`, \( \log_2 \infty = \infty \), and \( 0 \times \infty \) is `nan` (not a number) in floating point. One `nan` in a sum makes the whole sum `nan`. The mask applies the "\( 0 \cdot \log \frac10 = 0 \)" convention by removing those terms before they can poison the sum.

### The entropy of a coin

Here is entropy for every possible coin, from always-tails to always-heads. It is the shape every ML engineer should recognise: zero at the certain ends, highest at 50/50.

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

ph = np.linspace(0.001, 0.999, 500)               # probability of heads
H_coin = np.array([entropy([a, 1 - a]) for a in ph])

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(ph, H_coin, color="#3949ab", lw=2.5)
for a, txt, spot, ha in [(0.5, "fair coin: 1 bit", (0.5, 1.1), "center"),
                         (0.9, "0.9 / 0.1: 0.469 bits", (0.86, 0.4), "right")]:
    h = entropy([a, 1 - a])
    ax.plot(a, h, "o", color="#e53935", ms=7)
    ax.annotate(txt, (a, h), xytext=spot, ha=ha, color="#e53935")
ax.set_xlabel("probability of heads")
ax.set_ylabel("entropy H (bits)")
_ = ax.set_ylim(0, 1.25)          # assigned to _ so the Run button doesn't echo the returned limits
ax.set_title("Entropy of a coin: zero when certain, highest at 50/50")
ax.grid(alpha=0.3)
fig.savefig("day54-binary-entropy.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![A blue arch-shaped curve of the entropy of a coin against the probability of heads: it starts near 0 at probability 0, rises to a peak of 1 bit at probability 0.5, marked with a red dot labelled fair coin: 1 bit, and falls back to near 0 at probability 1; a second red dot at 0.9 is labelled 0.9 / 0.1: 0.469 bits](../../assets/img/phase-1/day54-binary-entropy.svg)
<figcaption>The entropy of a yes/no outcome. A model that answers "50/50" is maximally unsure; one that answers "100%" claims to be certain.</figcaption>
</figure>

### Entropy of a real column

The same function measures how unpredictable a column of data is. Load the Play Store dataset and apply the short cleaning from Days 17–19:

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")
apps = apps.drop_duplicates().assign(
    category=lambda d: d["category"].str.strip().str.title(),
)
cat_p = apps["category"].value_counts(normalize=True)   # share of each category, sums to 1
cat_p.round(3).to_dict()
# {'Games': 0.246, 'Tools': 0.16, 'Education': 0.142, 'Productivity': 0.133,
#  'Photography': 0.1, 'Health & Fitness': 0.083, 'Music & Audio': 0.073, 'Finance': 0.062}
entropy(cat_p), np.log2(len(cat_p))     # (2.864538795702962, np.float64(3.0))
```

No internet? Download the file from the [Datasets page](../../datasets.md) and pass the local path to `read_csv`. (Tested on pandas 3.0.)

Eight categories could carry at most \( \log_2 8 = 3 \) bits; the real mix has 2.865, because Games is more common than the rest. Keep this number: later today it becomes the score to beat for a model that predicts an app's category.

### Entropy is the best possible compression

**What it is.** Shannon's source-coding theorem says entropy is the **fewest bits per symbol any code can average**. **Why it matters:** it is where the word "bits" comes from, and it gives "surprise" a physical meaning: the length of a codeword.

Say you log each user's first tab to a server and want short messages. A fixed-length code needs 2 bits per tap (00, 01, 10, 11). A variable-length code gives the common tab a short codeword and rare tabs long ones: Home = `0`, Search = `10`, Library = `110`, Settings = `111`. No codeword is the start of another (a **prefix-free code**), so a stream like `0100110` decodes without separators: read bits until they match a codeword, emit it, repeat (Home, Search, Home, Library). Its average length is \( 0.5 \cdot 1 + 0.25 \cdot 2 + 0.125 \cdot 3 + 0.125 \cdot 3 = 1.75 \) bits, exactly the entropy. Here each codeword is exactly \( \log_2 \frac1p \) bits long, the outcome's surprise; that is only possible because these probabilities are powers of \( \frac12 \). For other probabilities, \( \log_2 \frac1p \) is not a whole number of bits, and **Huffman coding**, the algorithm that builds the best prefix-free code for any distribution (it repeatedly merges the two least likely symbols into a tree), gets an average length within 1 bit of the entropy. The DEFLATE format used by ZIP, gzip and PNG, and baseline JPEG, all end with a Huffman code.

```python
code = {0: "0", 1: "10", 2: "110", 3: "111"}       # Home, Search, Library, Settings
stream = "".join(code[t] for t in opens)            # encode all 100,000 simulated taps as one string
stream[:12]                                         # '010010001001'
len(stream) / len(opens)                            # 1.74719   bits per tap, vs 2 for fixed length
```

`"".join(...)` glues strings together with nothing between them, and `code[t] for t in opens` is a generator expression (like a lazy `opens.map { code[it] }` in Kotlin). The bits per tap equals the average surprise from before *to the last digit*, because each codeword's length is that tap's surprise. (`len(stream) / len(opens)` divides two Python `int`s, so the result is a plain `float`, displayed without `np.float64`.)

## Cross-entropy: surprise when you believe the wrong thing

**What it is.** **Cross-entropy** is your average surprise when events come from reality \( p \) but you judge them with your belief \( q \). **Why it exists:** you never know \( p \); a model *is* a \( q \). Cross-entropy scores a model against reality, and it is the loss you will minimize in every classifier.

**Intuition first.** You moved from London to the Atacama but your weather intuition is still London's. Every sunny day surprises you a little (you expected rain); you pay extra surprise every day until your belief adapts.

\[
H(p, q) = \sum_x p(x) \log_2 \frac{1}{q(x)}
\]

Compare it with entropy: the weights \( p(x) \) are still reality's (that's how often each outcome *actually* happens), but the surprise \( \log_2 \frac{1}{q(x)} \) is computed from your belief. If you believe \( q(x) \) is tiny and it happens often, you pay a lot.

**Worked example.** The tab analytics are \( p = [0.5, 0.25, 0.125, 0.125] \). A new teammate, who hasn't seen them, assumes every tab is equally likely: \( q = [0.25, 0.25, 0.25, 0.25] \). Every surprise is \( \log_2 4 = 2 \) bits, so \( H(p, q) = 2 \) bits, more than \( H(p) = 1.75 \). Someone who got it the wrong way round, \( q = [0.125, 0.125, 0.25, 0.5] \), does worse:

\[
H(p, q) = 0.5 \cdot 3 + 0.25 \cdot 3 + 0.125 \cdot 2 + 0.125 \cdot 1 = 1.5 + 0.75 + 0.25 + 0.125 = 2.625 \text{ bits}
\]

In code terms: if you design your codewords for the believed \( q \) but encode data that really follows \( p \), you average \( H(p, q) \) bits per symbol.

```python
def cross_entropy(p, q, base=2):
    """Average surprise when outcomes follow p but you measure them with belief q."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = p > 0                                      # outcomes that never happen cost nothing
    with np.errstate(divide="ignore"):             # q = 0 for a real outcome: inf, no warning
        return float(np.sum(p[m] * np.log(1 / q[m])) / np.log(base))

cross_entropy(tabs, np.full(4, 0.25))             # 2.0
cross_entropy(tabs, [0.125, 0.125, 0.25, 0.5])    # 2.625
cross_entropy(tabs, [0.4, 0.3, 0.15, 0.15])       # 1.7794468445267844   close guess, close to 1.75
cross_entropy(tabs, tabs)                         # 1.75   = entropy(tabs)
```

`np.errstate(divide="ignore")` silences NumPy's divide-by-zero warning, but only inside the `with` block (like a scoped setting that is restored when the block ends). We still get the correct answer, infinity, when the model gives probability 0 to something that happens; we only want to avoid a warning printed every time.

**Underneath: Gibbs' inequality.** The pattern above is a theorem: **\( H(p, q) \ge H(p) \), with equality only when \( q = p \)**. The coding view explains it. \( H(p) \) is the shortest possible average code length for data from \( p \); a code designed for the wrong distribution \( q \) is one more code among all possible ones, so it can't beat the best one. Check it on a thousand random pairs of distributions:

```python
P = rng.dirichlet(np.ones(5), size=1000)    # 1000 random distributions over 5 outcomes
Q = rng.dirichlet(np.ones(5), size=1000)    # (each row is non-negative and sums to 1)
gaps = np.array([cross_entropy(p, q) - entropy(p) for p, q in zip(P, Q)])
gaps.min() >= 0, gaps.min().round(4)        # (np.True_, np.float64(0.0366))
```

`rng.dirichlet` draws random probability vectors; you only need to know that each row is a valid distribution. Iterating over a 2-D array gives its rows, so `zip(P, Q)` walks the two matrices row by row in step, pairing row \( i \) of `P` with row \( i \) of `Q`. Every one of the 1000 gaps is positive. (`>=` on NumPy values gives a NumPy boolean, shown as `np.True_`.)

## KL divergence: the price of being wrong

**What it is.** The **Kullback–Leibler divergence** of \( q \) from \( p \) is the gap you measured above: the extra surprise you pay for believing \( q \) when the truth is \( p \). **Why it exists:** it isolates the part of the loss the model can fix. It also appears directly in variational autoencoders, knowledge distillation and the fine-tuning of large language models.

\[
D_{\mathrm{KL}}(p \,\|\, q) = H(p, q) - H(p) = \sum_x p(x) \log_2 \frac{p(x)}{q(x)}
\]

(The second form comes from subtracting the sums term by term: \( \log \frac1q - \log \frac1p = \log \frac pq \).) Read \( D_{\mathrm{KL}}(p \,\|\, q) \) as "the KL divergence of \( q \) from \( p \)": how far the belief \( q \) is from the truth \( p \), with the comparison averaged using \( p \)'s weights. From the examples: the uniform guess has \( D_{\mathrm{KL}} = 2 - 1.75 = 0.25 \) bits, the backwards guess \( 2.625 - 1.75 = 0.875 \) bits, and \( q = p \) has 0.

<figure class="diagram" markdown>
![Animation: four pairs of bars for the tabs Home, Search, Library and Settings; the solid blue bars p = 0.5, 0.25, 0.125, 0.125 stay fixed while the orange outlined bars q step through the wrong way round guess 0.125, 0.125, 0.25, 0.5, then uniform 0.25 each, then a close guess 0.4, 0.3, 0.15, 0.15, then exactly p; a panel shows entropy H(p) = 1.750 bits plus KL of 0.875, 0.250, 0.029 and 0.000 bits giving cross-entropy 2.625, 2.000, 1.779 and 1.750, drawn as a teal bar of fixed length with a red extension that shrinks to nothing](../../assets/img/phase-1/day54-cross-entropy-bars.svg)
<figcaption>Animated: the teal part (entropy) is set by reality; training can only remove the red part (KL divergence).</figcaption>
</figure>

```python
def kl_divergence(p, q, base=2):
    """Extra bits paid for believing q when the truth is p."""
    return cross_entropy(p, q, base) - entropy(p, base)

kl_divergence(tabs, np.full(4, 0.25))           # 0.25
kl_divergence(tabs, [0.4, 0.3, 0.15, 0.15])     # 0.029446844526784366
kl_divergence(tabs, tabs)                       # 0.0
```

### KL is not a distance: order matters

It is called a *divergence*, not a distance, because swapping the arguments changes the answer. A fair coin \( p = [0.5, 0.5] \) and a heavily biased one \( q = [0.9, 0.1] \):

\[
D_{\mathrm{KL}}(p \,\|\, q) = 0.5 \log_2 \tfrac{0.5}{0.9} + 0.5 \log_2 \tfrac{0.5}{0.1} = 0.5(-0.848) + 0.5(2.322) = 0.737 \text{ bits}
\]
\[
D_{\mathrm{KL}}(q \,\|\, p) = 0.9 \log_2 \tfrac{0.9}{0.5} + 0.1 \log_2 \tfrac{0.1}{0.5} = 0.9(0.848) + 0.1(-2.322) = 0.531 \text{ bits}
\]

```python
fair, biased = [0.5, 0.5], [0.9, 0.1]
kl_divergence(fair, biased), kl_divergence(biased, fair)
# (0.7369655941662063, 0.5310044064107187)
kl_divergence(fair, [1.0, 0.0]), kl_divergence([1.0, 0.0], fair)
# (inf, 1.0)
```

The last line is the extreme case. If reality is a fair coin but your model swears tails is **impossible** (\( q = 0 \)), the first tails is infinitely surprising: \( D_{\mathrm{KL}} = \infty \). The other way round costs only 1 bit. So \( D_{\mathrm{KL}}(p \,\|\, q) \) punishes a model hardest for giving near-zero probability to things that really happen. That is exactly the property you want in a training loss, and the reason the arguments go in the order **truth first, model second**.

!!! warning "Common mistake: swapping the arguments, or taking the log of 0"
    `kl_divergence(model, truth)` is a different quantity from `kl_divergence(truth, model)`, as the 0.737 vs 0.531 above shows; libraries document which order they expect (PyTorch's `nn.KLDivLoss`, for example, takes the model's **log**-probabilities as its first argument and the target as its second).

    The second trap: a model probability of exactly 0 for an outcome that occurs.

    ```python
    np.log(0.0)                              # np.float64(-inf)
    # (also prints: RuntimeWarning: divide by zero encountered in log)
    np.log(np.clip(0.0, 1e-12, 1.0))         # np.float64(-27.631021115928547)   clipped: large but finite
    ```

    A `-inf` in a loss becomes `inf` or `nan` in the gradients, and one `nan` spreads to every weight it touches: the training loss prints `nan` from that step on. Recognise it by that `RuntimeWarning` or a `nan` loss. Fix: work with logits and `log_softmax` (later today), or, if you must take the log of a probability, clip it first as above. `np.clip(x, lo, hi)` replaces anything below `lo` with `lo` and above `hi` with `hi`.

## Cross-entropy is the classification loss

**What it is.** For a classifier, the loss on one example is the cross-entropy between the true label and the model's predicted probabilities, which comes out as \( -\ln(\text{probability given to the true class}) \). **Why it exists:** it rewards putting probability on the right answer, punishes confident mistakes without limit, has a gradient everywhere (accuracy doesn't, Day 50), and equals maximum likelihood.

In classification, each training example has one true label. The "true distribution" for that example puts all its probability on the right class: a **one-hot** vector \( \mathbf{y} \) (all zeros except a single 1 at the true class), such as \( [1, 0, 0] \) for "cat" among (cat, dog, bird). The model outputs a distribution \( \mathbf{q} \), such as \( [0.7, 0.2, 0.1] \). Their cross-entropy, in nats (ML's unit, so \( \ln \)), has only one non-zero term:

\[
H(\mathbf{y}, \mathbf{q}) = \sum_k y_k \ln \frac{1}{q_k} = 1 \cdot \ln \frac{1}{q_{\text{cat}}} = -\ln q_{\text{cat}}
\]

So the loss for one example is **the model's surprise at the true label**: \( -\ln 0.7 = 0.357 \) nats here. Had the truth been "bird", the loss would be \( -\ln 0.1 = 2.303 \). For a dataset you average over the examples: \( L = \frac1n \sum_{i} -\ln q_{i, c_i} \), where \( c_i \) is example \( i \)'s true class and \( q_{i, c_i} \) the probability the model gave it. We average rather than sum so that the loss doesn't grow with the size of the batch: a loss of 0.36 means the same thing for 32 examples or 10,000, and the learning rate you picked keeps working when you change the batch size. And because \( H(\mathbf{y}) = 0 \) for a one-hot vector (no uncertainty), this loss is also exactly \( D_{\mathrm{KL}}(\mathbf{y} \,\|\, \mathbf{q}) \).

```python
y = np.array([1.0, 0.0, 0.0])                  # truth: cat
q = np.array([0.7, 0.2, 0.1])                  # model: 70% cat
cross_entropy(y, q, base=np.e), -np.log(0.7)   # (0.3566749439387324, np.float64(0.35667494393873245))
-np.log(0.1)                                   # np.float64(2.3025850929940455)   if the truth had been bird
```

(The two values differ in the 17th digit only because they are computed by different sequences of floating-point steps.)

**It is the binary cross-entropy from Day 50.** With two classes, a model that outputs \( \hat{y} \) = "probability of class 1" is the distribution \( [\hat{y}, 1 - \hat{y}] \), and the label \( y \in \{0, 1\} \) is \( [y, 1 - y] \). Their cross-entropy is \( -[\,y \ln \hat{y} + (1 - y)\ln(1 - \hat{y})\,] \), Day 50's log loss, term for term:

```python
y1, y_hat = 1, 0.8
-(y1 * np.log(y_hat) + (1 - y1) * np.log(1 - y_hat))   # np.float64(0.2231435513142097)   Day 50's formula
cross_entropy([y1, 1 - y1], [y_hat, 1 - y_hat], base=np.e)   # 0.22314355131420976   the same, up to the last digit
```

**It is maximum likelihood (Day 47).** Maximum likelihood picks the model that makes the observed data \( x_1, \dots, x_n \) most probable: maximize \( \prod_i q(x_i) \). Taking logs (which turns the product into a sum and doesn't move the maximum) and dividing by \( n \), that means maximizing \( \frac1n \sum_i \ln q(x_i) \), which is the same as **minimizing** \( \frac1n \sum_i -\ln q(x_i) \), the average surprise of the data.

Now the key step: **group equal outcomes**. If outcome \( x \) appears \( n_x \) times in the data, its surprise \( -\ln q(x) \) appears \( n_x \) times in the sum:

\[
\frac1n \sum_{i=1}^{n} -\ln q(x_i) = \sum_x \frac{n_x}{n}\,\bigl(-\ln q(x)\bigr) = H(p_{\text{data}}, q)
\]

where \( p_{\text{data}}(x) = n_x / n \) is the data's **empirical distribution** (how often each outcome occurred). So the average negative log-likelihood *is* a cross-entropy, with the data's frequencies as the truth. **Worked by hand** on Day 47's coin, 7 heads and 3 tails, for the model "heads with probability \( \theta = 0.6 \)": the ten flips contribute seven copies of \( -\ln 0.6 \) and three of \( -\ln 0.4 \), so the average is \( 0.7 \cdot (-\ln 0.6) + 0.3 \cdot (-\ln 0.4) = 0.7 \cdot 0.511 + 0.3 \cdot 0.916 = 0.632 \) nats.

```python
flips = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1])     # 7 heads (1) in 10 flips
p_data = np.array([flips.mean(), 1 - flips.mean()])  # empirical distribution [0.7, 0.3]
theta = 0.6
-np.mean(np.log(np.where(flips == 1, theta, 1 - theta)))   # np.float64(0.63246515619844)   flip by flip
0.7 * -np.log(theta) + 0.3 * -np.log(1 - theta)            # np.float64(0.63246515619844)   grouped

thetas = np.linspace(0.01, 0.99, 99)                 # 0.01, 0.02, ..., 0.99
ce = np.array([cross_entropy(p_data, [t, 1 - t], base=np.e) for t in thetas])
nll = np.array([-np.mean(np.log(np.where(flips == 1, t, 1 - t))) for t in thetas])
np.allclose(ce, nll)                                 # True   cross-entropy = average negative log-likelihood
thetas[ce.argmin()].round(2), ce.min().round(4)      # (np.float64(0.7), np.float64(0.6109))
entropy(p_data, base=np.e)                           # 0.6108643020548935   = the minimum
```

`np.where(flips == 1, theta, 1 - theta)` builds an array with \( \theta \) where the flip was heads and \( 1 - \theta \) where it was tails: the probability the model gave each flip. The grid search finds the smallest cross-entropy at \( \theta = 0.7 = k/n \), the maximum-likelihood answer from Day 47, and the smallest value equals the entropy of the data, because there \( q = p_{\text{data}} \) and the KL part is 0. (`np.allclose` returns a plain Python `bool`, so it shows as `True`, not `np.True_`.)

So "minimize cross-entropy", "minimize KL to the data" and "maximize likelihood" are three names for one training objective. Because \( H(p_{\text{data}}) \) is fixed by the data, the model can only shrink the KL part, which is the animation above.

**A baseline to beat.** Predict an app's category with no features at all. A model that always says "each of the 8 categories equally likely" scores \( \log_2 8 = 3 \) bits per app; one that always predicts the category frequencies scores the entropy, 2.865 bits. Any real classifier must get below that, or its features are adding nothing:

```python
codes = apps["category"].map({c: i for i, c in enumerate(cat_p.index)}).to_numpy()
codes[:5]                                         # array([3, 3, 1, 0, 3])   category numbers 0..7
freq = cat_p.to_numpy()
np.mean(np.log2(1 / np.full(8, 1 / 8)[codes]))    # np.float64(3.0)   uniform guess
np.mean(np.log2(1 / freq[codes]))                 # np.float64(2.864538795702962)   = the column's entropy
```

The dict comprehension `{c: i for i, c in enumerate(cat_p.index)}` numbers the categories in `cat_p`'s order (Games = 0, Tools = 1, ...), like `cat_p.index.withIndex().associate { (i, c) -> c to i }` in Kotlin; `.map` then replaces each app's category by its number. `freq[codes]` picks, for each app, the probability the frequency model gives its true category.

## Softmax + cross-entropy, worked by hand

**What it is.** **Softmax** turns a vector of arbitrary real scores into a probability distribution. **Why it exists:** a network's last layer (Day 12) outputs one raw score per class, called **logits** \( \mathbf{z} \): any real numbers, positive or negative, not summing to anything in particular. Cross-entropy needs probabilities, so softmax sits between them.

**Intuition first.** Softmax is a "soft" version of picking the maximum. It exaggerates differences (through \( e^z \)) and then shares out 100% of the probability in proportion, so the largest logit gets the largest share, but every class keeps some.

\[
q_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}
\]

- \( z_k \) is the logit of class \( k \).
- \( e^{z_k} \) makes every score positive, and keeps the order: a bigger logit gives a bigger number.
- Dividing by the sum \( \sum_j e^{z_j} \) over all classes makes the results add to 1.

**Worked example.** Three classes (cat, dog, bird), logits \( \mathbf{z} = [2.0, 1.0, 0.1] \), true class cat, so \( \mathbf{y} = [1, 0, 0] \).

1. Exponentiate: \( e^{2.0} = 7.389 \), \( e^{1.0} = 2.718 \), \( e^{0.1} = 1.105 \). Sum: \( 11.213 \).
2. Normalize: \( \mathbf{q} = [7.389, 2.718, 1.105] / 11.213 = [0.659, 0.242, 0.099] \).
3. Loss: \( L = -\ln q_{\text{cat}} = -\ln 0.659 = 0.417 \) nats.
4. Gradient with respect to the logits (derived next): \( \mathbf{q} - \mathbf{y} = [0.659 - 1,\ 0.242 - 0,\ 0.099 - 0] = [-0.341,\ 0.242,\ 0.099] \).

<figure class="diagram" markdown>
![Animation: three logit boxes 2.0, 1.0 and 0.1 for cat, dog and bird; arrows lead to exp values 7.389, 2.718 and 1.105 with sum 11.213, then to orange probability bars 0.659, 0.242 and 0.099 that add up to 1, then the cat probability goes to a yellow box showing minus ln 0.659 = 0.417; a red panel shows the gradient q minus y: cat 0.659 minus 1 = -0.341, dog +0.242, bird +0.099, and red arrows on the logits show a gradient-descent step pushing the cat logit up and the others down](../../assets/img/phase-1/day54-softmax-ce.svg)
<figcaption>Animated: logits → exp → normalize → surprise at the true class; the gradient q − y says "raise the true class, lower the rest, each in proportion to its probability".</figcaption>
</figure>

### Deriving the gradient q − y

Write the loss with the softmax plugged in, and use \( \ln \frac{a}{b} = \ln a - \ln b \) and \( \ln e^{z} = z \):

\[
L = -\ln \frac{e^{z_c}}{\sum_j e^{z_j}} = -z_c + \ln \sum_j e^{z_j}
\]

where \( c \) is the true class. Now take the partial derivative (Day 32) with respect to one logit \( z_k \), one term at a time:

- **First term**, \( -z_c \): its derivative with respect to \( z_k \) is \( -1 \) if \( k = c \) and \( 0 \) otherwise. That is exactly \( -y_k \), the one-hot vector.
- **Second term**, \( \ln u \) with \( u = \sum_j e^{z_j} \): chain rule (Day 31). The outer derivative is \( \frac{1}{u} \) (Day 30); the inner one, \( \frac{\partial u}{\partial z_k} \), is \( e^{z_k} \), since only one term of the sum contains \( z_k \). Together: \( \frac{e^{z_k}}{\sum_j e^{z_j}} = q_k \).

Adding them:

\[
\frac{\partial L}{\partial z_k} = q_k - y_k
\qquad\text{or, as a vector,}\qquad
\nabla_{\mathbf{z}} L = \mathbf{q} - \mathbf{y}
\]

That might be the most important formula in deep learning, and it is short: **prediction minus truth**. It is the same shape as the linear-regression gradient from Day 33 (prediction minus target, times the input). Each wrong class is pushed down in proportion to how much probability it stole; the true class is pushed up by how much it was missing. When the model is perfectly confident and right, \( \mathbf{q} = \mathbf{y} \) and the gradient is zero.

Check it all in code, with Day 29's `central_diff` applied to one logit at a time:

```python
def softmax(z):
    e = np.exp(z - z.max())             # subtracting the max changes nothing (see below) but avoids overflow
    return e / e.sum()

def softmax_ce(z, c):
    """Cross-entropy loss (nats) of logits z when the true class is c."""
    return -np.log(softmax(z)[c])

def central_diff(f, x, h=1e-5):         # Day 29
    return (f(x + h) - f(x - h)) / (2 * h)

z = np.array([2.0, 1.0, 0.1])
y = np.array([1.0, 0.0, 0.0])
q = softmax(z)
q                                       # array([0.65900114, 0.24243297, 0.09856589])
softmax_ce(z, 0)                        # np.float64(0.4170300162778335)
analytic = q - y
analytic                                # array([-0.34099886,  0.24243297,  0.09856589])

numeric = np.zeros(3)
for k in range(3):                      # nudge one logit at a time
    def loss_of_zk(t, k=k):             # k=k: freeze this loop's k (explained below)
        z2 = z.copy()                   # copy, so the real z is never modified
        z2[k] = t                       # replace logit k with the trial value t
        return softmax_ce(z2, 0)
    numeric[k] = central_diff(loss_of_zk, z[k])
numeric                                 # array([-0.34099886,  0.24243297,  0.09856589])
np.abs(numeric - analytic).max() < 1e-8 # np.True_   the formula checks out
```

`central_diff` wants a function of **one** number, but the loss depends on three logits. So for each \( k \), `loss_of_zk(t)` is "the loss when logit \( k \) is set to \( t \) and the others stay put", and `central_diff` nudges \( t \) around `z[k]`.

The `k=k` default argument is a Python closure trap worth knowing. A nested function looks up an outer variable like `k` **when it is called**, not when it is defined (Kotlin lambdas capture the variable too, but a Kotlin `for` loop gives each iteration a fresh `val`, so the trap doesn't arise there). Here each function is called immediately, so it would work either way; but if you stored the three functions and called them later, all three would see the final `k = 2`. A default value is evaluated once, at definition time, so `k=k` freezes the current value. It is the standard idiom, and you'll see it in other people's code.

### Softmax without overflow

`np.exp(1000)` is larger than the biggest float64 (about \( 1.8 \times 10^{308} \)), so a naive softmax of large logits returns `nan`. The fix uses a property worth seeing once: adding the same number \( c \) to every logit changes nothing, because \( e^{z_k + c} = e^{z_k} e^{c} \) and the \( e^{c} \) cancels between top and bottom. So subtract the largest logit first; then the biggest exponent is \( e^0 = 1 \) and nothing overflows. That is why `softmax` above starts with `z - z.max()`.

```python
big = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):   # silence the overflow warnings for this demo
    naive = np.exp(big) / np.exp(big).sum()
naive                                   # array([nan, nan, nan])   inf / inf
softmax(big)                            # array([0.09003057, 0.24472847, 0.66524096])
softmax(big - 1000)                     # array([0.09003057, 0.24472847, 0.66524096])   same: shifts cancel
```

For the loss, libraries go one step further and never form \( q \) at all: \( \ln q_k = z_k - \ln \sum_j e^{z_j} \), computed as \( (z_k - m) - \ln \sum_j e^{z_j - m} \) with \( m = \max_j z_j \). This **log-sum-exp** trick gives a finite loss even when a probability would underflow to exactly 0 (become smaller than the smallest float, about \( 5 \times 10^{-324} \)):

```python
def log_softmax(z):
    m = z.max()
    return (z - m) - np.log(np.exp(z - m).sum())

z_far = np.array([0.0, 800.0])          # the model is wildly confident in the WRONG class (class 1)
with np.errstate(divide="ignore"):
    bad = -np.log(softmax(z_far)[0])    # q_0 = e^(-800) / ... underflows to exactly 0
bad                                     # np.float64(inf)   the naive loss is useless
-log_softmax(z_far)[0]                  # np.float64(800.0)  the correct, finite loss
```

### Temperature: sharpening or flattening softmax

Dividing the logits by a **temperature** \( T > 0 \) before softmax, \( \text{softmax}(\mathbf{z}/T) \), controls how peaked the result is. It is the "temperature" setting of text generators and a knob in knowledge distillation:

```python
for T in [0.5, 1, 5]:
    qT = softmax(z / T)
    print(f"T = {T}: q = {qT.round(3)}, entropy = {entropy(qT):.3f} bits")
# T = 0.5: q = [0.864 0.117 0.019], entropy = 0.655 bits
# T = 1: q = [0.659 0.242 0.099], entropy = 1.222 bits
# T = 5: q = [0.4   0.327 0.273], entropy = 1.568 bits
```

(`{entropy(qT):.3f}` formats a float with 3 decimals.) A small \( T \) multiplies the gaps between logits, so the top class takes almost everything and the entropy falls; a large \( T \) shrinks the gaps, and the distribution flattens towards uniform, whose entropy is \( \log_2 3 = 1.585 \) bits.

### Training with q − y

Gradient descent (Day 51) on the logits of that one example, step size 1: every step moves \( \mathbf{z} \) against \( \mathbf{q} - \mathbf{y} \), and the loss falls.

```python
zz = z.copy()
for step in range(4):
    print(f"step {step}: q = {softmax(zz).round(3)}, loss = {softmax_ce(zz, 0):.4f}")
    zz -= 1.0 * (softmax(zz) - y)
# step 0: q = [0.659 0.242 0.099], loss = 0.4170
# step 1: q = [0.768 0.158 0.074], loss = 0.2636
# step 2: q = [0.826 0.115 0.059], loss = 0.1907
# step 3: q = [0.862 0.09  0.048], loss = 0.1488
```

A real classifier doesn't learn logits directly; it learns weights that *produce* them. **Softmax regression** (multi-class logistic regression) computes \( Z = XW + \mathbf{b} \), one row of logits per example, exactly the layer from Day 12. Here \( X \) is \( (n, d) \) (\( n \) examples, \( d \) features), \( W \) is \( (d, K) \) for \( K \) classes, and \( Z \) is \( (n, K) \).

**The gradient for \( W \), step by step.** The loss is the average of the \( n \) per-example losses, so each example's logit row gets gradient \( \mathbf{q}_i - \mathbf{y}_i \) divided by \( n \). Stacking those rows gives a matrix \( G = \frac1n (Q - Y) \), where \( Q \) holds every example's probabilities as rows and \( Y \) the one-hot labels. Then the chain rule goes back through \( Z = XW \): entry \( Z_{ik} = \sum_j X_{ij} W_{jk} \), so \( W_{jk} \) affects every example's logit \( k \) with weight \( X_{ij} \), and

\[
\frac{\partial L}{\partial W_{jk}} = \sum_i X_{ij}\, G_{ik}
\quad\Longrightarrow\quad
\nabla_W L = X^\top G = \frac1n X^\top (Q - Y), \qquad \nabla_{\mathbf{b}} L = \sum_i G_{i,:}
\]

The shapes confirm it (Day 33's rule: the gradient has the parameter's shape): \( X^\top \) is \( (d, n) \) and \( G \) is \( (n, K) \), so \( X^\top G \) is \( (d, K) \), the shape of \( W \). The bias \( \mathbf{b} \) is added to every row, so its gradient adds up \( G \)'s rows.

Before training on 300 points, trace every piece of the code on a batch of 3 examples, 2 features, 3 classes:

```python
def softmax_rows(Z):
    """Softmax of each row of Z separately."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

np.repeat([0, 1, 2], 2)                 # array([0, 0, 1, 1, 2, 2])   each element repeated twice
labels_s = np.array([0, 2, 1])          # true classes of the 3 examples
Y_s = np.eye(3)[labels_s]               # pick rows of the 3x3 identity: one-hot labels
Y_s
# array([[1., 0., 0.],
#        [0., 0., 1.],
#        [0., 1., 0.]])
X_s = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0]])        # (3, 2)
W_s = np.array([[0.1, -0.2, 0.3], [0.0, 0.4, -0.1]])         # (2, 3)
Z_s = X_s @ W_s                         # (3, 3): one row of logits per example
Z_s
# array([[ 0.1 ,  0.6 ,  0.1 ],
#        [ 0.05, -0.5 ,  0.25],
#        [ 0.2 , -0.4 ,  0.6 ]])
Z_s.max(axis=1), Z_s.max(axis=1, keepdims=True)
# (array([0.6 , 0.25, 0.6 ]),
#  array([[0.6 ],
#         [0.25],
#         [0.6 ]]))
Q_s = softmax_rows(Z_s)
Q_s.round(3)
# array([[0.274, 0.452, 0.274],
#        [0.357, 0.206, 0.436],
#        [0.329, 0.18 , 0.491]])
Q_s[np.arange(3), labels_s].round(3)    # array([0.274, 0.436, 0.18 ])   each row's TRUE-class probability
G_s = (Q_s - Y_s) / 3
X_s.T @ G_s                             # (2, 3) @ (3, 3) -> (2, 3): the shape of W
# array([[ 0.03683408, -0.36135504,  0.32452096],
#        [-0.60307196,  0.2325169 ,  0.37055505]])
```

What each piece does:

- `np.eye(3)` is the 3×3 identity matrix (Day 11). Its row \( c \) is the one-hot vector for class \( c \), so indexing it with the label array picks one row per example: one-hot encoding in one line.
- `axis=1` reduces **along each row** (Day 8), giving one max per example. Plain `max(axis=1)` returns shape `(3,)`; `keepdims=True` keeps the reduced axis as size 1, shape `(3, 1)`, so broadcasting (Day 9) subtracts each row's own max from that row. With shape `(3,)`, NumPy would line it up with the *columns* instead and subtract the wrong numbers.
- `Q_s[np.arange(3), labels_s]` pairs up two index arrays: row 0 with column 0, row 1 with column 2, row 2 with column 1. It picks each example's true-class probability without building one-hot vectors at all. Its `-np.log(...)` averaged is the loss.
- `X_s.T @ G_s` is \( X^\top G \). Check one entry against a numerical derivative:

```python
def loss_W(W):
    Q = softmax_rows(X_s @ W)
    return -np.mean(np.log(Q[np.arange(3), labels_s]))

def loss_of_w01(t):                     # the loss as a function of the single entry W[0, 1]
    W2 = W_s.copy()
    W2[0, 1] = t
    return loss_W(W2)

central_diff(loss_of_w01, W_s[0, 1])    # np.float64(-0.3613550358050332)
(X_s.T @ G_s)[0, 1]                     # np.float64(-0.36135503580970996)   agrees to 11 digits
```

Now the real thing. Three clouds of 2-D points, one per class:

```python
rng = np.random.default_rng(54)
centers = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
labels = np.repeat([0, 1, 2], 100)                       # 100 points per class: [0, 0, ..., 2, 2]
X = centers[labels] + rng.normal(size=(300, 2))          # each point: its class centre plus noise, (300, 2)
Y = np.eye(3)[labels]                                    # one-hot, (300, 3)

W = np.zeros((2, 3))
b = np.zeros(3)
losses = []
for step in range(200):
    Q = softmax_rows(X @ W + b)                          # (300, 3) probabilities
    losses.append(-np.mean(np.log(Q[np.arange(300), labels])))
    G = (Q - Y) / len(X)                                 # dL/dZ, one row per example
    W -= 0.5 * (X.T @ G)                                 # (2, 300) @ (300, 3) -> (2, 3)
    b -= 0.5 * G.sum(axis=0)                             # add up the rows: shape (3,)
losses[0], np.log(3)                    # (np.float64(1.0986122886681096), np.float64(1.0986122886681098))
round(losses[-1], 4)                    # np.float64(0.2044)
(Q.argmax(axis=1) == labels).mean()     # np.float64(0.9166666666666666)   92% of training points right
```

Notice `losses[0]`: with all weights zero, every logit is 0, so the model says \( \frac13 \) for every class and the loss is \( \ln 3 = 1.0986 \) nats, the entropy of a uniform guess over 3 classes. **This is a sanity check worth memorizing**: a freshly initialized \( K \)-class classifier should start near \( \ln K \) (2.303 for 10 classes, 6.908 for the 1000 classes of ImageNet). If your first loss is far above that, something is wrong before training even starts: typically initial weights that are too large (so the untrained model is already confidently wrong), or labels that don't line up with the rows of the logits (shuffled one but not the other).

### Perplexity, sparse labels and label smoothing

Three things you'll meet in the next months, each one line of NumPy on top of what you have.

**Perplexity** is \( e^{\text{loss}} \) (with the loss in nats): the number of equally likely classes that would be as confusing as the model. It is the standard way to report a language model's cross-entropy.

**Sparse labels.** Storing labels as class numbers (`labels`) instead of one-hot rows (`Y`) and picking with `Q[np.arange(n), labels]` gives the same loss with far less memory. PyTorch's `nn.CrossEntropyLoss` takes class numbers like this by default.

**Label smoothing** replaces the one-hot target by a softened one, \( (1 - \varepsilon)\mathbf{y} + \varepsilon / K \) (\( \varepsilon \), "epsilon", is a small number like 0.1). The model is then never pushed towards 100% certainty. Because the smoothed target has entropy above 0, the loss is no longer equal to the KL divergence: it can't go below \( H(\mathbf{y}_{\text{smooth}}) \).

```python
np.exp(losses[-1]), np.exp(np.log(3))               # (np.float64(1.2268282392746266), np.float64(3.0000000000000004))
loss_sparse = -np.mean(np.log(Q[np.arange(300), labels]))
loss_onehot = -np.mean((Y * np.log(Q)).sum(axis=1))
np.isclose(loss_sparse, loss_onehot)                # np.True_   same loss, no one-hot matrix needed
y_smooth = (1 - 0.1) * y + 0.1 / 3                  # smooth the cat label with epsilon = 0.1
y_smooth                                            # array([0.93333333, 0.03333333, 0.03333333])
entropy(y_smooth, base=np.e)                        # 0.291139838831965   the new floor for the loss
```

The trained classifier's perplexity is 1.23: it is as unsure as if it were choosing between 1.23 equally likely classes, against 3 for the untrained uniform guess (the `...0004` is a floating-point rounding of 3). In `loss_onehot`, `Y * np.log(Q)` multiplies element by element, so only the true-class column survives in each row; `sum(axis=1)` collects it.

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
qq = np.linspace(0.01, 1, 300)
ax1.plot(qq, -np.log(qq), color="#3949ab", lw=2.5)
for qv in [0.1, 0.7]:
    ax1.plot(qv, -np.log(qv), "o", color="#e53935", ms=7)
    ax1.annotate(f"q = {qv}: loss {-np.log(qv):.3f}", (qv, -np.log(qv)),
                 xytext=(qv + 0.05, -np.log(qv) + 0.4), ha="left" if qv < 0.5 else "center", color="#e53935")
ax1.set_xlabel("probability the model gave the TRUE class")
ax1.set_ylabel("loss = −ln q (nats)")
ax1.set_title("Cross-entropy for one example")
ax1.grid(alpha=0.3)

ax2.plot(losses, color="#00897b", lw=2.5, label="training loss")
ax2.axhline(np.log(3), color="#ef6c00", ls="--", label="ln 3: uniform guess")
ax2.set_xlabel("gradient descent step")
ax2.set_ylabel("average loss (nats)")
ax2.set_title("Softmax regression on 3 classes")
ax2.legend(frameon=False)
ax2.grid(alpha=0.3)
fig.savefig("day54-log-loss.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Left: the curve minus ln q falls steeply from about 4.6 at q = 0.01 to 0 at q = 1, with red dots marking q = 0.1 with loss 2.303 and q = 0.7 with loss 0.357. Right: a teal training-loss curve starting at the orange dashed line ln 3 = 1.0986 and falling quickly, then levelling off near 0.20 by step 200](../../assets/img/phase-1/day54-log-loss.svg)
<figcaption>Left: confident wrong answers cost the most, and the cost has no ceiling. Right: training starts at ln 3 (a uniform guess) and the loss falls as q moves towards the labels.</figcaption>
</figure>

The left panel explains why cross-entropy trains so well. Near \( q = 0 \) the curve is steep, so a model that is confidently wrong gets a large loss *and* a large gradient: it is pushed hardest exactly where it is worst. (Accuracy, by contrast, has no gradient at all, as Day 50 showed.) And softmax + cross-entropy is convex in \( W \) and \( \mathbf{b} \) (Day 53) for a single linear layer like this one, so softmax regression has no bad local minima.

!!! warning "Common mistake: applying softmax twice"
    Many frameworks' cross-entropy functions take **logits** and apply (log-)softmax themselves: PyTorch's `nn.CrossEntropyLoss` is `LogSoftmax` followed by `NLLLoss`. If your model already ends in a softmax layer and you feed its probabilities into that loss, softmax runs twice:

    ```python
    probs = softmax(z)                      # the model already applied softmax
    softmax(probs)                          # array([0.448377  , 0.29561745, 0.25600555])   nearly uniform
    softmax_ce(probs, 0), softmax_ce(z, 0)  # (np.float64(0.8021208912128679), np.float64(0.4170300162778335))
    ```

    No error is raised. Probabilities all lie between 0 and 1, so their differences are tiny and the second softmax is nearly uniform. Recognise it by a loss that falls slowly and **never gets near 0**, even on training data the model classifies perfectly (with 3 classes the doubled loss can't go below \( -\ln \frac{e}{e + 2} = 0.551 \)). Fix: feed logits to losses that expect logits, and read the docs of the one you use.

!!! android "Android bridge"
    An image classifier you run with LiteRT (TensorFlow Lite) usually has one float output tensor of shape `[1, numClasses]`. Whether those numbers are logits or probabilities depends on the model: open the `.tflite` file in [Netron](https://netron.app/) and look for a `SOFTMAX` op at the end. For picking the top label you don't need softmax at all (it keeps the order, so `argmax` of the logits is the same class). You do need it, done stably, to show a confidence or to apply a threshold, and the **entropy** of the result is a cheap "how unsure is the model?" signal: near 0 means one clear winner, near \( \log_2 K \) means it is guessing.

    ```kotlin
    import kotlin.math.exp
    import kotlin.math.ln

    fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()                          // non-null max() needs Kotlin 1.7+
        val e = FloatArray(logits.size) { exp(logits[it] - max) }   // subtract the max: no overflow
        val sum = e.sum()
        return FloatArray(e.size) { e[it] / sum }
    }

    fun entropyBits(probs: FloatArray): Float =
        probs.filter { it > 0f }.sumOf { -it * ln(it.toDouble()) / ln(2.0) }.toFloat()

    // val probs = softmax(outputBuffer)            // only if the model outputs logits
    // if (entropyBits(probs) > 1.5f) showNotSureUi() else showLabel(probs)
    ```

    **Choosing the threshold.** The maximum possible is \( \log_2 K \): 3.32 bits for 10 classes, 2 bits for 4. Run the model on a few hundred labelled images from real users, compute the entropy of each prediction, and pick the threshold where the "not sure" screen catches most of the wrong answers while hiding few right ones. It is a product decision, not a constant from the maths. And you have shipped entropy coding for years: an APK is a ZIP file, most of its entries are DEFLATE-compressed, and DEFLATE ends with a Huffman code, the same "short codewords for common symbols" idea from earlier today.

## Going further

These are real and useful, but not needed yet:

- **Joint and conditional entropy, mutual information.** How much knowing one variable tells you about another. "Information gain", the split criterion of decision trees, is mutual information; Month 3 (classic ML) uses it. [Wikipedia: Mutual information](https://en.wikipedia.org/wiki/Mutual_information).
- **Differential entropy** for continuous distributions, and why MSE is the negative log-likelihood of Gaussian noise (Day 47 showed the second part). [Wikipedia: Differential entropy](https://en.wikipedia.org/wiki/Differential_entropy).
- **Forward vs reverse KL**: which order you minimize changes what a model learns (mode-covering vs mode-seeking), central to variational autoencoders and distillation. [Wikipedia: Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
- **Jensen–Shannon divergence**: a symmetric, always-finite cousin of KL. [Wikipedia: Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence).
- **Perplexity in depth**, as the headline metric of language models. [Wikipedia: Perplexity](https://en.wikipedia.org/wiki/Perplexity).
- **Label smoothing in practice**: PyTorch's `CrossEntropyLoss` has a `label_smoothing` argument. [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
- **Focal loss**: a reweighted cross-entropy for very imbalanced classes (Lin et al., 2017, "Focal Loss for Dense Object Detection"). [arXiv:1708.02002](https://arxiv.org/abs/1708.02002).

---

## Exercises

**Warm-ups**

1. By hand, find the surprise in bits of rolling two sixes in a row with a fair die. Check that it equals twice the surprise of one six, and confirm in code.
2. A loaded die lands 6 half the time and each other face 10% of the time. Compute its entropy by hand and in code, in bits and in nats. Is it more or less than a fair die, and why?

**Core**

3. For \( p = [0.7, 0.2, 0.1] \) and \( q = [0.5, 0.3, 0.2] \), compute \( H(p) \), \( H(p, q) \), \( D_{\mathrm{KL}}(p \,\|\, q) \) and \( D_{\mathrm{KL}}(q \,\|\, p) \) in bits. Verify that \( H(p, q) - H(p) \) equals the \( \sum p \log \frac{p}{q} \) form.
4. Logits \( \mathbf{z} = [1.0, 3.0, 0.5] \), true class index 2 (the third). By hand (a calculator for \( e^x \) is fine), compute the softmax, the loss in nats and the gradient \( \mathbf{q} - \mathbf{y} \). Check the gradient with `central_diff`. Which logit does a gradient-descent step raise?
5. Play Store: compute the entropy in bits of the `has_ads` column. Then score two "models" that predict `has_ads` with no features by their average cross-entropy over all apps: a 50/50 guess, and the column's own frequencies. Which is better, by how many bits per app, and what is that gap called?

**Predict the output**

6. Write down your answers *before* running anything:
    - (a) `entropy([0.25, 0.25, 0.25, 0.25])` and `entropy([0.5, 0.5, 0.0, 0.0])`
    - (b) `softmax(np.array([2.0, 1.0, 0.1]) + 100)`
    - (c) the first loss, in nats, of a 10-class softmax classifier whose weights and biases start at zero, on any data
    - (d) `cross_entropy([1, 0], [0.5, 0.5])` (in bits)

**Find and fix the bug**

7. A teammate wrote this to measure how unpredictable a column is. It returns `-18.36452797660028` for a column with three values. Find the bug, fix it, and explain why the wrong answer is negative.

    ```python
    def column_entropy(col):
        counts = col.value_counts()
        p = counts.to_numpy(dtype=float)
        return float(np.sum(p * np.log2(1 / p)))
    ```

**Challenge**

8. **Label smoothing, derived.** For a smoothed target \( \mathbf{t} = (1 - \varepsilon)\mathbf{y} + \varepsilon/K \) and \( L = -\sum_k t_k \ln q_k \) with \( \mathbf{q} = \text{softmax}(\mathbf{z}) \), show that \( \nabla_{\mathbf{z}} L = \mathbf{q} - \mathbf{t} \) (hint: \( \sum_k t_k = 1 \)). Check it with `central_diff` for \( \mathbf{z} = [2.0, 1.0, 0.1] \), cat, \( \varepsilon = 0.1 \). Then run 2000 gradient-descent steps (step size 1) from those logits, once with the one-hot target and once with the smoothed one, and compare the final logits, probabilities and losses. What does smoothing stop from happening?

??? success "Solutions"
    **1.** Two independent sixes have probability \( \frac16 \cdot \frac16 = \frac1{36} \), so the surprise is \( \log_2 36 = \log_2 6 + \log_2 6 = 2.585 + 2.585 = 5.170 \) bits.

    ```python
    import numpy as np

    def surprise_bits(p):
        return np.log2(1 / p)

    surprise_bits(1 / 36), 2 * surprise_bits(1 / 6)
    # (np.float64(5.169925001442312), np.float64(5.169925001442312))
    ```
    It works because the log turns the product of the two probabilities into a sum of the two surprises: that was wish 3.

    **2.** By hand: \( H = 0.5 \log_2 2 + 5 \cdot 0.1 \log_2 10 = 0.5 + 0.5 \cdot 3.3219 = 2.161 \) bits.

    ```python
    import numpy as np

    def entropy(p, base=2):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return float(np.sum(p * np.log(1 / p)) / np.log(base))

    loaded = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
    round(entropy(loaded), 4), round(entropy(loaded, base=np.e), 4)   # (2.161, 1.4979)
    round(entropy(np.full(6, 1 / 6)), 4)                              # 2.585
    ```
    Less than the fair die's 2.585 bits: a 6 is now easy to guess, so on average you are less surprised. The uniform distribution is always the maximum, \( \log_2 6 \).

    **3.**

    ```python
    import numpy as np

    def entropy(p, base=2):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return float(np.sum(p * np.log(1 / p)) / np.log(base))

    def cross_entropy(p, q, base=2):
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        m = p > 0
        return float(np.sum(p[m] * np.log(1 / q[m])) / np.log(base))

    p3, q3 = np.array([0.7, 0.2, 0.1]), np.array([0.5, 0.3, 0.2])
    round(entropy(p3), 4)                              # 1.1568
    round(cross_entropy(p3, q3), 4)                    # 1.2796
    round(cross_entropy(p3, q3) - entropy(p3), 4)      # 0.1228   KL(p || q)
    np.sum(p3 * np.log2(p3 / q3)).round(4)             # np.float64(0.1228)   same, from the second formula
    round(cross_entropy(q3, p3) - entropy(q3), 4)      # 0.1328   KL(q || p): not the same
    ```
    The two forms agree because \( \log\frac1q - \log\frac1p = \log\frac pq \) term by term. The two directions differ because each one averages with a different distribution's weights: \( p \) weights the first class (where \( q \) is too low) at 0.7, while \( q \) gives it only 0.5.

    **4.** By hand: \( e^{\mathbf{z}} = [2.718, 20.086, 1.649] \), sum \( 24.453 \), so \( \mathbf{q} = [0.111, 0.821, 0.067] \). Loss \( -\ln 0.067 = 2.697 \) nats (with the unrounded 0.0674). Gradient \( \mathbf{q} - [0, 0, 1] = [0.111, 0.821, -0.933] \).

    ```python
    import numpy as np

    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def softmax_ce(z, c):
        return -np.log(softmax(z)[c])

    def central_diff(f, x, h=1e-5):
        return (f(x + h) - f(x - h)) / (2 * h)

    z4 = np.array([1.0, 3.0, 0.5])
    q4 = softmax(z4)
    q4.round(3)                        # array([0.111, 0.821, 0.067])
    softmax_ce(z4, 2).round(4)         # np.float64(2.6967)
    g4 = q4 - np.array([0.0, 0.0, 1.0])
    g4.round(3)                        # array([ 0.111,  0.821, -0.933])

    num4 = np.zeros(3)
    for k in range(3):
        def loss_of_zk(t, k=k):
            z2 = z4.copy()
            z2[k] = t
            return softmax_ce(z2, 2)
        num4[k] = central_diff(loss_of_zk, z4[k])
    np.abs(num4 - g4).max() < 1e-8     # np.True_
    ```
    The model put only 6.7% on the true class, so the loss is large. The gradient's third entry is negative, so a step \( \mathbf{z} \leftarrow \mathbf{z} - \eta(\mathbf{q} - \mathbf{y}) \) **raises** the true class's logit and lowers the other two, most of all the logit 3.0 that took 82% of the probability.

    **5.**

    ```python
    import numpy as np
    import pandas as pd

    def entropy(p, base=2):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return float(np.sum(p * np.log(1 / p)) / np.log(base))

    BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
    apps5 = pd.read_csv(BASE + "play_store_apps.csv").drop_duplicates()
    ads = (apps5["has_ads"] == "Yes").to_numpy().astype(int)   # 1 = has ads, 0 = no ads
    p_ads = np.array([1 - ads.mean(), ads.mean()])             # [P(no), P(yes)]
    p_ads.round(3)                                             # array([0.373, 0.627])
    round(entropy(p_ads), 4)                                   # 0.9529
    np.mean(np.log2(1 / np.array([0.5, 0.5])[ads]))            # np.float64(1.0)   50/50 guess
    np.mean(np.log2(1 / p_ads[ads])).round(4)                  # np.float64(0.9529)   frequencies = the entropy
    ```
    `p_ads[ads]` looks up, for each app, the probability the frequency model gives its true value (index 0 for "no", 1 for "yes"). The 50/50 guess costs 1 bit per app; predicting the frequencies costs the entropy, 0.9529 bits. The difference, 0.0471 bits per app, is \( D_{\mathrm{KL}}(p_{\text{data}} \,\|\, \text{uniform}) \): the price of the wrong belief. It is small because 63/37 is not far from 50/50.

    **6.** The answers:

    - (a) `2.0` and `1.0`. Four equally likely outcomes give \( \log_2 4 = 2 \) bits; the zero-probability outcomes are dropped, leaving a fair coin.
    - (b) The same as `softmax(np.array([2.0, 1.0, 0.1]))`, `array([0.65900114, 0.24243297, 0.09856589])`: adding a constant to every logit cancels out.
    - (c) \( \ln 10 \), shown as `np.float64(2.3025850929940455)`: zero weights give equal logits, so a uniform \( \frac1{10} \), whose surprise is \( \ln 10 \) whatever the label.
    - (d) `1.0`: the true outcome got probability 0.5, which is 1 bit of surprise.

    ```python
    import numpy as np

    def entropy(p, base=2):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return float(np.sum(p * np.log(1 / p)) / np.log(base))

    def cross_entropy(p, q, base=2):
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        m = p > 0
        return float(np.sum(p[m] * np.log(1 / q[m])) / np.log(base))

    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    entropy([0.25, 0.25, 0.25, 0.25]), entropy([0.5, 0.5, 0.0, 0.0])   # (2.0, 1.0)
    softmax(np.array([2.0, 1.0, 0.1]) + 100)    # array([0.65900114, 0.24243297, 0.09856589])
    -np.log(softmax(np.zeros(10))[3])           # np.float64(2.3025850929940455)   any true class gives the same
    cross_entropy([1, 0], [0.5, 0.5])           # 1.0
    ```
    If you got (b) wrong, reread "Softmax without overflow": shift invariance is exactly why subtracting the max is safe.

    **7.** `value_counts()` returns **counts**, not probabilities, so `p` holds numbers like 5, 3 and 2. Every count above 1 has \( \log_2 \frac1p < 0 \), so every term is negative.

    ```python
    import numpy as np
    import pandas as pd

    col = pd.Series(["Games"] * 5 + ["Tools"] * 3 + ["Finance"] * 2)

    def column_entropy(col):                     # the buggy version
        counts = col.value_counts()
        p = counts.to_numpy(dtype=float)
        return float(np.sum(p * np.log2(1 / p)))

    def column_entropy_fixed(col):
        p = col.value_counts(normalize=True).to_numpy(dtype=float)   # shares that sum to 1
        return float(np.sum(p * np.log2(1 / p)))

    column_entropy(col)          # -18.36452797660028
    column_entropy_fixed(col)    # 1.4854752972273344
    ```
    Entropy is defined for a distribution, so the inputs must sum to 1. A negative value (or one above \( \log_2 K \)) is the giveaway, because a real entropy always lies between 0 and \( \log_2 K \). `normalize=True` divides each count by the total, here 10.

    **8.** Write \( \ln q_k = z_k - \ln \sum_j e^{z_j} \). Then \( L = -\sum_k t_k z_k + \bigl(\sum_k t_k\bigr) \ln \sum_j e^{z_j} \), and since \( \sum_k t_k = 1 \), the derivative with respect to \( z_k \) is \( -t_k + q_k \), the same steps as for one-hot labels.

    ```python
    import numpy as np

    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def central_diff(f, x, h=1e-5):
        return (f(x + h) - f(x - h)) / (2 * h)

    z8 = np.array([2.0, 1.0, 0.1])
    onehot = np.array([1.0, 0.0, 0.0])
    t = (1 - 0.1) * onehot + 0.1 / 3              # smoothed target, epsilon = 0.1, K = 3

    def smooth_ce(z):
        return -np.sum(t * np.log(softmax(z)))

    num8 = np.zeros(3)
    for k in range(3):
        def loss_of_zk(v, k=k):
            z2 = z8.copy()
            z2[k] = v
            return smooth_ce(z2)
        num8[k] = central_diff(loss_of_zk, z8[k])
    (softmax(z8) - t).round(6), num8.round(6)
    # (array([-0.274332,  0.2091  ,  0.065233]), array([-0.274332,  0.2091  ,  0.065233]))

    for target, name in [(onehot, "one-hot "), (t, "smoothed")]:
        zz = z8.copy()
        for _ in range(2000):
            zz -= 1.0 * (softmax(zz) - target)
        loss = -np.sum(target * np.log(softmax(zz)))
        print(name, zz.round(2), softmax(zz).round(4), round(loss, 4))
    # one-hot  [ 6.83 -1.83 -1.91] [9.997e-01 2.000e-04 2.000e-04] 0.0003
    # smoothed [ 3.25 -0.08 -0.08] [0.9333 0.0333 0.0333] 0.2911
    ```
    With the one-hot target, the only way to reach loss 0 is \( q_{\text{cat}} = 1 \), which needs infinitely large logit gaps, so the logits keep growing for as long as you train (gap 8.7 after 2000 steps). With the smoothed target, the gradient \( \mathbf{q} - \mathbf{t} \) is exactly 0 once \( \mathbf{q} = \mathbf{t} \), so the logits settle at a finite gap of 3.33 and the loss stops at \( H(\mathbf{t}) = 0.2911 \) nats. Smoothing stops the model from becoming endlessly overconfident.

## Resources

- [Christopher Olah: Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/): the best picture-first explanation of entropy, cross-entropy and KL as code lengths.
- [3Blue1Brown: Solving Wordle using information theory](https://www.youtube.com/watch?v=v68zYyaEmEA): bits and entropy explained through choosing the most informative guess.
- [Khan Academy: Information theory](https://www.khanacademy.org/computing/computer-science/informationtheory): a gentle, historical series on what a bit is and how messages are compressed.
- [Dive into Deep Learning: Information Theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html): self-information, entropy, cross-entropy and KL with code, written for deep learning.
- [CS231n: Linear classification (the softmax classifier)](https://cs231n.github.io/linear-classify/): softmax, cross-entropy, the information-theory view and numeric stability, from Stanford's course notes.
- [PyTorch docs: `torch.nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): the loss you'll use in Phase 3; note that it expects raw logits.
- [Netron](https://netron.app/): open a `.tflite` model in the browser to see whether it ends with a `SOFTMAX` op.

## Checklist

- [ ] I can compute the surprise of an event and explain why the logarithm makes independent surprises add.
- [ ] I can compute entropy by hand and with `entropy`, in bits and in nats, and I know it is largest for a uniform distribution.
- [ ] I can explain entropy as the best average code length, using the four-tab code.
- [ ] I can compute cross-entropy and KL divergence, I know \( H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q) \), and I showed KL is not symmetric.
- [ ] I can explain why the classification loss is \( -\ln q_{\text{true}} \), and why minimizing it is maximum likelihood.
- [ ] I worked softmax + cross-entropy by hand, derived the gradient \( \mathbf{q} - \mathbf{y} \), checked it with `central_diff`, and trained a softmax regression with it.
- [ ] I know why softmax subtracts the max, and why a fresh \( K \)-class classifier should start near a loss of \( \ln K \).
- [ ] I can spot log of 0, bits vs nats, counts instead of probabilities, and softmax applied twice.

**Next:** [Days 55–56: Month 2 milestone and retrospective](day-55-month2-milestone.md)
