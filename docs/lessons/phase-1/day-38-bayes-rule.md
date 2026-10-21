# Day 38: Conditional Probability and Bayes' Rule

**Phase 1 · Week 6 · ~120 minutes**

## Objectives

- [ ] Read \( P(A \mid B) \) aloud correctly ("the probability of A **given** B") and compute it from a 2×2 count table.
- [ ] Explain why \( P(A \mid B) \) and \( P(B \mid A) \) are different numbers, with a real example.
- [ ] Test whether two events are independent, both by comparing conditional probabilities and with \( P(A \cap B) = P(A)\,P(B) \).
- [ ] Use the multiplication rule and the law of total probability, and draw them as a probability tree.
- [ ] Derive Bayes' rule from a count table, and name its parts: prior, likelihood, evidence, posterior.
- [ ] Work the medical-test and spam-filter base-rate examples by hand, with natural frequencies, and by simulation.
- [ ] Explain why a classifier's precision depends on how common the positive class is, and correct a model's output for a new base rate.

**Prerequisites.** [Day 36](day-36-probability-basics.md) (events, probability as long-run frequency, simulating with `rng`, the complement and multiplication rules) and [Day 37](day-37-distributions.md) (Bernoulli trials). The pandas you need is from [Day 16](day-16-pandas-selecting.md) (boolean masks) and [Day 18](day-18-pandas-groupby-merge.md) (`groupby`, `crosstab`).

---

## The big idea

Every probability so far was about the **whole** population: "what fraction of all apps have ads?" Real questions are usually about a **part** of it: "of the apps that *have ads*, what fraction are highly rated?" Asking a probability question about a part is called **conditioning**, and the part is the **condition**.

**Analogy.** You already do this in the Firebase Crashlytics console. "The crash-free rate is 99.2%" is about all sessions. Then you add a filter, *device = Pixel 6, Android 14*, and the dashboard recomputes everything **inside that filter**: the filtered sessions become the whole world, and the crash rate is "crashes in the filter ÷ sessions in the filter". That filtered rate is a conditional probability. A filter never adds data; it only changes what you divide by.

Bayes' rule is the tool for turning a conditional probability around. You often know one direction and need the other:

| You know | You want |
|---|---|
| How often the test is positive **given** the disease | How likely the disease is **given** a positive test |
| How often "free" appears **given** a message is spam | How likely a message is spam **given** it contains "free" |
| How often an image looks like this **given** it shows a cat | How likely it's a cat **given** the image |

The right column is what every classifier you will ever ship outputs. And the way people get it wrong, ignoring how common each class is (the **base rate**), is one of the most common reasoning mistakes there is, in medicine, in courtrooms and in ML dashboards.

## Conditional probability from a count table

Load the course dataset. We need only two facts about each app: does it show ads, and is its rating at least 4.5? Unrated apps can't answer the second question, so drop them. ([Download the CSV from the Datasets page](../../datasets.md) to work offline and pass the local path to `read_csv`. This lesson was tested with NumPy 2.5 and pandas 3.0.)

```python
import numpy as np
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")

apps = apps.drop_duplicates().dropna(subset=["rating"]).reset_index(drop=True)
apps = apps.assign(
    ads=apps["has_ads"] == "Yes",          # True / False per app
    high=apps["rating"] >= 4.5,            # True / False per app
)
print(len(apps))
print(pd.crosstab(apps["has_ads"], apps["high"], margins=True))
```

```text
462
high     False  True  All
has_ads                  
No         121    50  171
Yes        208    83  291
All        329   133  462
```

That is a **2×2 count table** (also called a **contingency table**) with its totals in the margins. Each of the 462 apps lands in exactly one of the four inner cells. Name the two events:

- \( A \) = "rating ≥ 4.5" (call it *high*),
- \( B \) = "has ads".

Three kinds of probability can be read off the table, and the difference between them is only **what you divide by**:

| Kind | Notation | Read it as | From the table |
|---|---|---|---|
| Marginal | \( P(A) \) | "probability of high" | \( 133 / 462 = 0.288 \) |
| Joint | \( P(A \cap B) \) | "probability of high **and** ads" | \( 83 / 462 = 0.180 \) |
| Conditional | \( P(A \mid B) \) | "probability of high **given** ads" | \( 83 / 291 = 0.285 \) |

The symbol \( \cap \) ("cap", or just "and") means both events happen. The vertical bar \( \mid \) means **given**: everything to the right of it is the filter. "Marginal" comes from the margins of the table, where the totals live.

Dividing the conditional count by 291 is the Crashlytics filter: only the "has ads" row is left, and it becomes the whole world. Write the same thing with probabilities instead of counts (divide top and bottom by 462):

\[
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
\]

Reading it symbol by symbol: the top is the fraction of **all** apps that are in both events, the bottom is the fraction of all apps that pass the filter. Their ratio is "of the ones that pass the filter, the fraction that are also in \( A \)". With numbers: \( \dfrac{83/462}{291/462} = \dfrac{0.1797}{0.6299} = 0.285 \), the same as \( 83/291 \). The formula only makes sense when \( P(B) > 0 \): you can't filter down to an empty set.

<figure class="diagram" markdown>
![Animation of the 2 by 2 count table of 462 rated apps with rows no ads and has ads and columns rating below 4.5 and rating at least 4.5: first the whole table with P of high rating equals 133 over 462 equals 0.288; then only the has-ads row is highlighted and the rest is faded, with P of high given ads equals 83 over 291 equals 0.285; then only the high-rating column is highlighted, with P of ads given high equals 83 over 133 equals 0.624; finally both results side by side](../../assets/img/phase-1/day38-condition-zoom.svg)
<figcaption>Animated: conditioning keeps one row (or one column) and makes it the whole world. The same 83 apps give 0.285 or 0.624 depending on what you divide by.</figcaption>
</figure>

In code, a condition is a boolean mask (Day 16), and "the fraction that are True" is the `.mean()` of a boolean column:

```python
ads, high = apps["ads"], apps["high"]

p_high = high.mean()
p_both = (high & ads).mean()
p_ads = ads.mean()
print(round(p_high, 3), round(p_both, 3), round(p_ads, 3))    # 0.288 0.18 0.63

print(round(p_both / p_ads, 3))         # 0.285   P(high | ads) by the formula
print(round(high[ads].mean(), 3))       # 0.285   the same, by filtering first
print(round(ads[high].mean(), 3))       # 0.624   P(ads | high): the other way round

print(apps.groupby("has_ads")["high"].mean().round(3))   # every row's conditional at once
# has_ads
# No     0.292
# Yes    0.285
# Name: high, dtype: float64
```

`pd.crosstab(..., normalize="index")` divides each row by its own total, which turns the whole count table into conditionals "given the row":

```python
print(pd.crosstab(apps["has_ads"], apps["high"], normalize="index").round(3))
```

```text
high     False  True 
has_ads              
No       0.708  0.292
Yes      0.715  0.285
```

Each row now adds up to 1: once you are inside a filter, the outcomes inside it must cover everything. (`normalize="columns"` conditions on the column instead, and `normalize="all"` gives the joint probabilities.)

!!! warning "Common mistake: swapping the two sides of the bar"
    \( P(\text{high} \mid \text{ads}) = 0.285 \) but \( P(\text{ads} \mid \text{high}) = 0.624 \). Same 83 apps on top, different denominators (291 vs 133). The second number is big only because most apps in the whole store have ads (63%); it says nothing about ads *causing* good ratings. Confusing the two directions has a name in law, the **prosecutor's fallacy**: "the chance of this evidence if the suspect is innocent is tiny" is not "the chance the suspect is innocent given this evidence is tiny". Before computing, always say out loud which event is the filter.

## Independence

Look at the two conditionals again: 29.2% of apps without ads are highly rated, and 28.5% of apps with ads. Knowing whether an app has ads barely changes the chance of a high rating. When knowing \( B \) doesn't change the probability of \( A \) **at all**, the events are **independent**:

\[
P(A \mid B) = P(A)
\]

Substitute the definition \( P(A \mid B) = P(A \cap B) / P(B) \) and multiply both sides by \( P(B) \) to get the version you met on Day 36, which is symmetric in \( A \) and \( B \) and needs no division:

\[
P(A \cap B) = P(A)\,P(B)
\]

So the test is: **does the joint probability equal the product of the marginals?** Here \( P(A)P(B) = 0.2879 \times 0.6299 = 0.1813 \), and the real joint is \( 83/462 = 0.1797 \). Very close: as far as this table can tell, ads and high ratings look independent.

Compare a pair that clearly isn't. Paid apps and ads:

```python
paid = apps["price_usd"] > 0
print(pd.crosstab(paid, apps["has_ads"], margins=True))
print(round((paid & ads).mean(), 4), round(paid.mean() * ads.mean(), 4))   # joint vs product
print(round(ads[paid].mean(), 3), round(ads.mean(), 3))                    # P(ads | paid) vs P(ads)
```

```text
has_ads     No  Yes  All
price_usd               
False      123  291  414
True        48    0   48
All        171  291  462
0.0 0.0654
0.0 0.63
```

Not one paid app shows ads. If the two were independent you'd expect about \( 0.0654 \times 462 \approx 30 \) apps that are both; there are 0. Knowing an app is paid moves \( P(\text{ads}) \) from 63% all the way down to 0%: strongly **dependent**.

**Is 28.5% vs 29.2% "really" different?** Real data never matches the product exactly, even for truly independent events, just as 100 coin flips rarely give exactly 50 heads. One honest way to judge: shuffle the "has ads" labels at random, which destroys any real link, and see how big a gap appears by pure chance. Day 45 builds this **permutation test** properly; here is a preview:

```python
rng = np.random.default_rng(38)
a, h = ads.to_numpy(), high.to_numpy()
observed = h[a].mean() - h[~a].mean()

gaps = np.empty(10_000)
for i in range(10_000):
    shuffled = rng.permutation(a)             # random "has ads" labels, same count of True
    gaps[i] = h[shuffled].mean() - h[~shuffled].mean()

print(round(observed, 4))                                 # -0.0072
print(np.mean(np.abs(gaps) >= abs(observed)))             # 0.9171
```

In 92% of random shuffles the gap was at least as big as the real one. A gap of 0.7 percentage points is exactly what chance produces, so the data gives no evidence against independence.

!!! warning "Common mistake: thinking \"independent\" means \"can't happen together\""
    Events that can't happen together are called **mutually exclusive** (or *disjoint*), and they are the opposite of independent. "Paid" and "has ads" are nearly disjoint in this data, and learning that an app is paid tells you a lot: it almost certainly has no ads. Independent means *learning one tells you nothing about the other*, so both can, and usually do, happen together (83 apps have ads **and** a high rating).

## The multiplication rule

Rearranging the definition of conditional probability gives a rule for "and" that works **whether or not** the events are independent:

\[
P(A \cap B) = P(B)\,P(A \mid B) = P(A)\,P(B \mid A)
\]

In words: the chance of both = the chance of the first × the chance of the second *given the first happened*. Day 36's \( P(A)P(B) \) is the special case where the "given" changes nothing.

**Worked example.** Pick two different apps at random from the 462. What is the chance that both have ads?

- First pick: 291 of 462 have ads, so \( P(\text{first has ads}) = 291/462 \).
- Second pick, **given** the first had ads: one ad app is gone, so 290 of the remaining 461 have ads.

\[
P(\text{both have ads}) = \frac{291}{462} \times \frac{290}{461} = 0.6299 \times 0.6291 = 0.3962
\]

The two picks are *not* independent (drawing without replacement changes what's left), and the multiplication rule handles it. Check by simulation:

```python
a = apps["ads"].to_numpy()
n = len(a)
trials = 200_000
first = rng.integers(n, size=trials)
second = rng.integers(n - 1, size=trials)
second = second + (second >= first)            # shift past the first pick: never the same app twice

print(round(291 / 462 * 290 / 461, 4))         # 0.3962   the rule
print(round((a[first] & a[second]).mean(), 4)) # 0.3979   the simulation
```

The rule chains: \( P(A \cap B \cap C) = P(A)\,P(B \mid A)\,P(C \mid A \cap B) \), each factor conditioned on everything before it. Language models use exactly this chain: the probability of a sentence is \( P(\text{word}_1) \times P(\text{word}_2 \mid \text{word}_1) \times P(\text{word}_3 \mid \text{words } 1, 2) \times \dots \), and the model's whole job is to predict each "next word given the previous ones".

## The law of total probability

Sometimes you know \( P(A) \) only **inside** each group. Then you get the overall \( P(A) \) by weighting each group's conditional by the group's size:

\[
P(A) = P(A \mid B)\,P(B) + P(A \mid \text{not } B)\,P(\text{not } B)
\]

Each term is the multiplication rule, \( P(A \cap B) \) and \( P(A \cap \text{not } B) \); together they cover every app exactly once. With our numbers:

\[
P(\text{high}) = 0.2852 \times 0.6299 + 0.2924 \times 0.3701 = 0.1797 + 0.1082 = 0.2879
\]

which is \( 133/462 \). The same works with any number of groups, as long as every item is in exactly one group. The Play Store categories split the apps into 8 such groups:

```python
apps["category"] = apps["category"].str.strip().str.title()      # fix the spelling mess (Day 17)
by_cat = apps.groupby("category")["high"].agg(p_high="mean", count="size")
by_cat["weight"] = by_cat["count"] / len(apps)                   # P(category)
print(by_cat.round(3))
print(round((by_cat["p_high"] * by_cat["weight"]).sum(), 4), round(apps["high"].mean(), 4))
```

```text
                  p_high  count  weight
category                               
Education          0.246     65   0.141
Finance            0.185     27   0.058
Games              0.360    114   0.247
Health & Fitness   0.200     40   0.087
Music & Audio      0.371     35   0.076
Photography        0.283     46   0.100
Productivity       0.279     61   0.132
Tools              0.270     74   0.160
0.2879 0.2879
```

\[
P(\text{high}) = \sum_{k} P(\text{high} \mid \text{category } k)\; P(\text{category } k)
\]

The \( \sum_k \) ("sum over \( k \)") adds one term per category. Notice it's a **weighted** average: the plain average of the 8 `p_high` values is 0.274, not 0.288, because Games, the biggest category, also has one of the highest rates.

A **probability tree** draws both rules at once. Here is the one for the medical test you're about to meet: multiply along each path to get a joint probability, add the paths that end in the same outcome to get its total probability.

<figure class="diagram" markdown>
![Probability tree: 1000 people split into 10 sick (1%) and 990 healthy (99%); the sick split into 9 who test positive (90%) and 1 who tests negative (10%); the healthy split into 89 who test positive (9%) and 901 who test negative (91%); next to each leaf the product of the branch probabilities, 0.009, 0.001, 0.0891 and 0.9009; the two positive leaves are highlighted and add up to P of positive equals 0.0981](../../assets/img/phase-1/day38-probability-tree.svg)
<figcaption>Multiply along a path (multiplication rule), add across paths that end the same way (total probability). The counts are rounded: 9% of 990 is 89.1.</figcaption>
</figure>

## Bayes' rule, derived from the table

The multiplication rule gave two ways to write the same joint probability:

\[
P(A \mid B)\,P(B) = P(A \cap B) = P(B \mid A)\,P(A)
\]

Divide both ends by \( P(B) \). That's it, that's Bayes' rule:

\[
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}
\]

It turns a conditional around: from \( P(B \mid A) \), which you know, to \( P(A \mid B) \), which you want. Check it on the table, starting from \( P(\text{ads} \mid \text{high}) = 83/133 = 0.624 \) and recovering \( P(\text{high} \mid \text{ads}) \):

\[
P(\text{high} \mid \text{ads}) = \frac{P(\text{ads} \mid \text{high})\,P(\text{high})}{P(\text{ads})} = \frac{0.6241 \times 0.2879}{0.6299} = 0.285
\]

Same as the direct count, 83/291. In count form it is obvious: \( \frac{83}{133} \times \frac{133}{462} \div \frac{291}{462} = \frac{83}{291} \). Bayes' rule is nothing more than "re-divide the same cell by the other margin".

The parts have names, which you'll meet all over ML. Use \( H \) for the **hypothesis** you care about (sick, spam, cat) and \( E \) for the **evidence** you observed (positive test, the word "free", these pixels):

\[
\underbrace{P(H \mid E)}_{\text{posterior}} = \frac{\overbrace{P(E \mid H)}^{\text{likelihood}}\;\; \overbrace{P(H)}^{\text{prior}}}{\underbrace{P(E)}_{\text{evidence}}}
\]

| Part | Meaning | Medical-test example |
|---|---|---|
| **Prior** \( P(H) \) | What you believed **before** seeing the evidence: the base rate | 1% of people have the disease |
| **Likelihood** \( P(E \mid H) \) | How expected the evidence is **if** the hypothesis is true | The test catches 90% of sick people |
| **Evidence** \( P(E) \) | How common the evidence is overall, from total probability | 9.81% of all tests come back positive |
| **Posterior** \( P(H \mid E) \) | What you should believe **after** seeing the evidence | ? (next section) |

The denominator is almost always computed with the law of total probability, over "H true" and "H false":

\[
P(E) = P(E \mid H)\,P(H) + P(E \mid \text{not } H)\,P(\text{not } H)
\]

## The base-rate trap: a medical test

A disease affects **1%** of people. A test for it has:

- **sensitivity** 90%: if you're sick, it's positive with probability 0.90, \( P(+ \mid \text{sick}) = 0.90 \);
- **false-positive rate** 9%: if you're healthy, it's *still* positive with probability 0.09, \( P(+ \mid \text{healthy}) = 0.09 \).

You test positive. How likely is it that you're sick? Most people, including many doctors in published surveys, answer "about 90%". Let's see.

**By natural frequencies** (counts of people, which brains handle far better than percentages). Imagine 1000 people:

- 1% are sick: **10** people. The test catches 90% of them: **9** positives (1 missed).
- 990 are healthy. 9% of them test positive anyway: \( 0.09 \times 990 = 89.1 \approx \) **89** false positives.
- Positives in total: \( 9 + 89 = 98 \). Of those, only **9** are sick.

\[
P(\text{sick} \mid +) \approx \frac{9}{98} \approx 9.2\%
\]

Not 90%: about 9%. Fewer than one positive in ten is actually sick, because the healthy group is 99 times bigger, so even its small 9% error rate produces ten times more positives than the sick group does.

<figure class="diagram" markdown>
![Animation of 1000 grey dots in a 40 by 25 grid: first the 10 sick people turn red; then the test runs and 9 of the red dots stay red as positives while 89 healthy dots turn orange as false positives; finally all negative dots fade out, leaving 98 positive dots of which only 9 are red, and the panel reads P of sick given positive equals 9 over 98, about 9.2 percent](../../assets/img/phase-1/day38-natural-frequencies.svg)
<figcaption>Animated: 1000 people, 10 sick. A positive test keeps 98 people, and only 9 of them (red) are sick: the 89 orange false positives come from the huge healthy group.</figcaption>
</figure>

**By Bayes' rule**, with the exact numbers:

\[
P(\text{sick} \mid +) = \frac{0.90 \times 0.01}{0.90 \times 0.01 + 0.09 \times 0.99} = \frac{0.009}{0.009 + 0.0891} = \frac{0.009}{0.0981} = 0.0917
\]

The top is the "sick and positive" path of the tree, the bottom is both positive paths added. Write it once as a function:

```python
def bayes(prior, p_e_given_h, p_e_given_not_h):
    """Posterior P(H | E) from the prior P(H) and the two likelihoods."""
    evidence = p_e_given_h * prior + p_e_given_not_h * (1 - prior)   # total probability
    return p_e_given_h * prior / evidence

print(round(bayes(0.01, 0.90, 0.09), 4))      # 0.0917
```

**By simulation**, which doesn't care about formulas at all: invent a million people, make 1% of them sick, run the test on everyone, then filter to the positives and count.

```python
rng = np.random.default_rng(38)
N = 1_000_000
sick = rng.random(N) < 0.01                     # True for about 1% of people
positive = np.where(sick,
                    rng.random(N) < 0.90,       # sick: positive with probability 0.90
                    rng.random(N) < 0.09)       # healthy: positive with probability 0.09

print(sick.sum(), positive.sum(), (sick & positive).sum())   # 10017 98165 9054
print(round(sick[positive].mean(), 4))                       # 0.0922   P(sick | positive)
```

Three methods, one answer: about 9%. (`np.where(cond, x, y)` picks from `x` where `cond` is True and from `y` elsewhere, elementwise.)

**The prior is doing the work.** Keep the same test and change only how common the disease is:

```python
for prior in [0.001, 0.01, 0.1, 0.5]:
    print(f"prior {prior:5}:  P(sick | +) = {bayes(prior, 0.90, 0.09):.4f}")
# prior 0.001:  P(sick | +) = 0.0099
# prior  0.01:  P(sick | +) = 0.0917
# prior   0.1:  P(sick | +) = 0.5263
# prior   0.5:  P(sick | +) = 0.9091
```

The identical test is nearly useless for screening a rare disease (1% posterior) and very convincing in a clinic where half the patients have it (91%). "How good is the test?" has no answer without "how common is the thing?".

!!! warning "Common mistake: ignoring the base rate"
    Treating \( P(+ \mid \text{sick}) = 0.9 \) as if it were \( P(\text{sick} \mid +) \) is the **base-rate fallacy**: it is the "swapping sides of the bar" mistake again, and it hides the prior. Whenever someone quotes an accuracy, a sensitivity or a "detects 99% of fraud", ask: *out of how many, and how rare is the thing?* Then draw 1000 dots.

### Updating twice, and the odds shortcut

A posterior can be the prior for the next piece of evidence. Test positive a **second** time (with an independent repeat of the test), and start from 9.17% instead of 1%:

```python
p1 = bayes(0.01, 0.90, 0.09)
p2 = bayes(p1, 0.90, 0.09)
print(round(p1, 4), round(p2, 4))     # 0.0917 0.5025
```

Two positives: about 50%. This "believe, observe, update, repeat" loop is the heart of **Bayesian** reasoning.

There is a tidier way to see it. **Odds** are "for : against", \( \text{odds} = p / (1 - p) \): a 1% chance is odds of 1 : 99. The **likelihood ratio** of the evidence is how many times more likely it is under \( H \) than under not-\( H \): here \( 0.90 / 0.09 = 10 \). Then Bayes' rule becomes a single multiplication:

\[
\text{posterior odds} = \text{prior odds} \times \text{likelihood ratio}
\]

- Before: 1 : 99. One positive: \( 10 : 99 \), probability \( 10/109 = 0.0917 \).
- Second positive: \( 100 : 99 \), probability \( 100/199 = 0.5025 \).

(Divide the posterior form of Bayes' rule for \( H \) by the same form for not-\( H \): the evidence \( P(E) \) cancels, which is why odds are so convenient.) Each independent piece of evidence multiplies the odds by its own likelihood ratio. Take logs and the multiplications become additions; that is exactly how the naive Bayes spam filter of Day 41 adds up the evidence from each word.

## A spam filter, by hand and by simulation

Suppose 20% of incoming messages are spam. The word "free" appears in 60% of spam messages but only 5% of normal messages ("ham"). A message contains "free". Is it spam?

- **Prior**: \( P(\text{spam}) = 0.20 \).
- **Likelihoods**: \( P(\text{free} \mid \text{spam}) = 0.60 \), \( P(\text{free} \mid \text{ham}) = 0.05 \).

Natural frequencies, out of 1000 messages: 200 spam, of which \( 0.6 \times 200 = 120 \) say "free"; 800 ham, of which \( 0.05 \times 800 = 40 \) say "free". So 160 messages say "free" and 120 of them are spam:

\[
P(\text{spam} \mid \text{free}) = \frac{0.60 \times 0.20}{0.60 \times 0.20 + 0.05 \times 0.80} = \frac{0.12}{0.12 + 0.04} = \frac{120}{160} = 0.75
\]

```python
print(bayes(0.20, 0.60, 0.05))          # 0.75

rng = np.random.default_rng(41)
N = 100_000
spam = rng.random(N) < 0.20
has_free = np.where(spam, rng.random(N) < 0.60, rng.random(N) < 0.05)
print(round(has_free.mean(), 3), round(spam[has_free].mean(), 3))   # 0.161 0.745
```

One word moved us from 20% to 75%: the likelihood ratio of "free" is \( 0.60 / 0.05 = 12 \), so the odds went from 1 : 4 to 12 : 4 = 3 : 1. A real filter combines dozens of words, each multiplying the odds by its own ratio (Exercise 4, and the Day 41 mini-project).

## Where Bayes shows up in machine learning

- **A classifier outputs a posterior.** When a model says "spam: 0.93", it's estimating \( P(\text{spam} \mid \text{this message}) \). That number silently contains the **prior** of the data the model was trained on.
- **Precision is a posterior too.** For a detector, *recall* (sensitivity) is \( P(\text{flagged} \mid \text{spam}) \), and *precision* is \( P(\text{spam} \mid \text{flagged}) \): Bayes' rule links them through the base rate. The same detector has very different precision on data where the positive class is rare, exactly like the medical test table above. This is why a fraud model with "99% accuracy" can be useless, and why Phase 2 measures precision and recall separately on **imbalanced** data.
- **Naive Bayes** classifiers (Day 41) apply the odds form literally: prior odds × one likelihood ratio per word.
- **Priors on parameters.** In Bayesian statistics the unknown is a model parameter itself: a prior belief about the weights, updated by the data. Day 47's maximum likelihood is the "no prior" version, and adding a prior that prefers small weights turns out to be the L2 regularization you previewed on Day 22.

!!! android "Android bridge: thresholds and base rates for an on-device notification classifier"
    Say you ship a small text model (a LiteRT model, or MediaPipe's Text Classifier task) that scores incoming notifications as spam. It was trained on a **balanced** dataset, half spam and half not, so its scores assume a prior of 0.5. On a real user's phone maybe only 5% of notifications are spam. A score of 0.9 then does **not** mean a 90% chance of spam.

    If the model's likelihoods carry over and only the base rate changed, the odds form fixes the score: divide out the training prior odds and multiply in the device's prior odds.

    ```kotlin
    fun adjustForBaseRate(score: Double, trainPrior: Double, devicePrior: Double): Double {
        val odds = score / (1 - score)
        val priorRatio = (devicePrior / (1 - devicePrior)) / (trainPrior / (1 - trainPrior))
        val newOdds = odds * priorRatio
        return newOdds / (1 + newOdds)
    }

    adjustForBaseRate(0.90, trainPrior = 0.5, devicePrior = 0.05)   // ≈ 0.321
    ```

    So with a "hide if score > 0.5" rule tuned on balanced data, most hidden notifications on that phone could be real ones. Either correct the score like this, or pick the threshold from **precision measured on data with the real base rate** (for example, from users' "not spam" feedback). It's the same thinking as a staged rollout: a crash rate measured on 1% of users is only meaningful once you know which 1%.

Check the Kotlin arithmetic in Python:

```python
def adjust_for_base_rate(score, train_prior, device_prior):
    odds = score / (1 - score)
    prior_ratio = (device_prior / (1 - device_prior)) / (train_prior / (1 - train_prior))
    new_odds = odds * prior_ratio
    return new_odds / (1 + new_odds)

print(round(adjust_for_base_rate(0.90, 0.5, 0.05), 4))    # 0.3214
```

!!! warning "Common mistake: trusting a conditional built on a handful of rows"
    Conditioning shrinks the data. \( P(\text{high} \mid \text{Finance}) = 0.185 \) rests on only 27 apps: five highly rated ones. One app more or less moves it by almost 4 percentage points. Always print the **count** next to a conditional probability (as `by_cat` does), and be suspicious of any filter that leaves fewer than a few dozen rows.

---

## Exercises

1. From the ads × rating table, compute by hand, then with pandas: \( P(\text{no ads} \mid \text{high}) \), \( P(\text{high} \mid \text{no ads}) \) and \( P(\text{rating} < 4.5 \mid \text{ads}) \). Which two of your answers, together with ones from the lesson, must add up to 1, and why?
2. Are "paid" and "high rating" independent in the dataset? Compare \( P(\text{high} \mid \text{paid}) \) with \( P(\text{high} \mid \text{free}) \), and the joint probability with the product of the marginals.
3. Redo the medical test for a clinic where 10% of patients have the disease (same 90% sensitivity, 9% false-positive rate): first with 1000 people as natural frequencies, then with `bayes`, then by simulating 1,000,000 patients.
4. Add a second word to the spam filter: "winner" appears in 30% of spam and 1% of ham. Assuming the two words appear independently **within** each class (the "naive" assumption of Day 41), what is \( P(\text{spam} \mid \text{free and winner}) \)? Compute it with the odds form and with `bayes` applied twice.
5. The Monty Hall problem. A prize is behind one of three doors. You pick door 1. The host, who knows where the prize is, always opens a door you didn't pick that has no prize, and offers you a switch. Simulate 100,000 games with `rng` and estimate the chance of winning if you stay and if you switch. Then explain the answer with natural frequencies.
6. A notification classifier has recall \( P(\text{flagged} \mid \text{spam}) = 0.95 \) and false-positive rate \( P(\text{flagged} \mid \text{not spam}) = 0.02 \). What is its precision on a phone where 2% of notifications are spam, and on one where 30% are? Write a function `precision(base_rate, recall, fpr)`.

??? success "Solutions"
    ```python
    # 1: 50 apps have no ads and a high rating; 133 are high, 171 have no ads, 208 of 291 ad apps are below 4.5
    print(round(50 / 133, 3), round(50 / 171, 3), round(208 / 291, 3))      # 0.376 0.292 0.715
    print(round((~apps["ads"])[apps["high"]].mean(), 3))                    # 0.376
    print(round(apps["high"][~apps["ads"]].mean(), 3))                      # 0.292
    print(round((~apps["high"])[apps["ads"]].mean(), 3))                    # 0.715
    ```
    \( P(\text{rating} < 4.5 \mid \text{ads}) = 0.715 \) and \( P(\text{high} \mid \text{ads}) = 0.285 \) add to 1: same filter (ads), complementary events inside it. Likewise \( P(\text{no ads} \mid \text{high}) = 0.376 \) and \( P(\text{ads} \mid \text{high}) = 0.624 \). But \( P(\text{high} \mid \text{no ads}) + P(\text{high} \mid \text{ads}) = 0.577 \) need not be 1: those are two different filters.

    ```python
    # 2
    paid = apps["price_usd"] > 0
    high = apps["high"]
    print(round(high[paid].mean(), 3), round(high[~paid].mean(), 3))          # 0.271 0.29
    print(round((paid & high).mean(), 4), round(paid.mean() * high.mean(), 4))  # 0.0281 0.0299
    ```
    27.1% vs 29.0%, and 0.0281 vs 0.0299 (13 apps vs the 13.8 independence would predict): close, on only 48 paid apps. No evidence of dependence, unlike paid vs ads.

    ```python
    # 3: 1000 patients: 100 sick → 90 positive; 900 healthy → 81 positive; 90 / 171
    print(round(90 / 171, 4), round(bayes(0.10, 0.90, 0.09), 4))    # 0.5263 0.5263
    rng = np.random.default_rng(3)
    N = 1_000_000
    sick = rng.random(N) < 0.10
    positive = np.where(sick, rng.random(N) < 0.90, rng.random(N) < 0.09)
    print(round(sick[positive].mean(), 4))                            # 0.5266
    ```

    ```python
    # 4: prior odds 0.2 / 0.8 = 0.25; likelihood ratios 0.60 / 0.05 = 12 and 0.30 / 0.01 = 30
    odds = 0.25 * 12 * 30
    print(odds, round(odds / (1 + odds), 4))                          # 90.0 0.989
    print(round(bayes(bayes(0.20, 0.60, 0.05), 0.30, 0.01), 4))       # 0.989
    ```
    The first posterior (0.75) is the prior for the second word. This is only correct if, among spam messages, seeing "free" doesn't change the chance of seeing "winner" (and the same among ham). Real words are not that independent, which is why the method is called *naive*; it still works surprisingly well.

    ```python
    # 5
    rng = np.random.default_rng(5)
    games = 100_000
    prize = rng.integers(3, size=games)       # doors 0, 1, 2; you always pick door 0
    # the host opens an empty door that isn't yours: a random one of 1 and 2 if the prize is behind 0,
    # otherwise the only empty one (prize behind 1 → open 2, prize behind 2 → open 1)
    host = np.where(prize == 0, rng.integers(1, 3, size=games), 3 - prize)
    switch_to = 3 - host                      # the door that is neither yours (0) nor the host's
    stay_wins = prize == 0
    switch_wins = prize == switch_to
    print(round(stay_wins.mean(), 3), round(switch_wins.mean(), 3))   # 0.333 0.667
    ```
    Out of 300 games, the prize is behind your door in about 100: staying wins those. In the other 200 the prize is behind one of the two other doors, and the host is **forced** to open the empty one of them, so switching wins all 200. The host's choice is evidence: it depends on where the prize is, so it changes the probabilities (a likelihood that isn't the same under every hypothesis).

    ```python
    # 6: precision = P(spam | flagged), which is Bayes' rule with prior = base rate
    def precision(base_rate, recall, fpr):
        return bayes(base_rate, recall, fpr)

    print(round(precision(0.02, 0.95, 0.02), 4))      # 0.4922
    print(round(precision(0.30, 0.95, 0.02), 4))      # 0.9532
    ```
    The same model flags mostly real notifications (51% not spam) on the first phone and is right 95% of the time on the second. Recall and false-positive rate are properties of the model; precision is a property of the model **and** the data it runs on.

## Resources

- [3Blue1Brown: Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM): the 1000-dots picture, in a beautiful 15 minutes. Watch it first.
- [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/): interactive visual chapters on probability, including conditional probability and Bayesian inference.
- [Khan Academy: Probability library](https://www.khanacademy.org/math/statistics-probability/probability-library): the "Conditional probability and independence" lessons with practice problems.
- [Think Bayes, 2nd edition (free online book) by Allen Downey](https://allendowney.github.io/ThinkBayes2/): Bayes' rule with Python and pandas, in today's style; chapter 1 computes conditional probabilities from a real survey dataset.
- [Harvard Stat 110 (Joe Blitzstein)](https://projects.iq.harvard.edu/stat110): free lectures and problem sets; lectures 4–6 cover conditional probability, Bayes and the Monty Hall problem.
- [Mathematics for Machine Learning (free book), ch. 6 "Probability and Distributions"](https://mml-book.github.io/): sum rule, product rule and Bayes' theorem, formally.
- [Wikipedia: Base rate fallacy](https://en.wikipedia.org/wiki/Base_rate_fallacy): more worked examples, including the medical-test and drunk-driver cases.

## Checklist

- [ ] I can compute marginal, joint and conditional probabilities from a 2×2 count table, and say which number I'm dividing by.
- [ ] I can explain with an example why \( P(A \mid B) \neq P(B \mid A) \).
- [ ] I can test independence with \( P(A \cap B) = P(A)P(B) \), and I know independent is not the same as mutually exclusive.
- [ ] I can use the multiplication rule (including drawing without replacement) and the law of total probability, and draw them as a tree.
- [ ] I can derive Bayes' rule from the multiplication rule and name the prior, likelihood, evidence and posterior.
- [ ] I can work the medical-test example with 1000 dots, with the formula and by simulation, and explain why the answer is about 9%, not 90%.
- [ ] I can update in odds form and explain why precision depends on the base rate.

**Next:** Day 39: Expectation, variance and the law of large numbers (coming soon)
