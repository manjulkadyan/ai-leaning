# Day 16: Selecting, Filtering and Sorting

**Phase 1 · Week 3 · ~90 minutes**

## Objectives

- [ ] Pick columns with `[]` and know when you get a Series and when you get a DataFrame.
- [ ] Pick rows and columns with `.loc` (by label) and `.iloc` (by position), and say which one includes the end of a slice.
- [ ] Filter rows with boolean masks, combining conditions with `&`, `|`, `~` and parentheses.
- [ ] Recognize the `and` error and the missing-parentheses error on sight, and fix them.
- [ ] Use `.isin`, `.between` and `.query` as shortcuts for common filters.
- [ ] Sort with `sort_values` (one or more keys), take the top rows with `nlargest`, add columns with `assign`, and count values with `value_counts`.

**Prerequisites.** [Day 15](day-15-pandas-dataframes.md) (Series, DataFrame, `read_csv`, `head`/`info`/`describe`, the index). Boolean arrays from [Day 4](day-04-numpy-arrays.md) help but aren't required.

---

## The big idea

Day 15 got the Play Store table into memory. Almost every question you'll ask about it has the same shape: *which rows, which columns, in what order?* "Free apps rated 4.5 or better, most reviewed first, just name and rating" is three operations: **filter** the rows, **select** the columns, **sort** the result.

**Analogy.** Think of a spreadsheet with filter buttons on the header row. You tick "price = 0" and the other rows don't vanish from the file; they're just hidden from your view. pandas works the same way: filtering builds a *new* table out of the rows you asked for, and the original stays as it was.

If you've written SQL or Room queries, you already know the concepts. `SELECT name, rating FROM apps WHERE price = 0 AND rating >= 4.5 ORDER BY reviews DESC` maps one-to-one onto what you'll write today. Only the syntax is new, and pandas has two syntax traps (`&` instead of `and`, and parentheses) that catch everyone once. You'll meet both on purpose.

Why it matters for ML: before any model is trained, someone chooses the feature columns (`X`), the target column (`y`), drops rows that don't belong, and splits the rows into training and test sets. All of that is selecting and filtering.

## Setup

Load the course dataset (see the [Datasets](../../datasets.md) page for the column guide, or download the CSV and pass a local path instead of the URL):

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")
apps.shape    # (486, 12)

cols = ["app_name", "category", "rating", "reviews", "price_usd"]
print(apps[cols].head().to_string())
```

```text
            app_name      category  rating  reviews  price_usd
0     Bright Quest X  Productivity     3.8     2581       0.99
1      Quick Cleaner  Productivity     3.9    65066       0.00
2  Pocket Radio Kids         Tools     3.7    54869       0.00
3    Lumen Learn Pro         Games     4.0     4936       4.99
4        Quick Quest  productivity     NaN     5041       0.00
```

This lesson works on the **raw** data on purpose. You can already see the mess: `productivity` in lower case, a missing rating (`NaN`, "not a number", pandas' marker for a missing value). Cleaning is Day 17; today, watch how the mess changes your answers. Everything here was tested on pandas 3.0.

## Selecting columns with `[]`

Square brackets on a DataFrame are for **columns**:

```python
r = apps["rating"]          # one name   → a Series
type(r)                     # <class 'pandas.Series'>

two = apps[["app_name", "rating"]]   # a LIST of names → a DataFrame
type(two)                   # <class 'pandas.DataFrame'>
apps[["rating"]].shape      # (486, 1)   a list of one name is still a DataFrame
```

The double brackets aren't special syntax: the outer `[]` is the selection, the inner `[...]` is an ordinary Python list of column names. You'll write `X = apps[["rating", "reviews", "price_usd"]]` to pick model features (a 2-D table) and `y = apps["rating"]` for a target (1-D), so this distinction matters.

A name that doesn't exist raises `KeyError`:

```python
apps["ratin"]                  # KeyError: 'ratin'
apps[["app_name", "ratin"]]    # KeyError: "['ratin'] not in index"
apps[0]                        # KeyError: 0     ([] looks for a COLUMN named 0)
```

`apps[0]` failing surprises people coming from lists: plain `[]` with a single value never means "row 0". (Confusingly, a *slice* such as `apps[0:3]` does select rows. Rather than memorize that exception, use `.iloc` for rows by position, as below.)

## `.loc` and `.iloc`: rows and columns together

DataFrames have two indexers that take `[rows, columns]`, just like a 2-D NumPy array:

- **`.loc[...]`**: **loc**ation by **label**. Rows by their index label, columns by name.
- **`.iloc[...]`**: **i**nteger **loc**ation. Rows and columns by position, counting from 0, exactly like NumPy.

```python
apps.loc[2, "app_name"]                  # 'Pocket Radio Kids'   row label 2, column "app_name"
apps.iloc[[0, 2, 4], [1, 3]]             # rows at positions 0, 2, 4; columns at positions 1 and 3
```

```text
            app_name  rating
0     Bright Quest X     3.8
2  Pocket Radio Kids     3.7
4        Quick Quest     NaN
```

Right after `read_csv`, the index is `0, 1, 2, ...`, so labels and positions happen to be the same numbers and the two indexers *look* identical. To see the real difference, give the table an index made of strings. Take the first five rows and use `app_id` as the index:

```python
small = apps.loc[0:4, ["app_id", "app_name", "rating"]].set_index("app_id")

small.loc["com.quick.cleaner":"com.quick.quest"]   # 4 rows: the stop label is INCLUDED
small.iloc[1:4]                                    # 3 rows: the stop position is EXCLUDED
```

<figure class="diagram" markdown>
![Two copies of the same five-row table with positions 0 to 4 and app_id labels. Left: small.loc from com.quick.cleaner to com.quick.quest highlights positions 1 to 4, four rows, because the stop label is included. Right: small.iloc[1:4] highlights positions 1 to 3, three rows, and row 4 is outlined in red dashes as excluded](../../assets/img/phase-1/day16-loc-vs-iloc.svg)
<figcaption>.loc slices by label and includes the end; .iloc slices by position and excludes it, like a Python list.</figcaption>
</figure>

Why would `.loc` include the end? Because with labels there is no "one past the last label": to say "up to `com.quick.quest`" you must be able to name `com.quick.quest` itself. With positions, Python's half-open convention (stop is excluded) still applies, so `.iloc[1:4]` has `4 - 1 = 3` rows.

In math terms, if a table has \( n \) rows at positions \( 0, 1, \dots, n-1 \), then `.iloc[a:b]` is the set of positions \( \{ i : a \le i < b \} \) and has \( b - a \) rows. For `.iloc[1:4]` that's \( \{1, 2, 3\} \): 3 rows. `.loc["x":"y"]` instead runs from wherever label `x` sits to wherever label `y` sits, **both included**.

### The trap: after sorting, labels and positions split apart

Sorting (later in this lesson) keeps each row's index label. So the "labels equal positions" coincidence breaks:

```python
top = apps.sort_values("rating", ascending=False)
print(top[["app_name", "rating"]].head(3).to_string())
```

```text
              app_name  rating
485     Daily Camera X     5.0
435  Bright Quest Lite     5.0
82       Mega Coach Go     5.0
```

```python
top.iloc[0:3]       # the 3 best-rated apps (positions 0, 1, 2), labels 485, 435, 82
top.loc[0:3]        # an EMPTY DataFrame, shape (0, 12)
top.loc[0, "app_name"]   # 'Bright Quest X'  the row LABELLED 0, wherever it now sits
```

`top.loc[0:3]` means "from the row labelled 0 to the row labelled 3", and in the sorted table label 3 comes *before* label 0, so the range is empty. No error, just nothing. Rule: **for "the first N rows" use `.iloc` (or `.head`); for "the row called X" use `.loc`.** If you want positions and labels to agree again, call `.reset_index(drop=True)`.

!!! warning "Common mistake: assuming labels are unique"
    Index labels don't have to be unique. Our dataset has 6 duplicated rows (fixed on Day 17), so with `app_id` as the index one label can match two rows:

    ```python
    by_id = apps.set_index("app_id")
    by_id.index.is_unique          # False
    by_id.loc["com.pocket.radio.kids", ["app_name", "rating", "reviews"]]
    #                                 app_name  rating  reviews
    # app_id
    # com.pocket.radio.kids  Pocket Radio Kids     3.7    54869
    # com.pocket.radio.kids  Pocket Radio Kids     3.7    54869
    ```

    You asked for "one app" and got a 2-row DataFrame instead of a Series. Code that expected one row then breaks later and far away. Check `df.index.is_unique` when you build an index from a column.

## Filtering with boolean masks

Here is the most important idea of the lesson. Comparing a column with a value doesn't give one `True` or `False`. Like NumPy (Day 4), it compares **every element**, producing a Series of booleans with the same index, called a **mask**:

```python
six = apps.loc[0:5, ["app_name", "rating"]]
mask = six["rating"] >= 4.0
mask.tolist()       # [False, False, False, True, False, True]

six[mask]           # [] with a boolean Series keeps the rows where it is True
```

```text
           app_name  rating
3   Lumen Learn Pro     4.0
5  Orbit Tasks Kids     4.8
```

<figure class="diagram" markdown>
![Animation: a six-row table of app names and ratings; a yellow mask column fills in False, False, False, True, False, True next to it, with a note that NaN >= 4.0 is False; the four False rows fade and slide away, and the two True rows, labelled 3 and 5, slide into a result table on the right](../../assets/img/phase-1/day16-boolean-mask.svg)
<figcaption>Animated: one comparison makes a True/False column, then only the True rows are kept, with their original index labels.</figcaption>
</figure>

Three things to notice:

1. **Row 4 has rating `NaN`, and `NaN >= 4.0` is `False`.** Any comparison with a missing value is `False`, so missing values silently drop out of every filter.
2. **The index labels 3 and 5 are kept.** Filtering never renumbers rows, which is exactly why the sorting trap above exists.
3. **`apps[mask]` is new data.** `apps` itself still has 486 rows.

On the full table:

```python
mask = apps["rating"] >= 4.5
mask.dtype                  # dtype('bool')
mask.sum()                  # 135          number of True values
mask.mean()                 # 0.2777...    fraction of apps rated 4.5+
apps[mask].shape            # (135, 12)
```

### The math: a mask is a 0/1 vector

Why do `sum()` and `mean()` give a count and a fraction? Write the mask as a vector \( \mathbf{m} \) with one entry per row:

\[
m_i = \begin{cases} 1 & \text{if row } i \text{ meets the condition} \\ 0 & \text{otherwise} \end{cases}
\]

- \( i \) is the row position, from 1 to \( n \) (here \( n = 486 \) rows).
- \( m_i \) is the entry for row \( i \): Python's `True` counts as 1 and `False` as 0.

Then the number of matching rows is the sum of the entries, and the fraction is their average:

\[
\text{count} = \sum_{i=1}^{n} m_i, \qquad \text{fraction} = \frac{1}{n} \sum_{i=1}^{n} m_i
\]

Worked example with the six-row table: \( \mathbf{m} = (0, 0, 0, 1, 0, 1) \), so count \( = 0+0+0+1+0+1 = 2 \) and fraction \( = 2/6 \approx 0.333 \), which is what `mask.sum()` and `mask.mean()` return for `six`. On the full table, \( 135 / 486 \approx 0.278 \).

Combining conditions is arithmetic on these 0/1 vectors: AND is multiplication (\( 1 \cdot 1 = 1 \), anything with a 0 gives 0), NOT is \( 1 - m_i \), and OR is "at least one is 1", \( \max(a_i, b_i) \).

### Combining conditions: `&`, `|`, `~` and parentheses

| Meaning | pandas | Kotlin equivalent |
|---|---|---|
| both | `(a) & (b)` | `a && b` |
| either | `(a) \| (b)` | `a \|\| b` |
| not | `~(a)` | `!a` |

```python
free_great = apps[(apps["rating"] >= 4.5) & (apps["price_usd"] == 0)]
free_great.shape     # (120, 12)

popular = apps[(apps["rating"] >= 4.8) | (apps["reviews"] > 100_000)]
popular.shape        # (58, 12)

no_ads = apps[~(apps["has_ads"] == "Yes")]     # same as apps["has_ads"] != "Yes"
no_ads.shape         # (183, 12)

# masks can be named and reused; long filters read better this way
quality = apps["rating"] >= 4.5
free = apps["price_usd"] == 0
no_ads_mask = apps["has_ads"] == "No"
print(apps.loc[quality & free & no_ads_mask, ["app_name", "rating", "reviews"]].head(3).to_string())
```

```text
            app_name  rating  reviews
35        Tiny Beats     4.5     2962
58  Pixel Radio Kids     4.5      463
76       Spark Radio     4.5      149
```

Notice the last line: `.loc[mask, columns]` filters rows and selects columns in one step. That's the idiomatic form.

### The two errors everyone hits

**Error 1: using `and` / `or`.** Python's `and` needs to turn each side into a single `True` or `False`. A Series of 486 booleans has no single truth value, so pandas refuses to guess:

```python
apps[(apps["rating"] >= 4.5) and (apps["price_usd"] == 0)]
# ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
```

`&` works element by element (it's NumPy's element-wise AND), which is what you want. When you see "truth value of a Series is ambiguous", search the line for `and`, `or`, `not`, or an `if` with a Series in it.

**Error 2: forgetting the parentheses.** In Python, `&` and `|` bind **tighter** than comparisons like `>=` and `==` (they were designed for integer bit operations, where that order makes sense). So this:

```python
apps[apps["rating"] >= 4.5 & apps["price_usd"] == 0]
```

is read by Python as `apps["rating"] >= (4.5 & apps["price_usd"]) == 0`, and it tries to compute `4.5 & <a column of prices>` first:

```text
TypeError: Cannot perform 'rand_' with a dtyped [float64] array and scalar of type [bool]
```

The message doesn't mention parentheses at all, which is what makes it hard to recognize. (`rand_` is "reversed and", the `&` being applied with the number on the left.) With a text column on the right you get a different message for the same mistake: `apps[apps["rating"] > 4.5 | apps["has_ads"] == "No"]` raises `TypeError: unsupported operand type(s) for |: 'float' and 'StringArray'`.

!!! warning "Common mistake: `&` without parentheses"
    The fix for both errors is the same habit: **every condition in its own parentheses, joined by `&`, `|` or `~`.**

    ```python
    apps[(apps["rating"] >= 4.5) & (apps["price_usd"] == 0)]    # correct
    ```

    If a `TypeError` mentions `rand_`, `ror_`, `&` or `|`, you've almost certainly lost a pair of parentheses.

!!! warning "Common mistake: `~` and missing values"
    `NaN >= 4` is `False`, so `~(apps["rating"] >= 4)` is `True` for the 18 unrated apps:

    ```python
    (apps["rating"] >= 4).sum()        # 317
    (apps["rating"] < 4).sum()         # 151
    (~(apps["rating"] >= 4)).sum()     # 169   = 151 + 18 unrated apps
    ```

    "Not rated 4 or more" and "rated below 4" are different questions when data is missing. Decide which you mean, and use `apps["rating"].isna()` to handle missing values explicitly. Day 17 is all about this.

## Shortcuts: `.isin`, `.between`, `.query`

**`.isin(list)`** replaces a chain of `==` joined with `|`:

```python
apps[apps["category"].isin(["Games", "Tools"])].shape     # (188, 12)
```

**`.between(low, high)`** replaces `(x >= low) & (x <= high)`. Both ends are included by default; pass `inclusive="neither"`, `"left"` or `"right"` to change that:

```python
apps[apps["rating"].between(4.0, 4.2)].shape                 # (104, 12)
apps["rating"].between(4.0, 4.2, inclusive="neither").sum()  # 29   only 4.1
```

**`.query("...")`** takes the whole filter as a string, where you can write `and`, `or`, `not` and chained comparisons such as `3.0 <= rating <= 3.5`. Refer to a Python variable with `@`:

```python
apps.query("rating >= 4.5 and price_usd == 0").shape     # (120, 12)  same as free_great

min_reviews = 50_000
print(apps.query("category == 'Tools' and reviews >= @min_reviews")[cols].to_string())
```

```text
              app_name category  rating  reviews  price_usd
2    Pocket Radio Kids    Tools     3.7    54869        0.0
145  Pocket Radio Kids    Tools     3.7    54869        0.0
188       Orbit Budget    Tools     4.4    84585        0.0
```

Two things to notice. Row 145 is a duplicate of row 2, so any count of "Tools apps with 50k+ reviews" is wrong by one until Day 17 removes duplicates. And `category == 'Tools'` only matches that exact text, which brings us to the mess.

!!! warning "Common mistake: trusting exact text matches on messy data"
    `apps["category"] == "Games"` matches 113 rows, but the raw data spells the category four ways:

    ```python
    [c for c in apps["category"].unique() if c.strip().lower() == "games"]
    # ['Games', 'games', ' Games', 'GAMES']
    ```

    Matching after normalizing the text finds 120. Seven games were silently left out of every filter, count and average. Day 17 fixes this once with `.str.strip().str.title()`; until then, treat any filter on `category` as approximate.

`.query` is pleasant for interactive exploration. In library code, many people prefer explicit masks: typos in the string only fail at run time, and column names with spaces need backticks inside the string.

## Sorting

`sort_values` returns a new, sorted DataFrame:

```python
print(apps.sort_values("reviews", ascending=False)[cols].head().to_string())
```

```text
             app_name       category  rating  reviews  price_usd
399   Orbit Lens Lite          Games     3.7  2525379       4.99
124  Daily Learn Lite      Education     4.1   690677       0.00
73         Mega Tasks          Games     4.4   674533       1.99
119       Calm Runner          Games     4.2   243129       0.00
170      Quick Budget  Music & Audio     4.7   214761       0.00
```

One app has 2.5 million reviews and the fifth has 215 thousand: a very skewed column (you'll plot this on Day 19).

**Several keys.** Pass lists: sort by the first key, and use the second only to break ties. Each key gets its own direction:

```python
by_quality = apps.sort_values(["rating", "reviews"], ascending=[False, False])
print(by_quality[cols].head().to_string())
```

```text
              app_name      category  rating  reviews  price_usd
138  Calm Cleaner Plus         Games     5.0   102896       0.00
160          Pixel Fit         Games     5.0    72984       0.00
485     Daily Camera X   Photography     5.0    34700       0.00
453      Zen Radio Pro     Education     5.0    31259       4.99
82       Mega Coach Go  Productivity     5.0    30302       0.00
```

Among the many apps tied at 5.0, the one with more reviews comes first. Missing values go to the end regardless of direction (`na_position="last"` is the default):

```python
apps.sort_values("rating")[["app_name", "rating"]].tail(3)
#            app_name  rating
# 440     Smart Tutor     NaN
# 458     Daily Tutor     NaN
# 470  Spark Beats Go     NaN
```

**Top N.** `nlargest(n, column)` and `nsmallest(n, column)` give the same result as sort-then-head, more directly and faster on big tables because they don't sort everything:

```python
apps.nlargest(5, "reviews")       # the same 5 rows as the sort above
```

Sorting by the index labels instead is `sort_index()`.

## Adding columns with `assign`

`assign` returns a **new** DataFrame with extra columns, leaving the original untouched:

```python
apps2 = apps.assign(
    is_free=apps["price_usd"] == 0,
    reviews_k=(apps["reviews"] / 1000).round(1),
)
apps.shape, apps2.shape      # ((486, 12), (486, 14))
print(apps2[["app_name", "price_usd", "is_free", "reviews", "reviews_k"]].head(3).to_string())
```

```text
            app_name  price_usd  is_free  reviews  reviews_k
0     Bright Quest X       0.99    False     2581        2.6
1      Quick Cleaner       0.00     True    65066       65.1
2  Pocket Radio Kids       0.00     True    54869       54.9
```

The new columns are computed for all 486 rows at once, element by element, just like NumPy arrays.

Because every step returns a new DataFrame, you can chain the whole question into one readable pipeline. Inside a chain, pass a `lambda` to `assign` so it sees the DataFrame *at that step* (`d` below), not the original variable:

```python
answer = (
    apps
    .assign(is_free=lambda d: d["price_usd"] == 0)
    .query("is_free and rating >= 4.8")
    .sort_values("reviews", ascending=False)
    .loc[:, ["app_name", "rating", "reviews"]]
    .head(5)
)
```

```text
              app_name  rating  reviews
107        Smart Notes     4.9   103411
138  Calm Cleaner Plus     5.0   102896
160          Pixel Fit     5.0    72984
485     Daily Camera X     5.0    34700
82       Mega Coach Go     5.0    30302
```

When you *do* want to change a DataFrame in place, assign with brackets or `.loc`, in **one** step:

```python
apps["is_free"] = apps["price_usd"] == 0                    # new column on apps itself
apps.loc[apps["rating"] >= 4.8, "badge"] = "Top rated"      # set only the matching rows; others get NaN
```

!!! warning "Common mistake: chained assignment does nothing"
    Writing the filter and the column as two separate `[]` steps looks reasonable but **never** changes `apps`:

    ```python
    apps[apps["rating"] >= 4.8]["badge"] = "Top rated"
    # ChainedAssignmentError: A value is being set on a copy of a DataFrame or Series
    # through chained assignment.
    # Such chained assignment never works to update the original DataFrame or Series,
    # because the intermediate object on which we are setting values always behaves
    # as a copy (due to Copy-on-Write).
    ```

    `apps[mask]` is a new DataFrame; the assignment lands on it and is thrown away. This is a warning, not an exception, so the program keeps running with unchanged data. pandas 3 enables **Copy-on-Write**: any result of indexing behaves as a copy. (In pandas 2 this same line sometimes worked and printed the infamous `SettingWithCopyWarning`; that warning is gone because the answer is now always "it doesn't work".) Use `apps.loc[mask, "badge"] = ...`.

## Counting with `value_counts`

`value_counts()` counts how often each distinct value appears, most frequent first:

```python
apps["has_ads"].value_counts()
# has_ads
# Yes    303
# No     183
# Name: count, dtype: int64

apps["has_ads"].value_counts(normalize=True)     # fractions instead of counts
# has_ads
# Yes    0.623457
# No     0.376543
# Name: proportion, dtype: float64
```

Missing values are skipped unless you pass `dropna=False`, which is how you'd see that the `badge` column we just made is mostly empty:

```python
apps["badge"].value_counts(dropna=False)
# badge
# NaN          437
# Top rated     49
# Name: count, dtype: int64
```

Now run it on `category`, and the mess from earlier is laid bare:

```python
apps["category"].nunique()          # 26   for what should be 8 categories
print(apps["category"].value_counts().head(12).to_string())
```

```text
category
Games               113
Tools                75
Education            68
Productivity         59
Photography          45
Health & Fitness     39
Music & Audio        33
Finance              28
 Games                5
productivity          3
Photography           2
Tools                 2
```

That second `Photography` and second `Tools` look identical to the first, but they're `'Photography '` and `'Tools '` with a **trailing** space you can't see in the printout (`.unique()` shows the quotes). `value_counts` is the first thing to run on any text column of a new dataset: it shows the typos before they skew your results. We'll fold these 26 spellings back into 8 on Day 17.

!!! warning "Common mistake: sorting numbers stored as text"
    `installs` looks numeric but is text (`"1,000,000+"`). Sort its counts by value and you get dictionary order, not numeric order:

    ```python
    apps["installs"].value_counts().sort_index().index.tolist()[:4]
    # ['1,000+', '1,000,000+', '10,000+', '10,000,000+']
    ```

    Text compares character by character, so `"1,000,000+"` comes before `"10,000+"`. Converting this column to integers is a Day 17 job.

!!! android "Android bridge"
    Today's lesson is a Room DAO query written as method calls. This DAO:

    ```kotlin
    @Query("""
        SELECT app_name, rating, reviews FROM apps
        WHERE price_usd = 0 AND rating >= :minRating
        ORDER BY reviews DESC LIMIT 5
    """)
    suspend fun topFree(minRating: Double): List<AppRow>
    ```

    is `apps.query("price_usd == 0 and rating >= @min_rating").sort_values("reviews", ascending=False).loc[:, ["app_name", "rating", "reviews"]].head(5)`. And in plain Kotlin over a `List<App>` it's `apps.filter { it.priceUsd == 0.0 && it.rating >= minRating }.sortedByDescending { it.reviews }.take(5)`: `filter` is the mask, `sortedByDescending` is `sort_values`, `take` is `head`, and `map { it.copy(...) }` is `assign`. Like Kotlin's collection functions, pandas returns new objects instead of mutating.

    Two differences bite. First, Kotlin also has `and` as an infix function on `Boolean`, and it's the *non*-short-circuiting one, much like pandas' `&`; pandas just spells it as a symbol. Second, missing values: in SQL, `NOT (rating >= 4)` is NULL for an unrated app, so `WHERE` drops the row; in pandas, `~(rating >= 4)` is `True` for `NaN`, so the row is **kept**. Same-looking query, different row count.

---

## Exercises

1. **Positions vs labels.** Show `app_name` and `size_mb` for the rows at positions 10 to 14 with `.iloc`. Then try `apps.loc[10:15, ["app_name", "size_mb"]]` and explain its shape. Why does `size_mb` show `NaN` for one row, and what dtype is the column?
2. **Ads in paid apps.** How many apps cost money? How many of those show ads? What fraction of *free* apps show ads? (One `value_counts(normalize=True)` call answers the last part.)
3. **Two ways, same answer.** Find apps rated between 3.0 and 3.5 inclusive with more than 10,000 reviews, sorted by reviews, most first. Write it once with `.between` and a mask and once with `.query`, and confirm the two results are identical with `.equals`.
4. **Fix the bug.** This line is meant to find free apps rated above 4 with no ads. Predict the error before running it, then fix it and report the row count.
   `apps[apps["rating"] > 4 & apps["has_ads"] == "No" and apps["price_usd"] == 0]`
5. **How much does the mess cost?** Count games and compute their mean rating with an exact `== "Games"` filter, then again after normalizing the text with `.str.strip().str.lower()`. How many games does the exact filter miss?
6. **Modern apps.** Use `assign` to add a boolean column `modern` that is `True` when `min_android >= 10`. Count `modern` with `value_counts`, then list the 3 lowest-rated modern apps with `nsmallest`.

??? success "Solutions"
    ```python
    import pandas as pd

    BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
    apps = pd.read_csv(BASE + "play_store_apps.csv")

    # 1
    print(apps.iloc[10:15][["app_name", "size_mb"]].to_string())
    #            app_name size_mb
    # 10      Pixel Coach    26.7
    # 11    Bright Camera    20.9
    # 12  Quick Camera Go     NaN
    # 13    Tiny Fit Plus    14.6
    # 14  Zen Budget Plus    25.2
    apps.loc[10:15, ["app_name", "size_mb"]].shape    # (6, 2)  labels 10..15, end INCLUDED
    apps["size_mb"].dtype                             # str   (text, not float)
    ```
    `.loc` includes label 15, so it returns 6 rows to `.iloc`'s 5. Row 12's size is missing from the file. The column is text (`str`), not numbers, because some rows say `Varies with device`, so pandas couldn't parse it as floats. Day 17 converts it.

    ```python
    # 2
    paid = apps[apps["price_usd"] > 0]
    len(paid)                                  # 53
    (paid["has_ads"] == "Yes").sum()           # 0     no paid app shows ads
    apps.loc[apps["price_usd"] == 0, "has_ads"].value_counts(normalize=True)
    # has_ads
    # Yes    0.699769
    # No     0.300231

    # 3
    a = (apps[apps["rating"].between(3.0, 3.5) & (apps["reviews"] > 10_000)]
         .sort_values("reviews", ascending=False))
    b = (apps.query("3.0 <= rating <= 3.5 and reviews > 10000")
         .sort_values("reviews", ascending=False))
    print(a[["app_name", "rating", "reviews"]].to_string())
    #           app_name  rating  reviews
    # 64    Pixel Camera     3.0    64988
    # 149  Spark Scanner     3.5    64328
    # 255    Pixel Radio     3.1    34854
    # 158   Pixel Budget     3.3    28622
    # 206  Pixel Notes X     3.5    23036
    # 371  Daily Notes X     3.3    21184
    # 173    Swift Learn     3.1    21024
    # 231      Daily Fit     3.4    14796
    # 45    Zen Quest Go     3.1    12167
    a.equals(b)                                # True

    # 4  The buggy line raises:
    # TypeError: unsupported operand type(s) for &: 'int' and 'StringArray'
    # (Python evaluates 4 & apps["has_ads"] first; the `and` would fail next.)
    fixed = apps[(apps["rating"] > 4) & (apps["has_ads"] == "No") & (apps["price_usd"] == 0)]
    fixed.shape                                # (71, 12)

    # 5
    exact = apps[apps["category"] == "Games"]
    loose = apps[apps["category"].str.strip().str.lower() == "games"]
    len(exact), len(loose)                     # (113, 120)   7 games missed
    round(exact["rating"].mean(), 3), round(loose["rating"].mean(), 3)   # (4.288, 4.276)

    # 6
    with_modern = apps.assign(modern=apps["min_android"] >= 10)
    with_modern["modern"].value_counts()
    # modern
    # True     318
    # False    168
    print(with_modern.query("modern").nsmallest(3, "rating")[["app_name", "rating", "min_android"]].to_string())
    #         app_name  rating  min_android
    # 270    Mega Lens     2.9         10.0
    # 173  Swift Learn     3.1         11.0
    # 255  Pixel Radio     3.1         11.0
    ```

## Resources

- [pandas: How do I select a subset of a DataFrame?](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html): the official getting-started tutorial for exactly today's topic.
- [pandas user guide: Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html): the full reference for `[]`, `.loc`, `.iloc`, boolean indexing, `.isin` and `.query`.
- [pandas user guide: Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html): why chained assignment never works in pandas 3.
- [pandas API: `DataFrame.sort_values`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html): every option, including `na_position` and `key`.
- [Python Data Science Handbook: Data Indexing and Selection](https://jakevdp.github.io/PythonDataScienceHandbook/03.02-data-indexing-and-selection.html): a free book chapter on `loc`/`iloc` with clear examples (written for older pandas, but the indexing rules are unchanged).
- [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas): a free short course with in-browser exercises on selecting, sorting and counting.

## Checklist

- [ ] I know `df["a"]` gives a Series and `df[["a", "b"]]` gives a DataFrame.
- [ ] I can explain the difference between `.loc` (labels, end included) and `.iloc` (positions, end excluded), and why they diverge after sorting.
- [ ] I write every condition in its own parentheses and join them with `&`, `|`, `~`, never `and`/`or`.
- [ ] I recognize "truth value of a Series is ambiguous" and the `rand_` `TypeError` and know the fix for each.
- [ ] I know `NaN` fails every comparison, so `~` keeps missing rows.
- [ ] I can use `.isin`, `.between`, `.query` (with `@variable`), `sort_values` with several keys, and `nlargest`.
- [ ] I add columns with `assign` or a single `.loc[mask, col] = value`, never chained `[][]` assignment.
- [ ] I run `value_counts` on text columns to spot messy spellings before trusting a filter.

**Next:** Day 17: Missing data and cleaning (coming soon)
