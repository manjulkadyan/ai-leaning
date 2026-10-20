# Day 18: groupby, merge and pivot

**Phase 1 · Week 3 · ~120 minutes**

## Objectives

- [ ] Explain split-apply-combine and predict what `df.groupby(key)[col].mean()` returns before you run it.
- [ ] Summarise groups with named aggregation (`.agg(n_apps=("app_id", "size"), ...)`) and tell `size` from `count`.
- [ ] Use `transform` to put a group statistic back on every row, and know when you need it instead of `agg`.
- [ ] Join two tables with `merge`, choose between `how="inner"` and `how="left"`, and find orphan rows with `indicator=True`.
- [ ] Guard a join with `validate="many_to_one"` so duplicate keys fail loudly instead of silently adding rows.
- [ ] Reshape with `pivot_table`, count with `crosstab`, and go back from wide to long with `melt`.

**Prerequisites.** [Day 15](day-15-pandas-dataframes.md) (DataFrames and Series), [Day 16](day-16-pandas-selecting.md) (selecting, masks, sorting) and [Day 17](day-17-pandas-cleaning.md) (cleaning, and the `clean_apps` function we start from). The code was tested on pandas 3.0.

---

## The big idea

So far every question has been about *rows*: which apps are rated above 4.5, which are free. The interesting questions are usually about *groups*: **which category is rated best? Which country's studios ship the most installs? Do apps with ads score lower?** Answering those needs two new moves:

1. **Group and summarise** (`groupby`): cut the table into piles by some key, reduce each pile to a few numbers, and glue the answers into a new, smaller table.
2. **Combine tables** (`merge`): the apps table only knows a `developer_id` like `D025`. The studio's name and country live in a second table, `developers.csv`. A merge looks up each app's developer and attaches those columns.

**Analogy.** You are sorting a sack of post at a sorting office. *Split*: drop every letter into the pigeonhole for its postcode. *Apply*: weigh each pigeonhole. *Combine*: write a one-page report, one line per postcode. That is `groupby`. A merge is the next desk along: someone takes each letter's postcode and looks it up in a directory to stamp the town name on it. If a postcode is not in the directory, you have to decide: throw the letter away (inner join), or keep it with "town unknown" (left join).

You already know both moves from SQL: `GROUP BY` and `JOIN`. pandas does the same things on in-memory tables, with NumPy doing the arithmetic underneath.

## Setup: start from clean data

Day 17 ended with a `clean_apps` function. Here is a short version of it so this lesson runs on its own. It drops the 6 duplicate rows (renumbering the rest 0 to 479), fixes the category spelling, turns installs into a number, turns `"Varies with device"` into a missing value so sizes become floats, and parses the dates. (Day 17's full version also turns `min_android` into a version string; we don't need that column today.)

```python
import numpy as np
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps_raw = pd.read_csv(BASE + "play_store_apps.csv")
devs = pd.read_csv(BASE + "developers.csv")

def clean_apps(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the Play Store apps table (short Day 17 version)."""
    out = df.drop_duplicates(ignore_index=True)
    return out.assign(
        category=out["category"].str.strip().str.title(),
        installs=(out["installs"]
                  .str.replace(",", "", regex=False)
                  .str.replace("+", "", regex=False)
                  .astype("int64")),
        size_mb=out["size_mb"].replace("Varies with device", np.nan).astype("float64"),
        last_updated=pd.to_datetime(out["last_updated"], format="%Y-%m-%d"),
    )

apps = clean_apps(apps_raw)
apps.shape, devs.shape     # ((480, 12), (40, 4))
apps[["category", "rating", "installs", "size_mb", "last_updated"]].dtypes
# category                   str
# rating                 float64
# installs                 int64
# size_mb                float64
# last_updated    datetime64[us]
```

No internet? Download both files from the [Datasets page](../../datasets.md) and pass the local paths to `read_csv` instead. After cleaning there are exactly 8 categories: Games, Tools, Education, Productivity, Photography, Health & Fitness, Music & Audio and Finance.

## Split, apply, combine

Start tiny so you can check every number by hand. Take the first 8 apps that are Games, Tools or Finance:

```python
small = apps.loc[apps["category"].isin(["Games", "Tools", "Finance"]),
                 ["app_name", "category", "rating"]].head(8)
print(small.to_string())
#              app_name category  rating
# 2   Pocket Radio Kids    Tools     3.7
# 3     Lumen Learn Pro    Games     4.0
# 5    Orbit Tasks Kids  Finance     4.8
# 7      Spark Camera X    Tools     NaN
# 8    Spark Camera Pro    Tools     4.4
# 12    Quick Camera Go    Games     3.7
# 16    Mega Beats Lite  Finance     4.0
# 19     Pixel Budget X    Tools     4.6

small.groupby("category")["rating"].mean()
# category
# Finance    4.400000
# Games      3.850000
# Tools      4.233333
# Name: rating, dtype: float64
```

Read the call left to right, it is the whole algorithm:

- `small.groupby("category")`: **split**. Rows with the same category go into the same group.
- `["rating"]`: of each group, we only care about the rating column.
- `.mean()`: **apply** a reduction to each group, then **combine** the answers into a Series whose index is the group keys (sorted alphabetically by default).

<figure class="diagram" markdown>
![Animation of split-apply-combine on 8 rows: the rows, coloured by category, slide into three groups (Finance with 4.8 and 4.0, Games with 4.0 and 3.7, Tools with 3.7, NaN, 4.4 and 4.6); each group is reduced to its mean, (4.8 + 4.0) / 2 = 4.40, (4.0 + 3.7) / 2 = 3.85 and (3.7 + 4.4 + 4.6) / 3 = 4.23 with the NaN skipped; the three means fly into a result table indexed by category](../../assets/img/phase-1/day18-split-apply-combine.svg)
<figcaption>Animated: groupby splits rows into one group per key, reduces each group to one number, and combines the answers into a new table.</figcaption>
</figure>

### The math: a mean per group

Let \( g \) be one group, say Tools. Write \( r_i \) for the rating in row \( i \), and \( i \in g \) for "row \( i \) belongs to group \( g \)". Let \( n_g \) be the number of rows in \( g \) that **have** a rating. Then the group mean is

\[
\bar{r}_g = \frac{1}{n_g} \sum_{i \in g} r_i
\]

Read it as: add up the ratings of the rows in this group, divide by how many there were. The bar over \( r \) is the usual symbol for "average".

**Worked example.** Tools has 4 rows, but one rating is missing, so pandas skips it and \( n_g = 3 \):

\[
\bar{r}_{\text{Tools}} = \frac{3.7 + 4.4 + 4.6}{3} = \frac{12.7}{3} = 4.2\overline{3}
\]

That is exactly the `4.233333` pandas printed. Finance is \( (4.8 + 4.0)/2 = 4.40 \) and Games is \( (4.0 + 3.7)/2 = 3.85 \).

### The GroupBy object is lazy

`groupby` alone computes nothing; it returns a plan. You can inspect it and loop over it, which is a good way to see the split:

```python
g = small.groupby("category")
g.ngroups                      # 3
for name, group in g:
    print(name, group["rating"].tolist())
# Finance [4.8, 4.0]
# Games [4.0, 3.7]
# Tools [3.7, nan, 4.4, 4.6]

print(g.get_group("Tools").to_string())   # one group is a normal DataFrame
#              app_name category  rating
# 2   Pocket Radio Kids    Tools     3.7
# 7      Spark Camera X    Tools     NaN
# 8    Spark Camera Pro    Tools     4.4
# 19     Pixel Budget X    Tools     4.6
```

Notice the groups keep the original row labels (2, 7, 8, 19). That will matter for `transform`. In real code you rarely loop: calling a reduction on the GroupBy runs it for every group in fast compiled code.

### `size` vs `count`

```python
small.groupby("category")["rating"].agg(["size", "count", "mean"])
#           size  count      mean
# category
# Finance      2      2  4.400000
# Games        2      2  3.850000
# Tools        4      3  4.233333
```

`size` counts **rows**. `count` counts **non-missing values**. Tools has 4 apps but only 3 ratings.

!!! warning "Common mistake: using `count` to count rows"
    "How many apps per category?" is `size`, not `count`. With `count`, every missing rating quietly shrinks your app count, and the number changes depending on which column you happen to pick. Use `.size()` on the GroupBy (or `("any_column", "size")` in `agg`) for row counts, and `count` only when you mean "how many have a value".

## Named aggregation: a summary table in one call

Now the real data. One row per category, with several statistics, each with a name you choose:

```python
summary = (
    apps.groupby("category")
        .agg(
            n_apps=("app_id", "size"),
            avg_rating=("rating", "mean"),
            total_installs=("installs", "sum"),
            median_size=("size_mb", "median"),
        )
        .sort_values("total_installs", ascending=False)
)
print(summary.round(2).to_string())
#                   n_apps  avg_rating  total_installs  median_size
# category
# Games                118        4.28       148011000        78.50
# Education             68        4.04        19069000        26.70
# Music & Audio         35        4.22        12990000        26.65
# Productivity          64        4.13        10670000        26.70
# Tools                 77        4.15         9057000        24.30
# Photography           48        4.18         7671000        24.85
# Finance               30        4.11         6185000        20.70
# Health & Fitness      40        4.08         3810000        21.60
```

Each keyword argument reads `new_column=(source_column, function)`. The function can be a name pandas knows (`"mean"`, `"median"`, `"sum"`, `"min"`, `"max"`, `"size"`, `"count"`, `"nunique"`, `"first"`) or your own function. Named aggregation is the form to reach for: the output columns have clean names, and you can read the line a month later.

Two things to see in this table. Games has 118 apps and almost 150 million installs, most of it from one 100,000,000+ app: sums of skewed data are dominated by a few rows (Day 19 will plot this). And the result is just a DataFrame indexed by `category`, so everything from Day 16 (`.loc`, `sort_values`, masks) works on it.

### Grouping by two keys

Pass a list to group by combinations. The result has one row per (category, has_ads) pair, and its index has two levels, a **MultiIndex**:

```python
two = apps.groupby(["category", "has_ads"])["rating"].mean().round(2)
two.head(6)
# category   has_ads
# Education  No         4.14
#            Yes        4.00
# Finance    No         4.05
#            Yes        4.14
# Games      No         4.28
#            Yes        4.29
# Name: rating, dtype: float64
two.loc[("Games", "Yes")]          # 4.29   look up with a tuple

apps.groupby(["category", "has_ads"], as_index=False)["rating"].mean().round(2).head(4)
#     category has_ads  rating
# 0  Education      No    4.14
# 1  Education     Yes    4.00
# 2    Finance      No    4.05
# 3    Finance     Yes    4.14
```

`as_index=False` (or `.reset_index()` afterwards) keeps the keys as ordinary columns, which is often easier to work with and is what you want before saving to CSV.

## `transform`: a group statistic on every row

`agg` shrinks: 480 rows in, 8 rows out. Sometimes you want the group statistic **next to each original row**, for example "how does this app's rating compare with its category's average?" That is `transform`: it computes per group, then broadcasts the answer back to every row of that group, so the output has the same length and the same index as the input.

```python
avg = apps.groupby("category")["rating"].mean()
len(avg), len(apps.groupby("category")["rating"].transform("mean"))   # (8, 480)

apps2 = apps.assign(cat_avg=apps.groupby("category")["rating"].transform("mean"))
apps2 = apps2.assign(vs_cat=apps2["rating"] - apps2["cat_avg"])
print(apps2[["app_name", "category", "rating", "cat_avg", "vs_cat"]].head(4).round(2).to_string())
#             app_name      category  rating  cat_avg  vs_cat
# 0     Bright Quest X  Productivity     3.8     4.13   -0.33
# 1      Quick Cleaner  Productivity     3.9     4.13   -0.23
# 2  Pocket Radio Kids         Tools     3.7     4.15   -0.45
# 3    Lumen Learn Pro         Games     4.0     4.28   -0.28
```

In symbols, row \( i \) in group \( g \) gets \( r_i - \bar{r}_g \). Pocket Radio Kids: \( 3.7 - 4.15 = -0.45 \). Subtracting a group mean like this is called **centering within groups**, and it is a feature-engineering move you will use in Phase 2.

The three verbs side by side:

| Verb | Rows out | Use it for |
|---|---|---|
| `agg` | one per group | summary tables ("average rating per category") |
| `transform` | same as input | per-row features ("rating minus category average", "share of category installs") |
| `filter` | the rows of the groups that pass | keeping whole groups, e.g. `apps.groupby("category").filter(lambda grp: len(grp) >= 60)` keeps the 327 rows of the 4 biggest categories |

!!! warning "Common mistake: assigning an `agg` result back as a column"
    `apps.assign(cat_avg=avg)` looks reasonable but gives a column that is **entirely NaN** (480 missing values). pandas lines Series up by index label: `avg` is indexed by category names, `apps` by row numbers 0 to 479, and no labels match. Use `transform("mean")` when you want one value per row. (Alternatively, merge the summary table back on `category`, which is the next section.)

## Merge: joining two tables on a key

`apps` has a `developer_id` column, and `developers.csv` has one row per developer with a name, country and founding year. To ask "which country's studios get the most installs?" we need both in one table. Start with 4 real apps and 3 real developers:

```python
few_apps = apps.loc[[2, 3, 5, 229], ["app_name", "developer_id"]]
few_devs = devs.loc[devs["developer_id"].isin(["D009", "D025", "D040"]),
                    ["developer_id", "developer_name", "country"]]

few_apps.merge(few_devs, on="developer_id", how="inner")
#             app_name developer_id developer_name country
# 0  Pocket Radio Kids         D025     South Apps  Canada
# 1    Lumen Learn Pro         D009     Pine Works   Japan
# 2   Orbit Tasks Kids         D040    Blue Studio     USA

few_apps.merge(few_devs, on="developer_id", how="left")
#             app_name developer_id developer_name country
# 0  Pocket Radio Kids         D025     South Apps  Canada
# 1    Lumen Learn Pro         D009     Pine Works   Japan
# 2   Orbit Tasks Kids         D040    Blue Studio     USA
# 3      Spark Quest X         D099            NaN     NaN
```

<figure class="diagram" markdown>
![Animation of a join: four apps on the left with developer_id D025, D009, D040 and D099; three developers on the right with D009, D025 and D040. Orange lines connect matching ids; D099 gets a red dashed line labelled no match. The inner-join result has 3 rows and shows Spark Quest X struck out as dropped; the left-join result has 4 rows with NaN developer_name and country for Spark Quest X](../../assets/img/phase-1/day18-join-inner-left.svg)
<figcaption>Animated: an inner join keeps only keys found in both tables; a left join keeps every left row and fills the missing side with NaN.</figcaption>
</figure>

For each left row, `merge` finds the right rows with the same key and glues their columns on. The `how` argument decides what happens to rows with no partner:

| `how=` | Keeps | SQL |
|---|---|---|
| `"inner"` (the default) | only keys present in **both** tables | `INNER JOIN` |
| `"left"` | every row of the left table; NaN where the right side is missing | `LEFT JOIN` |
| `"right"` | every row of the right table | `RIGHT JOIN` |
| `"outer"` | every row of both | `FULL OUTER JOIN` |

`D099` is in the apps table but not in `developers.csv`, so an inner join silently **drops** Spark Quest X. That is the trap: the default `how` loses data without a warning. For "attach extra info to my main table", `how="left"` is almost always the right choice, because the main table keeps every row.

If the key columns have different names, use `left_on="developer_id", right_on="dev"` instead of `on=`.

### The full join, with two safety checks

```python
full = apps.merge(devs, on="developer_id", how="left",
                  validate="many_to_one", indicator=True)
full.shape                       # (480, 16)   12 + 3 new columns + _merge
full["_merge"].value_counts()
# _merge
# both          478
# left_only       2
# right_only      0
# Name: count, dtype: int64

print(full.loc[full["_merge"] == "left_only",
               ["app_name", "developer_id", "developer_name", "country"]].to_string())
#            app_name developer_id developer_name country
# 229   Spark Quest X         D099            NaN     NaN
# 357  Quick Fit Lite         D099            NaN     NaN

apps.merge(devs, on="developer_id", how="inner").shape    # (478, 15)
```

- `indicator=True` adds a `_merge` column saying where each row came from: `both`, `left_only` (no developer found) or `right_only` (only possible with right/outer joins). It is the quickest way to find **orphans**. Here, 2 apps point at the missing developer D099.
- `validate="many_to_one"` tells pandas what shape you expect: many apps per developer, but **at most one row per developer** on the right. If that is not true, the merge stops with an error instead of producing a wrong table.

Why that matters: suppose someone appends a second row for D001 to the developers file.

```python
devs_bad = pd.concat([devs, devs.iloc[[0]].assign(developer_name="Pine Studio Ltd")],
                     ignore_index=True)

apps.merge(devs_bad, on="developer_id", how="left").shape     # (489, 15)   9 extra rows!
apps.merge(devs_bad, on="developer_id", how="left", validate="many_to_one")
# pandas.errors.MergeError: Merge keys are not unique in right dataset; not a many-to-one merge
#
# Duplicates in right:
#  developer_id
#         D001 ...
```

D001 has 9 apps. Each of them now matches 2 developer rows, so each appears twice, and the table grows from 480 to 489 rows. Every sum and count you compute after that is wrong, and nothing told you. With `validate`, it fails loudly at the line that caused it.

!!! warning "Common mistake: a merge that changes the row count"
    After a left join on a lookup table, the row count should be **exactly** the left table's row count. If it grew, the right table has duplicate keys (a many-to-many join). Make it a habit: `validate="many_to_one"` on every lookup join, or at least `assert len(merged) == len(left)`. And check `devs["developer_id"].is_unique` (here `True`) before you trust a key.

### Merge, then group

Now the question we came for: which countries' studios ship the most?

```python
by_country = (
    full.groupby("country", dropna=False)
        .agg(n_apps=("app_id", "size"),
             avg_rating=("rating", "mean"),
             total_installs=("installs", "sum"))
        .sort_values("n_apps", ascending=False)
)
print(by_country.round(2).to_string())
#              n_apps  avg_rating  total_installs
# country
# USA             165        4.15       140245000
# India           100        4.15         8767000
# Japan            53        4.26         7338000
# Germany          41        4.22        24975000
# Brazil           38        4.06         3042000
# South Korea      36        4.26        27500000
# Canada           23        4.16         2908000
# UK               22        4.11         2488000
# NaN               2        3.70          200000
```

The `NaN` row is the two D099 apps, which have no country. By default `groupby` **drops** rows whose key is missing (`full.groupby("country").size().sum()` is 478, not 480); `dropna=False` keeps them as their own group, so you can see them instead of silently losing them.

!!! android "Android bridge: `GROUP BY` and `JOIN` in Room"
    You have written this exact logic in SQL inside Room `@Query` annotations. The mapping is almost one to one:

    ```kotlin
    data class CategoryStats(
        val category: String,
        @ColumnInfo(name = "n_apps") val nApps: Int,
        @ColumnInfo(name = "avg_rating") val avgRating: Double?,
    )

    @Dao
    interface AppDao {
        // apps.groupby("category").agg(n_apps=("app_id", "size"), avg_rating=("rating", "mean"))
        @Query("SELECT category, COUNT(*) AS n_apps, AVG(rating) AS avg_rating FROM apps GROUP BY category")
        suspend fun categoryStats(): List<CategoryStats>

        // apps.merge(devs, on="developer_id", how="left")
        @Query("SELECT apps.*, developers.country FROM apps LEFT JOIN developers ON apps.developer_id = developers.developer_id")
        suspend fun appsWithCountry(): List<AppWithCountry>   // @Embedded val app: App, val country: String?
    }
    ```

    - `COUNT(*)` counts rows like `size`; `COUNT(rating)` skips NULLs like `count`; `AVG(rating)` ignores NULLs just as pandas' `mean` ignores NaN. The `size`/`count` trap exists in SQL too.
    - A `LEFT JOIN` fills missing right-side columns with NULL, which is why `country` must be nullable in Kotlin, the same way pandas fills NaN.
    - `HAVING COUNT(*) >= 60` is `groupby(...).filter(lambda grp: len(grp) >= 60)` (strictly, HAVING returns the filtered groups' summary rows, while pandas' `filter` returns their original rows).
    - `transform` corresponds to a SQL window function, `AVG(rating) OVER (PARTITION BY category)`. SQLite only gained window functions in version 3.25, so on older Android versions' bundled SQLite you cannot use them; with pandas you can.
    - Room's `@Relation` and multimap return types (for example `Map<Developer, List<App>>`, Room 2.4+) are the "one developer, many apps" relationship that `validate="many_to_one"` checks from the apps side.

## Reshaping: `pivot_table`, `crosstab`, `melt`

The two-key groupby above answered "average rating per (category, has_ads)", but as a long, thin Series. Humans read a **grid** more easily: categories down the side, Yes/No across the top. `pivot_table` is groupby plus that reshape in one call:

```python
pt = apps.pivot_table(index="category", columns="has_ads",
                      values="rating", aggfunc="mean")
print(pt.round(2).to_string())
# has_ads             No   Yes
# category
# Education         4.14  4.00
# Finance           4.05  4.14
# Games             4.28  4.29
# Health & Fitness  3.98  4.16
# Music & Audio     4.22  4.22
# Photography       4.13  4.22
# Productivity      3.99  4.22
# Tools             4.19  4.12
```

- `index`: what goes down the side (one row per value).
- `columns`: whose values become the column headers.
- `values`: the column being summarised; `aggfunc`: how (the default is `"mean"`).

It is exactly `apps.groupby(["category", "has_ads"])["rating"].mean().unstack()`: `unstack` moves the inner index level up into the columns.

<figure class="diagram" markdown>
![Animation of pivot_table: a long table with six rows (Finance No 4.05, Finance Yes 4.14, Games No 4.28, Games Yes 4.29, Tools No 4.19, Tools Yes 4.12) is read one row at a time; an arrow carries each rating into a grid with categories Finance, Games and Tools as rows and No and Yes as columns](../../assets/img/phase-1/day18-pivot-long-wide.svg)
<figcaption>Animated: each (category, has_ads) row of the long table lands in one cell of the wide grid; melt reverses the move.</figcaption>
</figure>

Add `margins=True` to get an "All" row and column, computed from the raw rows (not by averaging the cells):

```python
apps.pivot_table(index="category", columns="has_ads", values="rating",
                 aggfunc="mean", margins=True).round(2).tail(3)
# has_ads         No   Yes   All
# category
# Productivity  3.99  4.22  4.13
# Tools         4.19  4.12  4.15
# All           4.14  4.18  4.17
```

**`crosstab`** is the special case "count rows for each combination", a two-dimensional `value_counts`:

```python
pd.crosstab(apps["category"], apps["has_ads"])
# has_ads           No  Yes
# category
# Education         21   47
# Finance           12   18
# Games             41   77
# Health & Fitness  18   22
# Music & Audio     11   24
# Photography       22   26
# Productivity      22   42
# Tools             32   45

pd.crosstab(apps["category"], apps["has_ads"], normalize="index").round(2).head(3)
# has_ads      No   Yes
# category
# Education  0.31  0.69
# Finance    0.40  0.60
# Games      0.35  0.65
```

`normalize="index"` turns each row into fractions that sum to 1: 69% of Education apps show ads.

**`melt`** goes the other way, wide back to long. Long ("tidy") data, one row per observation, is what `groupby`, plotting libraries and ML code want; wide data is what people want to read:

```python
wide = pt.round(2).reset_index()          # category, No, Yes as plain columns
long = wide.melt(id_vars="category", var_name="has_ads", value_name="avg_rating")
long.shape                                # (16, 3)   8 categories x 2
long.head(4)
#            category has_ads  avg_rating
# 0         Education      No        4.14
# 1           Finance      No        4.05
# 2             Games      No        4.28
# 3  Health & Fitness      No        3.98
```

`id_vars` are the columns to keep as they are; every other column is unpivoted into a (name, value) pair. (There is also `df.pivot(...)`, which reshapes without aggregating and raises an error if any (index, column) pair appears twice; `pivot_table` is the forgiving one.)

!!! warning "Common mistake: reading a pivot table as a fact about causes"
    Productivity apps *with* ads average 4.22 and those without 3.99. That does not mean adding ads raises ratings: the groups differ in many other ways (price, size, developer). Group summaries describe the data; they don't explain it. Keep that in mind when the same tables become ML features in Phase 2.

---

## Exercises

1. **Best seller per category.** For each category, find the single app with the most reviews: show `category`, `app_name` and `reviews`. Hint: `groupby(...)["reviews"].idxmax()` gives row labels you can pass to `.loc`.
2. **Ads vs no ads.** Group by `has_ads` and use named aggregation to get `n_apps`, `avg_rating`, `median_installs` and `avg_price`. What do you notice about the price of apps with ads?
3. **Share of the category.** Add a column `install_share` = each app's installs divided by its category's total installs (use `transform`). Check that the shares add up to 1 within every category, then show the 3 apps with the largest share.
4. **Top studios.** Left-join `apps` with `devs` (with `validate="many_to_one"`), then list the 5 developers with the most total installs, showing their name, country, number of apps and total installs. Finally, use an outer join with `indicator=True` to list every row that did not match on both sides.
5. **Country grid.** Using the joined table, build a pivot table of mean rating with `country` down the side and `has_ads` across the top, with margins.
6. **Back to long.** Take `pd.crosstab(apps["category"], apps["has_ads"])`, turn it into a long table with columns `category`, `has_ads`, `n_apps` using `melt`, and sort it so the biggest combinations come first. Check that `n_apps` sums to 480.

??? success "Solutions"
    ```python
    # 1
    top = apps.loc[apps.groupby("category")["reviews"].idxmax(),
                   ["category", "app_name", "reviews"]]
    print(top.to_string(index=False))
    #         category         app_name  reviews
    #        Education Daily Learn Lite   690677
    #          Finance       Zen Budget    71758
    #            Games  Orbit Lens Lite  2525379
    # Health & Fitness    Smart Learn X    54862
    #    Music & Audio     Quick Budget   214761
    #      Photography    Echo Quest Go   109220
    #     Productivity    Lumen Scanner   112885
    #            Tools     Orbit Budget    84585

    # 2
    print(apps.groupby("has_ads").agg(
        n_apps=("app_id", "size"),
        avg_rating=("rating", "mean"),
        median_installs=("installs", "median"),
        avg_price=("price_usd", "mean"),
    ).round(2).to_string())
    #          n_apps  avg_rating  median_installs  avg_price
    # has_ads
    # No          179        4.14          50000.0       1.32
    # Yes         301        4.18          10000.0       0.00
    # Every app with ads is free in this dataset: ads and a price tag don't mix here.

    # 3
    apps3 = apps.assign(
        install_share=apps["installs"] / apps.groupby("category")["installs"].transform("sum")
    )
    apps3.groupby("category")["install_share"].sum().round(6)   # 1.0 for all 8 categories
    print(apps3.nlargest(3, "install_share")[["app_name", "category", "installs", "install_share"]]
          .round(3).to_string())
    #              app_name       category   installs  install_share
    # 169      Quick Budget  Music & Audio   10000000          0.770
    # 396   Orbit Lens Lite          Games  100000000          0.676
    # 124  Daily Learn Lite      Education   10000000          0.524

    # 4
    full = apps.merge(devs, on="developer_id", how="left", validate="many_to_one")
    top_devs = (
        full.groupby(["developer_id", "developer_name", "country"])
            .agg(n_apps=("app_id", "size"), total_installs=("installs", "sum"))
            .sort_values("total_installs", ascending=False)
            .head(5)
    )
    print(top_devs.to_string())
    #                                          n_apps  total_installs
    # developer_id developer_name country
    # D002         Blue Works     USA               9       103032000
    # D034         Bit Works      South Korea      10        12222000
    # D006         Pine Soft      USA              13        12130000
    # D026         Stone Apps     South Korea       9        11316000
    # D031         Stone Studio   Germany          10        10773000

    check = apps.merge(devs, on="developer_id", how="outer", indicator=True)
    check["_merge"].value_counts()      # both 478, left_only 2, right_only 0
    print(check.loc[check["_merge"] != "both",
                    ["app_name", "developer_id", "developer_name"]].to_string())
    #            app_name developer_id developer_name
    # 478   Spark Quest X         D099            NaN
    # 479  Quick Fit Lite         D099            NaN
    # right_only is 0: every developer has at least one app. (An outer join
    # re-sorts by key and renumbers rows, so D099 ends up at the bottom.)

    # 5
    print(full.pivot_table(index="country", columns="has_ads", values="rating",
                           aggfunc="mean", margins=True).round(2).to_string())
    # has_ads        No   Yes   All
    # country
    # Brazil       3.95  4.10  4.06
    # Canada       4.22  4.10  4.16
    # Germany      4.21  4.22  4.22
    # India        4.06  4.20  4.15
    # Japan        4.22  4.29  4.26
    # South Korea  4.46  4.14  4.26
    # UK           4.31  4.02  4.11
    # USA          4.09  4.18  4.15
    # All          4.15  4.18  4.17

    # 6
    ct = pd.crosstab(apps["category"], apps["has_ads"])
    long = (ct.reset_index()
              .melt(id_vars="category", value_name="n_apps")
              .sort_values("n_apps", ascending=False))
    print(long.head(4).to_string(index=False))
    #     category has_ads  n_apps
    #        Games     Yes      77
    #    Education     Yes      47
    #        Tools     Yes      45
    # Productivity     Yes      42
    long["n_apps"].sum()      # 480
    ```

    In 6, `melt` names the new column `has_ads` without being told: it uses the name of the crosstab's column axis.

## Resources

- [pandas user guide: Group by (split-apply-combine)](https://pandas.pydata.org/docs/user_guide/groupby.html): the full story of `agg`, `transform`, `filter` and named aggregation.
- [pandas user guide: Merge, join, concatenate and compare](https://pandas.pydata.org/docs/user_guide/merging.html): every `how`, `validate` and `indicator` option, with pictures.
- [pandas user guide: Reshaping and pivot tables](https://pandas.pydata.org/docs/user_guide/reshaping.html): `pivot`, `pivot_table`, `stack`/`unstack`, `melt`, `crosstab`.
- [pandas: Comparison with SQL](https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html): your SQL, translated line by line into pandas. Ideal for an Android developer who knows Room.
- [Wes McKinney, *Python for Data Analysis*, 3rd ed. (free online)](https://wesmckinney.com/book/): chapter 10, "Data Aggregation and Group Operations", by the creator of pandas.
- [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas): short free course with in-browser exercises; do the "Grouping and Sorting" lesson.
- [Hadley Wickham, "The Split-Apply-Combine Strategy for Data Analysis"](https://www.jstatsoft.org/article/view/v040i01): the 2011 paper that named the pattern.

## Checklist

- [ ] I can say what `groupby(key)[col].mean()` returns (a Series indexed by the keys) and checked the Tools mean 4.233 by hand.
- [ ] I know `size` counts rows and `count` counts non-missing values.
- [ ] I built a named-aggregation summary table per category.
- [ ] I used `transform` to put a category average next to every app, and I know why assigning an `agg` result back gives NaN.
- [ ] I can explain inner vs left join using the D099 orphan, and I found it with `indicator=True`.
- [ ] I use `validate="many_to_one"` on lookup joins and know a growing row count means duplicate keys.
- [ ] I built a `pivot_table` with margins, a `crosstab`, and melted a wide table back to long.

**Next:** Day 19: Plotting with Matplotlib (coming soon)
