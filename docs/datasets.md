# Datasets

Course datasets live in the site, so every lesson works offline after one download. They are **synthetic**: generated with a fixed seed to look like real data, mess included, so the numbers in the lessons are reproducible. They don't describe real apps or companies.

| File | Rows × columns | Used in | Download |
|---|---|---|---|
| `play_store_apps.csv` | 486 × 12 | Week 3 (pandas, plotting) | [play_store_apps.csv](assets/data/play_store_apps.csv) |
| `developers.csv` | 40 × 4 | Week 3 (merging tables) | [developers.csv](assets/data/developers.csv) |

## `play_store_apps.csv`

One row per Android app in an imaginary Play Store category sample.

| Column | Meaning | Mess to watch for |
|---|---|---|
| `app_id` | Package name, e.g. `com.quick.cleaner` | 6 exact duplicate rows |
| `app_name` | Display name | |
| `category` | Store category | Inconsistent spelling: `games`, `GAMES`, `Games ` (trailing space) |
| `rating` | Average star rating, 1.0–5.0 | 18 missing (unrated apps) |
| `reviews` | Number of reviews | Very skewed: a few apps have most reviews |
| `installs` | Install bucket as text, e.g. `1,000,000+` | Text, not a number |
| `size_mb` | Download size in MB | 12 missing, and `Varies with device` in some rows |
| `price_usd` | Price, `0.0` for free apps | |
| `has_ads` | `Yes` / `No` | |
| `min_android` | Minimum Android version | Parsed as a float (`8.0`, `10.0`), not a version string |
| `last_updated` | Date as `YYYY-MM-DD` | Text until you parse it |
| `developer_id` | Joins to `developers.csv` | `D099` has no matching developer |

## `developers.csv`

| Column | Meaning |
|---|---|
| `developer_id` | `D001`–`D040` |
| `developer_name` | Studio name |
| `country` | Headquarters country |
| `founded_year` | Year founded |

## Loading

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")      # or a local path after downloading
devs = pd.read_csv(BASE + "developers.csv")
apps.shape, devs.shape    # ((486, 12), (40, 4))
```
