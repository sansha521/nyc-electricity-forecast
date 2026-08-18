# `visualize_data.ipynb` — Exploratory visualizations

Companion to [`all_data.md`](all_data.md). That file documents *what the columns
are*; this one documents *what the plots showed* and which modelling decision
each one settles.

All figures below are computed on the **2708 observed rows** (`is_imputed == 0`),
so the 52 gap-filled days can't prop up a relationship they were derived from.

| | |
|---|---|
| **Notebook** | `visualize_data.ipynb` |
| **Data** | `data_loading/all_data.csv` |
| **Target** | `value` — daily NYC (NYISO Zone J) electricity demand, MWh |

---

## The plots, and what each one decides

| # | Plot | Decides |
|---|---|---|
| 1 | Demand over time | Whether there's a regime break or trend to model |
| 2 | Demand vs temperature (scatter) | Whether the temperature response is linear |
| 3 | Mean demand by 5°F bin | Where the degree-day bases go |
| 4 | Year-over-year overlay | Whether seasonality is stable enough to lean on |
| 5 | Summer demand by weekday (box) | Whether calendar features earn their place |
| 6 | Autocorrelation | Whether lagged demand belongs in the model |
| 7 | Feature correlation heatmap | Which features are redundant with each other |
| 8 | Correlation by temperature regime | Which features matter *where* |

---

### 1. Demand over time

A single line, 2019-01-01 → 2026-07-22. Establishes the shape of the series:
strong annual cycle with summer peaks roughly 40% above the winter shoulder.

The finding that matters is the **level shift**, not the cycle:

| year | n | mean demand |
|---|---|---|
| 2019 | 365 | **142,449** |
| 2020 | 366 | **131,301** |
| 2021 | 365 | 133,776 |
| 2022 | 365 | 136,262 |
| 2023 | 365 | 132,263 |
| 2024 | 365 | 135,812 |
| 2025 | 318 | 138,299 |
| 2026 | 199 | 140,388 |

2020 falls ~8% below 2019 and the series has still not returned to 2019 levels
six years later. No column in the dataset expresses this, so a model will
distort its *weather* coefficients trying to explain it. See
[COVID and trend terms](#covid-and-trend-terms) below.

> **Caveat on 2025 and 2026.** Those row counts are not full years — 2026 stops
> at July 22 and has no autumn or winter, and 2025 is missing 47 observed days
> to the EIA outage. Both annual means are biased warm. Do not read a recovery
> trend off those two rows.

### 2. Demand vs temperature — the V

Scatter of `value` against `tavg`, one point per day. **This is the most
important plot in the notebook.** It shows the load/temperature response is
V-shaped: demand rises when it is cold (heating) and again when it is hot
(air conditioning), with a flat floor between.

The direct consequence: **a linear model given raw temperature cannot represent
this shape.** `tavg` correlates only +0.477 with demand because it averages two
opposing arms together. Splitting into one-sided `hdd`/`cdd` variables
linearises each arm separately and recovers the signal — `cdd` alone reaches
+0.848.

A second feature of the scatter is visible only as a shape: **the spread widens
sharply with temperature.** Around the 55°F trough the cloud is roughly 15k MWh
tall; at 80°F it spans about 160k–230k. The residual variance is not constant —
it scales with cooling load. Two consequences:

- OLS assumes homoscedastic errors, so it is misspecified here. The fit still
  works, but standard errors and prediction intervals will be too narrow in
  summer and too wide in winter.
- Absolute error metrics (MAE, RMSE) will be dominated by summer days regardless
  of model quality. Report MAPE alongside, or evaluate by regime.

The regime standard deviations quantify what the eye sees: 7,546 MWh in the mild
band against 24,945 in cooling.

### 3. Mean demand by 5°F temperature bin

The binned version of plot 2, which turns the scatter into something you can
read balance points off.

| `tavg` bin | n | mean `value` | change |
|---|---|---|---|
| (40, 45] | 249 | 126,149 | −3,961 |
| (45, 50] | 238 | 120,924 | −5,224 |
| (50, 55] | 251 | 116,312 | −4,612 |
| **(55, 60]** | 230 | **116,141** | −171 |
| (60, 65] | 221 | 120,268 | +4,127 |
| (65, 70] | 259 | 129,021 | +8,753 |
| (70, 75] | 295 | 147,009 | +17,987 |
| (75, 80] | 286 | 166,300 | +19,291 |

Demand bottoms out in the 55–60°F bin and climbs away in both directions. The
cooling arm is far steeper than the heating arm.

**This plot is what moved the HDD base from 65°F to 50°F.** A 65°F heating base
assigns positive `hdd` to all 719 days in the flat 50–65°F floor, asking the
model to fit a heating slope through a region with no heating response. Full
justification, including the base scan and holdout results, is in
[`all_data.md` → Degree days](all_data.md#degree-days).

> The extreme bins are nearly empty — 90–95°F has **n = 1**, 85–90°F has n = 23.
> The fitted response above ~88°F is essentially unconstrained, which is exactly
> where peak-day forecasting needs it most.

### 4. Year-over-year overlay

Each year drawn as its own line against day-of-year. Intended to test whether
the seasonal shape is stable enough that a model can learn it.

**As drawn, the plot does not answer that question.** Eight years of unsmoothed
daily lines overplot into a tangle, and with eight cycled default hues no
individual year can be traced through it. What the figure does establish is the
*seasonal envelope* shared by all years:

| day-of-year | period | demand band |
|---|---|---|
| 0–60 | Jan–Feb | 120–170k, winter shoulder |
| 90–140 | Apr–May | 100–125k, annual trough |
| 175–250 | Jun–Sep | 150–230k, summer peak |
| 340–365 | Dec | modest rise back to ~150k |

That envelope is consistent across years, which is the useful finding. But the
2020 level shift is **not** visible here — it is a ~8% difference against daily
swings several times larger, so it is buried. The evidence for the COVID shift
is the annual means in plot 1, not this figure.

To make the plot do its intended job, smooth before overlaying and drop the
categorical color problem:

```python
fig, ax = plt.subplots(figsize=(11, 5))
for yr, g in df_full.groupby(df_full["period"].dt.year):
    if len(g) < 300:                      # skip partial 2026
        continue
    ax.plot(g["period"].dt.dayofyear, g["value"].rolling(7, center=True).mean(),
            linewidth=1.5, alpha=0.85, label=yr)
ax.legend(frameon=False, ncol=4, fontsize=8)
```

A 7-day centred rolling mean removes the weekday cycle, which is most of the
visual noise, and leaves the year-to-year level differences legible. Seven
series is still at the edge of what categorical color can carry — highlighting
2019 and 2020 against the rest in gray would read better than seven competing
hues.

### 5. Summer demand by weekday

Boxplot of `value` by `day_name`, restricted to June–August (n = 696). The
seasonal restriction is the point: pooled across the year, weather variance
swamps the calendar effect.

| day | n | mean |
|---|---|---|
| Monday | 100 | 167,030 |
| Tuesday | 100 | 167,384 |
| Wednesday | 100 | **168,574** |
| Thursday | 99 | 167,918 |
| Friday | 98 | 165,362 |
| Saturday | 100 | 152,191 |
| Sunday | 99 | **150,421** |

Monday–Thursday are flat within ~1,500 MWh of each other; Friday is slightly
lower; the weekend drops off a cliff. **Summer weekday mean 167,260 vs weekend
151,310 — a 9.5% gap.**

So `is_weekend` carries nearly all of the day-of-week signal, and `day_of_week`
as a 7-level categorical adds little beyond it. Holidays are a separate,
smaller effect: restricted to weekdays, holidays average 131,171 MWh against
139,814 for ordinary weekdays, a **6.2% drop** (n = 78 holiday weekdays).

### 6. Autocorrelation

| lag | autocorrelation |
|---|---|
| 1 day | **0.903** |
| 2 days | 0.781 |
| 7 days | 0.745 |
| 14 days | 0.706 |
| 30 days | 0.397 |
| 365 days | 0.699 |

Yesterday's demand correlates with today's at 0.903 — higher than any weather
feature in the dataset. The lag-365 spike at 0.699 is the annual cycle.

**This does not automatically mean you should add lagged demand.** It depends
entirely on forecast horizon:

- **Day-ahead forecasting** — `value.shift(1)` is legitimate and will dominate
  every weather feature.
- **Seasonal / capacity planning** — you will not have recent actuals at
  prediction time, so a lag feature is **leakage** and will produce a validation
  score you cannot reproduce in use.

Decide the horizon before deciding on lags. `all_data.csv` deliberately ships
without them.

Two consequences of 0.903 regardless of horizon:

1. **Never use a random train/test split.** It puts July 15 in train and July 16
   in test; at this autocorrelation that leaks and your score becomes fiction.
   Split by time — train 2019–2024, test 2025+.
2. Residuals will be autocorrelated, so OLS standard errors are optimistic.

### 7. Feature correlation heatmap

Lower-triangle heatmap of the numeric columns against each other, diverging
colormap centred at zero. Read this one **feature-vs-feature**, not against
`value` — its job is finding redundancy.

Clusters at |r| ≥ 0.85:

| cluster | members |
|---|---|
| **Temperature** | `tmin` ~ `tmax` (0.956), `dew_point` (0.956), `tavg` (0.988), `tavg_3day` (0.964), `tavg_lag1` (0.942) |
| **Cooling** | `cdd` ~ `cdd_3day` (0.942) |
| **Heating** | `hdd` ~ `hdd_3day` (0.917) |
| **Solar** | `solar_rad` ~ `sunshine_hours` (0.882) |

Variance-inflation factors on the full 21-feature candidate set put `tmin`,
`tmax` and `tavg` at **infinity** — `tavg` is defined as `(tmin + tmax) / 2`, an
exact linear dependency, so including all three makes the design matrix
singular. `tavg_3day` (161), `dew_point` (100) and `tavg_lag1` (43) follow.

Take **one representative per cluster** for any linear or regularised model.
Tree models tolerate the collinearity but will split importance arbitrarily
across the duplicates, which makes importances unreadable.

---

## Feature correlation by temperature regime

Plots 2 and 3 established that demand responds to temperature in two opposite
directions. That breaks pooled correlation as a feature-ranking tool: a single
Pearson coefficient over all 2708 days averages the heating and cooling arms
together and reports the residue.

The fix is to compute correlations **within** regimes, split at the same
boundaries as the fitted degree-day bases:

| regime | definition | n | share | mean demand | sd |
|---|---|---|---|---|---|
| **Heating** | `tavg < 50` | 983 | 36% | 130,165 | 11,091 |
| **Mild** | `50 ≤ tavg ≤ 65` | 719 | 27% | 117,541 | 7,546 |
| **Cooling** | `tavg > 65` | 1006 | 37% | 154,984 | 24,945 |

Note the standard deviations: the cooling regime is **3.3× more variable** than
the mild regime. Most of the forecastable variance — and most of the error — lives
in summer.

### The table

Correlation with `value`, by regime. `—` marks a feature that is constant within
that regime, where correlation is undefined rather than zero.

| feature | heating | mild | cooling | **pooled** |
|---|---|---|---|---|
| `cdd` | — | — | **+0.863** | +0.848 |
| `cdd_3day` | −0.098 | +0.274 | **+0.886** | +0.849 |
| `hdd` | **+0.740** | — | — | **−0.007** |
| `hdd_3day` | **+0.777** | +0.150 | −0.063 | −0.026 |
| `tavg` | −0.740 | +0.203 | +0.863 | +0.477 |
| `tmin` | −0.705 | +0.248 | +0.863 | +0.493 |
| `tmax` | −0.704 | +0.101 | +0.740 | +0.453 |
| `dew_point` | −0.591 | +0.288 | +0.746 | +0.467 |
| `is_weekend` | −0.386 | **−0.633** | −0.276 | −0.236 |
| `daylight_hours` | −0.400 | −0.145 | +0.293 | +0.351 |
| `snwd` | +0.328 | +0.020 | — | +0.049 |
| `rh_max` | −0.232 | +0.155 | +0.149 | +0.098 |
| `snow` | +0.130 | — | — | +0.014 |
| `solar_rad` | −0.131 | −0.203 | +0.106 | +0.295 |
| `rh_mean` | −0.109 | +0.205 | +0.133 | +0.073 |
| `sunshine_hours` | +0.020 | −0.196 | +0.127 | +0.259 |
| `is_holiday` | −0.060 | −0.070 | −0.056 | −0.040 |
| `cloud_cover` | −0.061 | +0.115 | +0.020 | −0.035 |
| `prcp` | −0.049 | +0.093 | +0.152 | +0.060 |

Reproduce with:

```python
obs = df_full[df_full["is_imputed"] == 0]
regimes = {
    "heating": obs[obs["tavg"] < 50],
    "mild":    obs[(obs["tavg"] >= 50) & (obs["tavg"] <= 65)],
    "cooling": obs[obs["tavg"] > 65],
}
tbl = pd.DataFrame({k: g[feats].corrwith(g["value"]) for k, g in regimes.items()})
tbl["pooled"] = obs[feats].corrwith(obs["value"])
```

### What the split reveals

**`hdd` is the headline.** Pooled, it correlates **−0.007** with demand — dead
last, indistinguishable from noise. Within the heating regime it correlates
**+0.740**. The pooled figure is an artifact of a one-sided variable that is
zero on 64% of days: `hdd` is large in winter, demand peaks in summer, so a
single coefficient across all seasons cancels out. In forward selection
`hdd_3day` is the **second** feature chosen, cutting CV MAE from 10,212 to
7,936.

> Never drop a one-sided feature on its pooled correlation. This is the
> concrete case that proves the point.

**The mild regime belongs to the calendar, not the weather.** `is_weekend`
reaches **−0.633** there — its strongest showing anywhere, and stronger than any
weather feature in that band. With the temperature response flat and demand
varying by only 7,546 MWh sd, what's left is the commercial/residential weekday
pattern. In the cooling regime the same feature drops to −0.276, not because the
weekend effect vanishes but because cooling variance dwarfs it.

**Thermal inertia is real but strictly seasonal.** `cdd_3day` edges out
same-day `cdd` in the cooling regime (**+0.886 vs +0.863**) — day 3 of a heat
wave draws more than day 1 at the same temperature, because the building stock
is already heat-saturated. Outside summer it is noise (−0.098 heating).
`hdd_3day` likewise beats `hdd` on the heating side (+0.777 vs +0.740).

**`dew_point` is mostly temperature in disguise.** It correlates 0.956 with
`tmin` and its regime pattern tracks `tavg` almost exactly. Its +0.746 in the
cooling regime is real latent-heat load on air conditioning, but it is nearly
collinear with the temperature cluster — take it *or* a temperature feature, not
both, in a linear model.

**The solar group is weak and sign-unstable.** `solar_rad` is +0.295 pooled but
only **+0.106** within the cooling regime, and *negative* in heating (−0.131) and
mild (−0.203). Its pooled correlation is largely "it is summer", which `cdd`
already encodes. `sunshine_hours` behaves the same way and is additionally
quantized. `daylight_hours` flips sign between regimes (−0.400 heating, +0.293
cooling) — it is a pure function of date, so it is proxying season, not weather.

**Precipitation and cloud stay near-flat everywhere.** `prcp` peaks at +0.152,
`cloud_cover` at +0.115, `rh_mean` at +0.205. None survives as a standalone
feature. `snow` and `snwd` are structurally confined to the heating regime, where
`snwd` reaches +0.328 — likely confounded with cold rather than an independent
snow effect.

**`is_holiday` is uniformly weak by this metric** (≈ −0.06 in every regime) but
should be kept anyway: only 91 days are flagged, so correlation is the wrong
lens. The conditional comparison in plot 5 — a 6.2% drop against ordinary
weekdays — is the honest measure.

### Reading the `—` cells (and the blank bars on the chart)

Several cells are blank, and on the plotted version those bars are simply
missing — `cdd` has no heating or mild bar, `hdd` has no mild or cooling bar.
This is **not** missing data and **not** a zero relationship. In those regimes
the feature is a literal constant, and correlation with a constant is
mathematically undefined.

It follows directly from the degree-day definitions, which clip at their base:

```
hdd = max(0, 50 - tavg)
cdd = max(0, tavg - 65)
```

On a 40°F day, `cdd = max(0, 40 - 65) = 0`. On a 75°F day,
`hdd = max(0, 50 - 75) = 0`. Across the observed rows:

| regime | n | `cdd` distinct values / sd | `hdd` distinct values / sd |
|---|---|---|---|
| heating (`tavg < 50`) | 983 | **1 / 0.000** | 73 / 7.874 |
| mild (`50 ≤ tavg ≤ 65`) | 719 | **1 / 0.000** | **1 / 0.000** |
| cooling (`tavg > 65`) | 1006 | 49 / 5.405 | **1 / 0.000** |

Maximum `cdd` over all non-cooling days is 0.0; maximum `hdd` over all
non-heating days is 0.0. No exceptions.

Pearson correlation is

```
r = cov(x, y) / (sd_x * sd_y)
```

When `sd_x = 0` the denominator is zero, and the covariance of a constant with
anything is also zero — so the expression is `0/0`. NumPy raises
`RuntimeWarning: invalid value encountered in divide`, pandas stores `NaN`, and
matplotlib silently skips `NaN` bars.

**A blank is the honest rendering.** A bar at zero would assert "measured, no
relationship found," when the truth is that the feature is switched off and
there is nothing to measure.

Why the pattern is this clean: the regime boundaries are deliberately set to the
**same constants the degree-day features clip at**, 50 and 65. Split anywhere
else — say 55/70 — and `cdd` would be nonzero across part of the mild band,
producing a real-looking coefficient estimated from a handful of barely-nonzero
days. The alignment is what makes each one-sided feature either fully active or
fully off within a regime, never partially.

`snow` and `snwd` blank out in the cooling regime for the same structural
reason: no snow falls in a NYC summer.

> **Do not `.fillna(0)` before plotting.** It renders `hdd` as a zero-height bar
> in the mild and cooling regimes, visually identical to `is_holiday`'s genuine
> ≈ −0.06 — and `hdd` is precisely the feature whose importance the pooled
> correlation already hides. If the gap bothers you, annotate it instead:
>
> ```python
> for j, feat in enumerate(tbl.index):
>     for i, reg in enumerate(tbl.columns):
>         if pd.isna(tbl.loc[feat, reg]):
>             ax.text(0.01, j + (i - 1) * 0.25, "n/a — always 0 in this regime",
>                     va="center", fontsize=6, color="#999", style="italic")
> ```

---

## COVID and trend terms

Plots 1 and 4 both show the 2020 level shift, and nothing in the dataset encodes
it. Expanding-window CV on a fixed weather+calendar feature set:

| specification | CV MAPE |
|---|---|
| baseline (no regime terms) | 4.34% |
| + `covid` dummy (2020-03-15 → 2021-06-30) | 3.93% |
| + `covid` + `post` (after 2021-06-30) | **3.81%** |
| + `covid` + `post` + linear time trend | 4.95% |

Two dummies help materially. **A linear time trend makes things clearly worse** —
it extrapolates a slope out of sample that the data does not support. Model the
shift as a level change, not a trend.

---

## Known limitations of these plots

- **Pearson correlation only sees straight lines.** Every number in the regime
  table understates any curved relationship. Within regimes the arms are roughly
  linear, which is what makes the split work at all.
- **Correlation is not importance.** It is univariate — it cannot see that
  `cdd` and `cdd_3day` carry nearly the same information. Use it to find
  redundancy and check signs; use CV against a held-out time block to rank
  features for inclusion.
- **Regime boundaries are inherited, not re-fitted.** The 50 / 65 split comes
  from the degree-day base scan in `all_data.md`. Days near a boundary are
  assigned somewhat arbitrarily.
- **The mild regime is thin on weather variance by construction** — that is what
  makes it mild — so weather correlations there are measured over a narrow range
  and are less stable than the counts suggest.
- **52 imputed days are excluded throughout.** 50 of them fall in 2025–2026, so
  the most recent period is thinner than the row count implies.
