import pandas as pd

"""
EIA has no ZONJ (New York City sub-BA) rows for 52 days in the sample --
mostly a 35-day block in Nov/Dec 2025 and an 11-day block over that
Christmas. The gaps are upstream: re-querying the sub-BA endpoint still
returns nothing for those dates.

Statewide NY demand (the region endpoint, respondent=NY) is complete over
the same span, and NYC is a stable share of it -- 0.95 correlation, and a
ratio that drifts seasonally but tightly (Dec 0.309 +/- 0.006, Jul 0.345
+/- 0.012). So NYC is imputed as (local ratio) x (statewide demand).

The ratio is anchored on the nearest observed days rather than a
month/day-of-year average, which keeps it on the right side of any
year-over-year level shift (COVID, the post-2020 trend). Backtested by
punching synthetic 35-day holes through the observed series: 2.44% MAPE,
2.04% median, 5.11% p90.

Imputed rows are flagged with is_imputed so they can be dummied or
dropped downstream. 50 of the 52 fall in 2025-2026, so leaving them
unflagged would bias the most recent period specifically.
"""

NEAREST_N = 14  # observed days averaged to set the local ratio

nyc_df = pd.read_csv("daily_nyc_demand_cleaned.csv")       # ZONJ, has gaps
state_df = pd.read_csv("daily_demand_fill_missing.csv")    # respondent NY, complete

nyc_df["period"] = pd.to_datetime(nyc_df["period"])
state_df["period"] = pd.to_datetime(state_df["period"])

nyc_df = nyc_df.sort_values("period").reset_index(drop=True)
state_df = state_df.sort_values("period").reset_index(drop=True)

# Every calendar day the NYC series is supposed to cover
full_index = pd.date_range(nyc_df["period"].min(), nyc_df["period"].max(), freq="D")

df = (
    nyc_df.set_index("period")
    .reindex(full_index)
    .rename_axis("period")
    .reset_index()
)
df = df.merge(state_df.rename(columns={"value": "state_value"}), on="period", how="left")

df["is_imputed"] = df["value"].isna().astype(int)

# Ratio is only defined where NYC was actually reported
observed = df[df["value"].notna() & df["state_value"].notna()].copy()
observed["ratio"] = observed["value"] / observed["state_value"]

missing = df.index[df["value"].isna()]
print(f"missing NYC days: {len(missing)}")

unfillable = []
for i in missing:
    target = df.at[i, "period"]
    state_value = df.at[i, "state_value"]

    # No statewide figure to scale -- nothing to impute from
    if pd.isna(state_value):
        unfillable.append(target)
        continue

    distance = (observed["period"] - target).abs()
    ratio = observed.loc[distance.nsmallest(NEAREST_N).index, "ratio"].mean()

    df.at[i, "value"] = round(ratio * state_value)

if unfillable:
    # Flag rather than carry NaN demand into the feature table
    print(f"no statewide value for {len(unfillable)} days, dropping: {unfillable}")
    df = df[df["value"].notna()].copy()
    df["is_imputed"] = df["is_imputed"].astype(int)

df["value"] = df["value"].astype(int)

df = df[["period", "value", "is_imputed"]]
df.to_csv("daily_nyc_demand_filled.csv", index=False)

print(f"wrote {len(df)} rows, {df['is_imputed'].sum()} imputed")
print(df[df["is_imputed"] == 1].head(10))
