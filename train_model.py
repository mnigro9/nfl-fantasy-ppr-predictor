"""
train_model.py

Pulls historical NFL weekly stats, engineers rolling-average features,
trains a RandomForestRegressor to predict a player's NEXT game's PPR
fantasy points, and saves the model + a "latest features" table that
the Streamlit app will use for live predictions.

Run this once locally:
    python train_model.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import nflreadpy as nfl

# Seasons to train on. Feel free to widen this range.
# Only include a season once Week 1 of it has actually been played.
YEARS = [2022, 2023, 2024, 2025, 2026]

# Stats we'll turn into rolling 3-game averages
RAW_FEATURES = [
    "fantasy_points_ppr",
    "targets",
    "carries",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
]


def load_data() -> pd.DataFrame:
    print(f"Downloading weekly data for {YEARS}...")
    # nflreadpy returns a Polars DataFrame by default; .to_pandas() converts
    # it so the rest of this script (written with pandas) works unchanged.
    df = nfl.load_player_stats(YEARS).to_pandas()
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]
    df = df[df["fantasy_points_ppr"].notna()]
    return df


def build_features(df: pd.DataFrame):
    df = df.sort_values(["player_id", "season", "week"]).copy()
    feature_cols = [c for c in RAW_FEATURES if c in df.columns]

    # Rolling 3-game average, using only PAST games (shift(1) before rolling)
    for col in feature_cols:
        df[f"{col}_roll3"] = (
            df.groupby("player_id")[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )

    roll_cols = [f"{c}_roll3" for c in feature_cols]

    # Rookies / players in their first career game have no prior game to
    # roll from, so their roll3 features are NaN. Rather than dropping them
    # (which means the app can never predict for a rookie), fall back to the
    # league average for that position in that season/week. This is a crude
    # "replacement level" estimate -- not as good as using real history, but
    # it keeps every player predictable.
    position_avg = (
        df.groupby(["season", "week", "position"])[roll_cols].transform("mean")
    )
    df[roll_cols] = df[roll_cols].fillna(position_avg)

    # Label: NEXT week's fantasy points for that player
    df["target"] = df.groupby("player_id")["fantasy_points_ppr"].shift(-1)

    return df, roll_cols


def main():
    df = load_data()
    print(f"Loaded {len(df)} player-week rows")

    df, feature_cols = build_features(df)

    # Training needs a known target (next week's actual points) -- a
    # player's most recent game won't have one yet, which is expected.
    # It also still needs non-null features; the position-average fallback
    # covers almost everyone, but the very first week of the very first
    # season in YEARS has no prior data at all to average from, so a
    # handful of rows can still be dropped here.
    train_df = df.dropna(subset=feature_cols + ["target"])
    print(f"After feature engineering: {len(train_df)} usable training rows")
    print(f"Features: {feature_cols}")

    X = train_df[feature_cols]
    y = train_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Test MAE: {mae:.2f} fantasy points (lower is better)")

    joblib.dump({"model": model, "features": feature_cols}, "model.joblib")
    print("Saved model.joblib")

    # Save each player's MOST RECENT rolling stats -> this is what the
    # Streamlit app will feed into the model for a live prediction.
    # Use the full df (not train_df): a rookie's most recent game has no
    # "target" yet (their next game hasn't happened), but thanks to the
    # position-average fallback it DOES have usable features, so it should
    # still show up here.
    latest_source = df.dropna(subset=feature_cols)
    latest = (
        latest_source.sort_values(["season", "week"])
        .groupby("player_id")
        .tail(1)[["player_id", "player_display_name", "position", "team"] + feature_cols]
    )
    latest.to_csv("latest_player_features.csv", index=False)
    print(f"Saved latest_player_features.csv ({len(latest)} players)")


if __name__ == "__main__":
    main()
