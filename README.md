# Fantasy Football Points Predictor

A beginner-friendly, end-to-end ML demo: pull real NFL data → engineer features →
train a model → serve predictions in a web app → deploy it for free.

**Stack:** Python, pandas, scikit-learn, Streamlit, `nflreadpy`
**Cost:** $0 — everything here is free.

---

## How it works

- `train_model.py` downloads weekly NFL stats (2021–2025) via `nflreadpy`
  (the actively maintained Python port of the nflverse `nflreadr` package —
  `nfl_data_py` is deprecated), computes each player's
  rolling 3-game average (points, targets, yards, TDs, etc.), and trains a
  `RandomForestRegressor` to predict their **next** game's PPR fantasy points.
  It saves `model.joblib` (the trained model) and `latest_player_features.csv`
  (each player's most recent stats, so the app has something to predict from).
- `app.py` is a Streamlit web app: pick a position and player, see their recent
  form, click a button, get a predicted point total.

---

## Step 1 — Set up your environment

You need Python 3.9+ installed. Then, in a terminal, in a new project folder
containing these 4 files:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2 — Train the model

```bash
python train_model.py
```

This downloads a few seasons of NFL data (takes 1–3 minutes the first time),
trains the model, and prints a test error (Mean Absolute Error, in fantasy
points — lower is better). You should see two new files appear:
`model.joblib` and `latest_player_features.csv`. **Keep both** — the app needs them.

## Step 3 — Run the app locally

```bash
streamlit run app.py
```

This opens a browser tab at `localhost:8501`. Pick a position, pick a player,
click **Predict next game points**. That's your working ML app, running locally.

## Step 4 — Put it on GitHub

Streamlit Community Cloud deploys straight from a GitHub repo.

1. Create a free account at [github.com](https://github.com) if you don't have one.
2. Create a new repository (e.g. `fantasy-points-predictor`), public.
3. Push these files, **including** `model.joblib` and `latest_player_features.csv`:

```bash
git init
git add train_model.py app.py requirements.txt README.md model.joblib latest_player_features.csv
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fantasy-points-predictor.git
git push -u origin main
```

> **Why commit the model file?** It keeps deployment simple — the cloud app just
> loads the pre-trained model instead of re-downloading NFL data and retraining
> on startup. (`model.joblib` will likely be a few MB, well within GitHub's limits.)

## Step 5 — Deploy for free on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Pick your repo, branch `main`, and main file `app.py`.
4. Click **Deploy**.

In a minute or two you'll get a public URL like
`https://your-app-name.streamlit.app` — a live, shareable ML web app.

---

## Current-season data & rookies

- `YEARS` in `train_model.py` already includes the current season. `nflreadpy`
  pulls from the public nflverse data repo, which updates within about a day
  after each week's games — no manual download needed. Just re-run
  `train_model.py` periodically to refresh `model.joblib` and
  `latest_player_features.csv` with new games.
- Rookies and players with little game history have no prior game to average,
  so `build_features()` falls back to the **league-average rolling stats for
  their position** that season/week. This is a rough "replacement level"
  estimate, not real history — it's there so the app can still produce a
  prediction instead of silently dropping the player. Once a rookie has
  played a game or two, their real rolling stats take over automatically.

## Ways to extend this (good for a portfolio)

- Add more features: opponent defense strength, home/away, weather, injury status.
- Try other models (XGBoost, gradient boosting) and compare MAE.
- Replace the rookie fallback with real draft capital / combine data from
  `nflreadpy.load_draft_picks()` and `load_combine()` for a better rookie
  prior than a plain position average.
- Add a chart of the player's last N games vs. the prediction (Streamlit has
  built-in `st.line_chart`).
- Retrain automatically each week with a scheduled GitHub Action.
- Swap Random Forest for a simple neural net to show a deep learning example too.

## Notes

- `nflreadpy` pulls from the public, freely available nflverse data repositories
  — no API key needed. It returns Polars DataFrames; `train_model.py` converts
  these to pandas with `.to_pandas()` so the rest of the code is unchanged.
- This is a demo/portfolio project, not a betting or DFS advice tool.
# nfl-fantasy-ppr-predictor
