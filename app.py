"""
app.py

Streamlit app: pick a player, see their recent form, get a predicted
fantasy points total for their next game.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fantasy Football Points Predictor", page_icon="🏈")


@st.cache_resource
def load_model():
    bundle = joblib.load("model.joblib")
    return bundle["model"], bundle["features"]


@st.cache_data
def load_players():
    return pd.read_csv("latest_player_features.csv")


model, feature_cols = load_model()
players = load_players()

st.title("🏈 Fantasy Football Points Predictor")
st.caption(
    "A demo ML app: predicts a player's next-game PPR fantasy points "
    "based on their rolling average over their last 3 games."
)

position = st.selectbox("Position", sorted(players["position"].unique()))
filtered = players[players["position"] == position]
player_name = st.selectbox("Player", sorted(filtered["player_display_name"].unique()))

row = filtered[filtered["player_display_name"] == player_name].iloc[0]

st.subheader("Recent form (avg over last 3 games)")
display = row[feature_cols].rename("value").to_frame()
st.dataframe(display)

if st.button("Predict next game points", type="primary"):
    X = row[feature_cols].to_frame().T
    pred = model.predict(X)[0]
    st.metric(f"Predicted fantasy points — {player_name}", f"{pred:.1f}")
    st.caption("PPR scoring. This is a demo model, not betting advice.")
