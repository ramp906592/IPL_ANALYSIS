import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(page_title="IPL Analytics Dashboard", layout="wide")
st.title("🏏 IPL Match Prediction & Player Comparison")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    model = joblib.load("ipl_match_model.pkl")
    team_enc = joblib.load("team_encoder.pkl")
    city_enc = joblib.load("city_encoder.pkl")
    return model, team_enc, city_enc

match_model, team_encoder, city_encoder = load_models()

# ================= LOAD PLAYER DATA =================
@st.cache_data
def load_players():
    df = pd.read_csv("cricket_data_2025.csv")
    df.replace("No stats", np.nan, inplace=True)

    non_numeric = ["Player_Name", "Year", "Highest_Score", "Best_Bowling_Match"]
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

player_df = load_players()
PLAYER_COL = "Player_Name"

# ================= SIDEBAR =================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Feature",
    ["🏠 Home", "🏏 Match Prediction", "📊 Player Comparison"],
    key="main_nav_radio"
)

# =================================================
# 🏠 HOME PAGE
# =================================================
if page == "🏠 Home":

    st.markdown("## 🏏 Cricket Prediction Analytics")
    st.markdown("### Welcome to Cricket Prediction Analytics!")

    c1, c2 = st.columns(2)
    c1.metric("Players Available", player_df[PLAYER_COL].nunique())
    c2.metric("Model Status", "Loaded")

    st.markdown("### 🚀 Features Available")
    st.markdown("""
    • **Player Comparison** – Batting, Bowling & Overall  
    • **Match Prediction** – Live win probability  
    • **Raw Career Stats** – Non-normalized data  
    """)

    st.info("👉 Use sidebar to navigate")

# =================================================
# 🏏 MATCH PREDICTION
# =================================================
elif page == "🏏 Match Prediction":

    st.header("🏆 Match Winning Probability")

    teams = team_encoder.classes_
    cities = city_encoder.classes_

    c1, c2 = st.columns(2)
    with c1:
        batting_team = st.selectbox("Batting Team", teams, key="bat_team")
        target = st.number_input("Target", 0, 300, 160)
        overs = st.number_input("Overs Completed", 0.0, 20.0, 6.0)

    with c2:
        bowling_team = st.selectbox("Bowling Team", teams, key="bowl_team")
        score = st.number_input("Current Score", 0, 300, 50)
        wickets = st.number_input("Wickets Lost", 0, 10, 2)

    city = st.selectbox("City", cities, key="city")

    if st.button("Predict Probability", key="predict_btn"):

        bt = team_encoder.transform([batting_team])[0]
        bw = team_encoder.transform([bowling_team])[0]
        ct = city_encoder.transform([city])[0]

        balls_left = max(0, 120 - int(overs * 6))
        runs_left = max(0, target - score)

        features = np.array([[bt, bw, ct, target, score,
                              overs, wickets, runs_left, balls_left]])

        probs = match_model.predict_proba(features)[0]

        bat_prob = round(probs[1] * 100, 2)
        bowl_prob = round(probs[0] * 100, 2)

        st.subheader("📊 Winning Probability")

        c1, c2 = st.columns(2)
        c1.metric(batting_team, f"{bat_prob}%")
        c1.progress(bat_prob / 100)

        c2.metric(bowling_team, f"{bowl_prob}%")
        c2.progress(bowl_prob / 100)

# =================================================
# 📊 PLAYER COMPARISON
# =================================================
elif page == "📊 Player Comparison":

    st.header("📊 Player Comparison (RAW Career Stats)")

    players = sorted(player_df[PLAYER_COL].dropna().unique())

    c1, c2 = st.columns(2)
    p1 = c1.selectbox("Select Player 1", players, key="p1")
    p2 = c2.selectbox("Select Player 2", players, key="p2")

    if st.button("Compare Players", key="compare_btn"):

        df1 = player_df[player_df[PLAYER_COL] == p1]
        df2 = player_df[player_df[PLAYER_COL] == p2]

        def career(df):
            return {
                "Matches Batted": df["Matches_Batted"].sum(),
                "Runs": df["Runs_Scored"].sum(),
                "Batting Avg": round(df["Batting_Average"].mean(), 2),
                "Strike Rate": round(df["Batting_Strike_Rate"].mean(), 2),
                "Centuries": df["Centuries"].sum(),
                "Half Centuries": df["Half_Centuries"].sum(),
                "Wickets": df["Wickets_Taken"].sum(),
                "Economy": round(df["Economy_Rate"].mean(), 2),
                "Bowling Avg": round(df["Bowling_Average"].mean(), 2),
                "Bowling SR": round(df["Bowling_Strike_Rate"].mean(), 2),
                "Matches Played": df["Year"].count()
            }

        s1, s2 = career(df1), career(df2)
        win_count = {p1: 0, p2: 0}

        # ================= BATTTING =================
        st.subheader("🏏 Batting Comparison")
        bat_rows = []
        for m in ["Matches Batted", "Runs", "Batting Avg", "Strike Rate", "Centuries", "Half Centuries"]:
            winner = p1 if s1[m] > s2[m] else p2
            win_count[winner] += 1
            bat_rows.append({
                "Metric": m,
                p1: s1[m],
                p2: s2[m],
                "Winner": f"✅ {winner}"
            })
        st.dataframe(pd.DataFrame(bat_rows), use_container_width=True)

        # ================= BOWLING =================
        st.subheader("🎯 Bowling Comparison")
        bowl_rows = []
        for m in ["Wickets", "Economy", "Bowling Avg", "Bowling SR"]:
            winner = p1 if (s1[m] > s2[m] if m == "Wickets" else s1[m] < s2[m]) else p2
            win_count[winner] += 1
            bowl_rows.append({
                "Metric": m,
                p1: s1[m],
                p2: s2[m],
                "Winner": f"✅ {winner}"
            })
        st.dataframe(pd.DataFrame(bowl_rows), use_container_width=True)

        # ================= OVERALL =================
        st.subheader("🏆 Overall Comparison")
        overall_rows = []
        for m in ["Matches Played", "Runs", "Wickets"]:
            winner = p1 if s1[m] > s2[m] else p2
            win_count[winner] += 1
            overall_rows.append({
                "Metric": m,
                p1: s1[m],
                p2: s2[m],
                "Winner": f"✅ {winner}"
            })
        st.dataframe(pd.DataFrame(overall_rows), use_container_width=True)

        # ================= SUMMARY =================
        total = sum(win_count.values())
        st.subheader("🏆 Overall Statistics")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{p1} Wins", win_count[p1])
        c2.metric(f"{p2} Wins", win_count[p2])
        c3.metric("Total Metrics", total)

        final_winner = p1 if win_count[p1] > win_count[p2] else p2
        st.success(f"🏆 {final_winner} is the OVERALL WINNER!")
        st.info(f"Won {max(win_count.values())} out of {total} metrics")

        st.caption("All stats shown are RAW career values (non-normalized).")
