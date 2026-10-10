import math

import joblib
from huggingface_hub import hf_hub_download
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="IPL Score Predictor 2026",
    page_icon="🏏",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
    --ink: #e9f3ee;
    --muted: #91a89d;
    --surface: #0d211a;
    --surface-raised: #11291f;
    --line: rgba(160, 205, 180, .16);
    --green: #35d39a;
    --gold: #e6b75a;
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(ellipse at 82% 0%, rgba(22, 94, 66, .28), transparent 42%),
        radial-gradient(ellipse at 0% 70%, rgba(18, 69, 54, .2), transparent 40%),
        #07130f;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
footer,
.stDeployButton {
    display: none !important;
}

.block-container {
    max-width: 1320px;
    padding: 2.5rem 2rem 3.5rem;
}

h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.page-heading {
    margin: 0 0 2rem;
}

.eyebrow {
    color: var(--green);
    font-size: .73rem;
    font-weight: 800;
    letter-spacing: .2em;
    text-transform: uppercase;
}

.page-title {
    color: #f3f8f5;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2rem, 4vw, 3.15rem);
    font-weight: 600;
    letter-spacing: -.04em;
    line-height: 1.08;
    margin: .5rem 0 .65rem;
}

.page-subtitle {
    color: var(--muted);
    font-size: 1rem;
    margin: 0;
}

.section-title {
    color: #edf6f0;
    font-size: 1.05rem;
    font-weight: 700;
    margin: .25rem 0 1rem;
}

.section-note {
    color: var(--muted);
    font-size: .84rem;
    margin: -.6rem 0 1.2rem;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #c2d2c9 !important;
    font-size: .86rem;
    font-weight: 600;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background: #10251d !important;
    border: 1px solid rgba(160, 205, 180, .2) !important;
    border-radius: 11px !important;
    color: var(--ink) !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="input"] input {
    color: var(--ink) !important;
}

div[data-testid="stNumberInput"] button {
    background: #19362a !important;
    border: 0 !important;
    color: var(--green) !important;
}

div[data-testid="stNumberInput"] button:hover {
    background: #214534 !important;
}

button[kind="primary"] {
    background: linear-gradient(110deg, #1d9b70, #35d39a) !important;
    border: 0 !important;
    border-radius: 11px !important;
    color: #04150f !important;
    font-weight: 800 !important;
    height: 3rem;
    letter-spacing: .04em;
}

button[kind="primary"]:hover {
    background: linear-gradient(110deg, #28b985, #65e6b5) !important;
    box-shadow: 0 8px 24px rgba(53, 211, 154, .16);
}

button[kind="secondary"] {
    background: #14271f !important;
    border: 1px solid var(--line) !important;
    border-radius: 11px !important;
    color: #d4e3da !important;
    font-weight: 700 !important;
    height: 3rem;
}

button[kind="secondary"]:hover {
    border-color: rgba(53, 211, 154, .55) !important;
    color: var(--green) !important;
}

/* Pushes the right layout card up to align with the page header eyebrow */
.signal-card {
    align-items: center;
    background:
        linear-gradient(rgba(152, 194, 169, .035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(152, 194, 169, .035) 1px, transparent 1px),
        radial-gradient(ellipse at 50% 35%, rgba(22, 100, 69, .28), transparent 58%),
        linear-gradient(145deg, #10291f, #081710);
    background-size: 34px 34px, 34px 34px, auto, auto;
    border: 1px solid rgba(103, 178, 138, .23);
    border-radius: 24px;
    box-shadow: 0 28px 70px rgba(0, 0, 0, .3);
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-top: -160px;
    min-height: 465px;
    overflow: hidden;
    padding: 2rem;
    position: relative;
    text-align: center;
}

.signal-meta {
    align-items: center;
    color: #a7bdb1;
    display: flex;
    font-size: .7rem;
    font-weight: 800;
    justify-content: space-between;
    left: 2rem;
    letter-spacing: .12em;
    position: absolute;
    right: 2rem;
    text-transform: uppercase;
    top: 2rem;
}

.status-dot {
    background: var(--gold);
    border-radius: 50%;
    display: inline-block;
    height: 7px;
    margin-right: .55rem;
    vertical-align: 1px;
    width: 7px;
}

.signal-orb {
    align-items: center;
    border: 1px solid rgba(53, 211, 154, .28);
    border-radius: 50%;
    color: var(--green);
    display: flex;
    font-size: 1.8rem;
    height: 64px;
    justify-content: center;
    margin: 0 auto 2.2rem;
    position: relative;
    width: 64px;
}

.signal-orb::before,
.signal-orb::after {
    border: 1px solid rgba(53, 211, 154, .18);
    border-radius: 50%;
    content: "";
    inset: -14px;
    position: absolute;
}

.signal-orb::after {
    inset: -27px;
}

.signal-card h2 {
    color: #f2f7f3;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(1.6rem, 3vw, 2.35rem);
    margin: 0 0 .65rem;
}

.signal-card p {
    color: #a5baae;
    line-height: 1.65;
    margin: 0 auto;
    max-width: 390px;
}

.signal-divider {
    background: linear-gradient(90deg, transparent, rgba(126, 171, 146, .28), transparent);
    height: 1px;
    margin: 2rem auto 1.25rem;
    max-width: 75%;
    width: 340px;
}

.signal-footnote {
    color: #71897c;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.metric-card {
    background: linear-gradient(145deg, rgba(17, 41, 31, .95), rgba(11, 29, 21, .95));
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1rem 1.1rem;
}

.metric-label {
    color: #91a89d;
    display: block;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .1em;
    margin-bottom: .4rem;
}

.metric-value {
    color: #f0f7f2;
    font-size: 1.75rem;
    font-weight: 750;
    letter-spacing: -.04em;
}

.metric-unit {
    color: var(--green);
    font-size: .78rem;
    font-weight: 700;
    margin-left: .35rem;
}

.result-card {
    background:
        radial-gradient(ellipse at 50% 0%, rgba(53, 211, 154, .12), transparent 58%),
        linear-gradient(145deg, #10291f, #091710);
    border: 1px solid rgba(53, 211, 154, .38);
    border-radius: 22px;
    margin-top: 1.5rem;
    padding: 1.8rem;
    text-align: center;
}

.result-label {
    color: #9eb7aa;
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: .16em;
}

.result-score {
    color: var(--green);
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2.6rem, 6vw, 4.5rem);
    line-height: 1.1;
    margin: .6rem 0 1rem;
}

[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
}

@media (max-width: 760px) {
    .block-container {
        padding: 1.5rem 1rem 2rem;
    }

    .signal-card {
        margin-top: 0rem;
        min-height: 360px;
        padding: 1.5rem;
    }

    .signal-meta {
        left: 1.25rem;
        right: 1.25rem;
        top: 1.25rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

if "reset_key" not in st.session_state:
    st.session_state.reset_key = 0


def reset_all():
    st.session_state.reset_key += 1


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="VikasSharma04/IPL_Match_Score_Predictor", filename="ml_model.joblib"
    )
    return joblib.load(model_path)


try:
    with st.spinner("📥 Loading ML model (first run may take a minute)..."):
        model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load ML model: {e}")
    st.stop()

ui_to_model_map = {
    "Chennai Super Kings": "Chennai Super Kings",
    "Delhi Capitals": "Delhi Daredevils",
    "Punjab Kings": "Kings XI Punjab",
    "Kolkata Knight Riders": "Kolkata Knight Riders",
    "Mumbai Indians": "Mumbai Indians",
    "Rajasthan Royals": "Rajasthan Royals",
    "Royal Challengers Bangalore": "Royal Challengers Bangalore",
    "Sunrisers Hyderabad": "Sunrisers Hyderabad",
}
model_teams = list(ui_to_model_map.values())

st.markdown(
    """
<div class="page-heading">
    <div class="eyebrow">IPL · Predictive analytics</div>
    <h1 class="page-title">Score Predictor</h1>
    <p class="page-subtitle">Live match inputs, powered by machine learning.</p>
</div>
""",
    unsafe_allow_html=True,
)

key = st.session_state.reset_key
input_column, insight_column = st.columns(
    [1.05, 0.95],
    gap="large",
)

with input_column:
    st.markdown('<div class="section-title">Match context</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Set the teams and enter the current innings state.</div>',
        unsafe_allow_html=True,
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        batting_team = st.selectbox(
            "Batting team", list(ui_to_model_map.keys()), key=f"bat_{key}"
        )
    with col_t2:
        bowling_team = st.selectbox(
            "Bowling team", list(ui_to_model_map.keys()), key=f"bowl_{key}"
        )

    st.markdown('<div class="section-title">Live match data</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        overs = st.number_input(
            "Overs completed", 5.1, 19.5, 10.0, 0.1, key=f"o_{key}"
        )
    with c2:
        runs = st.number_input("Current runs", 0, 350, 80, 1, key=f"r_{key}")
    with c3:
        wickets = st.number_input("Wickets out", 0, 9, 2, 1, key=f"w_{key}")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        runs_last_5 = st.number_input(
            "Runs in last 5 overs", 0, runs, 40, 1, key=f"r5_{key}"
        )
    with col_p2:
        wickets_last_5 = st.number_input(
            "Wickets in last 5 overs", 0, wickets, 1, 1, key=f"w5_{key}"
        )

    btn1, btn2 = st.columns(2)
    with btn1:
        predict_btn = st.button(
            "PREDICT SCORE", use_container_width=True, type="primary"
        )
    with btn2:
        st.button(
            "CLEAR INPUTS",
            on_click=reset_all,
            use_container_width=True,
            type="secondary",
        )

valid_overs = math.floor(overs) + (overs % 1) * 10 / 6
crr = round(runs / valid_overs, 2) if valid_overs > 0 else 0.0
projected_total = int(crr * 20)

with insight_column:
    st.markdown(
        f"""
<div class="signal-card">
    <div class="signal-meta">
        <span><i class="status-dot"></i>System idle</span>
        <span>AI engine ready</span>
    </div>
    <div class="signal-orb">✦</div>
    <h2>Awaiting input signal</h2>
    <p>Enter the live match details to activate your real-time score forecast.</p>
    <div class="signal-divider"></div>
    <div class="signal-footnote">Current innings snapshot</div>
</div>
<div style="height: 1rem"></div>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:.8rem;">
    <div class="metric-card">
        <span class="metric-label">CURRENT RUN RATE</span>
        <span class="metric-value">{crr:.2f}</span><span class="metric-unit">RR</span>
    </div>
    <div class="metric-card">
        <span class="metric-label">PROJECTED TOTAL</span>
        <span class="metric-value">{projected_total}</span><span class="metric-unit">RUNS</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

if predict_btn:
    if batting_team == bowling_team:
        st.error("Batting and bowling teams must be different.")
    else:
        m_bat = ui_to_model_map[batting_team]
        m_bow = ui_to_model_map[bowling_team]
        encoded = [1 if m_bat == team else 0 for team in model_teams] + [
            1 if m_bow == team else 0 for team in model_teams
        ]
        encoded.extend([runs, wickets, overs, runs_last_5, wickets_last_5])

        feature_cols = (
            [f"batting_team_{team}" for team in model_teams]
            + [f"bowling_team_{team}" for team in model_teams]
            + ["runs", "wickets", "overs", "runs_last_5", "wickets_last_5"]
        )

        prediction = model.predict(pd.DataFrame([encoded], columns=feature_cols))
        final_score = int(round(prediction[0]))

        st.markdown(
            f"""
<div class="result-card">
    <div class="result-label">ML PREDICTED TOTAL</div>
    <div class="result-score">{final_score - 5} — {final_score + 5}</div>
</div>
""",
            unsafe_allow_html=True,
        )