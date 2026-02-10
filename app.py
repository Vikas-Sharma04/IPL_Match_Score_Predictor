import math
import joblib
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

# --- CONFIG & STYLING ---
st.set_page_config(page_title='IPL Score Predictor 2026', page_icon='🏏', layout="centered")

st.markdown("""
<style>

/* =======================
   BACKGROUND + OVERLAY
======================= */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?q=80&w=2073&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(5, 15, 30, 0.75);
    z-index: -1;
}

/* =======================
   MAIN GLASS CONTAINER
======================= */
.main-container {
    max-width: 1100px;
    margin: auto;
    padding: 40px;
    border-radius: 25px;
    background: rgba(15, 30, 50, 0.55);
    backdrop-filter: blur(14px);
    box-shadow: 0 30px 60px rgba(0,0,0,0.85);
}

/* =======================
   INPUT STYLING
======================= */
div[data-baseweb="input"],
div[data-baseweb="select"] > div,
.stNumberInput div {
    background-color: #0f1f33 !important;
    border-radius: 12px !important;
    border: 1px solid #3a4b63 !important;
    color: white !important;
}

/* +/- buttons */
.stNumberInput button {
    background-color: #1f344d !important;
    color: #00ff99 !important;
}

/* =======================
   BUTTONS
======================= */
button[kind="primary"] {
    background: linear-gradient(45deg, #ff4b2b, #ff416c) !important;
    height: 52px;
    font-weight: 800;
    letter-spacing: 1px;
    border-radius: 14px;
}

button[kind="secondary"] {
    background: #2c3e50 !important;
    height: 52px;
    font-weight: 700;
    border-radius: 14px;
}

/* =======================
   STAT CARDS
======================= */
.stat-card {
    background: rgba(10, 25, 45, 0.85);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #3a4b63;
}

.stat-value {
    font-size: 34px;
    color: #00ff99;
    font-weight: 900;
}

/* =======================
   RESULT CARD
======================= */
.result-card {
    margin-top: 30px;
    padding: 35px;
    border-radius: 22px;
    background: rgba(5, 20, 40, 0.95);
    border: 2px solid #00ff99;
    text-align: center;
}

label, p {
    color: white !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# =======================
# SESSION RESET LOGIC
# =======================
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def reset_all():
    st.session_state.reset_key += 1


# =======================
# MODEL LOAD FROM HUGGING FACE
# =======================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="VikasSharma04/IPL_Match_Score_Predictor",
        filename="ml_model.joblib"
    )
    return joblib.load(model_path)

try:
    with st.spinner("📥 Loading ML model (first run may take a minute)..."):
        model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load ML model: {e}")
    st.stop()


# =======================
# TEAM MAPPING
# =======================
ui_to_model_map = {
    'Chennai Super Kings': 'Chennai Super Kings',
    'Delhi Capitals': 'Delhi Daredevils',
    'Punjab Kings': 'Kings XI Punjab',
    'Kolkata Knight Riders': 'Kolkata Knight Riders',
    'Mumbai Indians': 'Mumbai Indians',
    'Rajasthan Royals': 'Rajasthan Royals',
    'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
    'Sunrisers Hyderabad': 'Sunrisers Hyderabad'
}

model_teams = list(ui_to_model_map.values())


# =======================
# APP TITLE
# =======================
st.markdown(
    "<h1 style='text-align:center; color:white; font-size:3.5rem; text-shadow:4px 4px 10px black;'>🏏 IPL Score Predictor 2026</h1>",
    unsafe_allow_html=True
)

key = st.session_state.reset_key


# =======================
# TEAM SELECTORS
# =======================
col_t1, col_t2 = st.columns(2)
with col_t1:
    batting_team = st.selectbox('Select Batting Team', list(ui_to_model_map.keys()), key=f"bat_{key}")
with col_t2:
    bowling_team = st.selectbox('Select Bowling Team', list(ui_to_model_map.keys()), key=f"bowl_{key}")

st.markdown("<br>", unsafe_allow_html=True)


# =======================
# INPUT FIELDS
# =======================
c1, c2, c3 = st.columns(3)
with c1:
    overs = st.number_input('Overs Completed', 5.1, 19.5, 10.0, 0.1, key=f"o_{key}")
with c2:
    runs = st.number_input('Current Runs', 0, 350, 80, 1, key=f"r_{key}")
with c3:
    wickets = st.number_input('Wickets Out', 0, 9, 2, 1, key=f"w_{key}")


# =======================
# RUN RATE CALCULATION
# =======================
valid_overs = math.floor(overs) + (overs % 1) * 10 / 6
crr = round(runs / valid_overs, 2) if valid_overs > 0 else 0.0

st.markdown(f"""
<div style="display: flex; justify-content: space-around; margin: 20px 0;">
    <div style="background:rgba(26, 38, 52, 0.8); padding:15px; border-radius:15px; width:45%; text-align:center; border:1px solid #3e4a59;">
        <small style="color:#bdc3c7;">CURRENT RUN RATE</small><br>
        <b style="font-size:30px; color:#00FF00;">{crr}</b>
    </div>
    <div style="background:rgba(26, 38, 52, 0.8); padding:15px; border-radius:15px; width:45%; text-align:center; border:1px solid #3e4a59;">
        <small style="color:#bdc3c7;">PROJECTED TOTAL</small><br>
        <b style="font-size:30px; color:#00FF00;">{int(crr * 20)}</b>
    </div>
</div>
""", unsafe_allow_html=True)


col_p1, col_p2 = st.columns(2)
with col_p1:
    runs_last_5 = st.number_input('Runs in Last 5 Overs', 0, runs, 40, 1, key=f"r5_{key}")
with col_p2:
    wickets_last_5 = st.number_input('Wickets in Last 5 Overs', 0, wickets, 1, 1, key=f"w5_{key}")


# =======================
# ACTION BUTTONS
# =======================
st.markdown("<br>", unsafe_allow_html=True)
btn1, btn2 = st.columns(2)
with btn1:
    predict_btn = st.button('🚀 PREDICT SCORE', use_container_width=True, type="primary")
with btn2:
    st.button('🧹 CLEAR ALL', on_click=reset_all, use_container_width=True)


# =======================
# PREDICTION LOGIC
# =======================
if predict_btn:
    if batting_team == bowling_team:
        st.error("Teams must be different!")
    else:
        m_bat = ui_to_model_map[batting_team]
        m_bow = ui_to_model_map[bowling_team]

        encoded = (
            [1 if m_bat == t else 0 for t in model_teams] +
            [1 if m_bow == t else 0 for t in model_teams] +
            [runs, wickets, valid_overs, runs_last_5, wickets_last_5]
        )

        feature_cols = (
            [f'batting_team_{t}' for t in model_teams] +
            [f'bowling_team_{t}' for t in model_teams] +
            ['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']
        )

        prediction = model.predict(pd.DataFrame([encoded], columns=feature_cols))
        final_score = int(round(prediction[0]))

        win_prob = min(max(int((final_score / 190) * 100), 5), 95)

        st.balloons()
        st.markdown(f"""
        <div class="result-card">
            <h3 style="color:white; margin:0;">ML PREDICTED TOTAL</h3>
            <h1 style="color:#00FF00; font-size:75px; margin:10px 0;">
                {final_score - 5} — {final_score + 5}
            </h1>
            <hr style="border:0.5px solid #3e4a59">
            <h3 style="color:white; margin-top:10px;">
                WIN PROBABILITY: <span style="color:#FFD700;">{win_prob}%</span>
            </h3>
            <div style="background:#1a2634; border-radius:10px; height:20px; width:100%; margin-top:10px;">
                <div style="background:linear-gradient(to right, #00FF00, #FFD700);
                            width:{win_prob}%; height:100%; border-radius:10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)