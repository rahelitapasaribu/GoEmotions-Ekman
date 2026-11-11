import sys
from pathlib import Path

# Menentukan root directory project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Menambahkan root directory ke Python path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="EmotionSense",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# SESSION STATE
# ==========================================

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "predicted_labels" not in st.session_state:
    st.session_state.predicted_labels = []

if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5

from src.inference import EmotionPredictor


@st.cache_resource
def load_predictor():
    return EmotionPredictor()


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F7F5FB;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.hero {
    background: linear-gradient(120deg, #302642, #594477, #8170AA);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero p {
    color: #E3D9F1;
    font-size: 15px;
}

.section-title {
    color: #403451;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 12px;
}

.input-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #E9E3F1;
    box-shadow: 0 4px 15px rgba(65, 48, 87, 0.04);
}

.stTextArea textarea {
    background: #FBFAFD !important;
    border: 1px solid #E5DEEF !important;
    border-radius: 12px !important;
    color: #403451 !important;
}

.stButton > button {
    background: linear-gradient(100deg, #7660A9, #947BC0);
    color: white;
    border: none;
    border-radius: 10px;
    height: 46px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #594477;
    color: white;
    transform: translateY(-2px);
}

.emotion-card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #EAE5F1;
    box-shadow: 0 4px 12px rgba(65, 48, 87, 0.04);
    margin-bottom: 12px;
}

.emotion-name {
    color: #514461;
    font-size: 15px;
    font-weight: 700;
}

.emotion-score {
    color: #75609C;
    font-size: 23px;
    font-weight: 800;
}

.progress-bg {
    background: #EEEAF4;
    border-radius: 20px;
    height: 9px;
    margin-top: 12px;
    overflow: hidden;
}

.progress-fill {
    height: 9px;
    border-radius: 20px;
    background: linear-gradient(90deg, #A28BCB, #7660A9);
}

.active-label {
    display: inline-block;
    background: #E9F5ED;
    color: #34794B;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}

.inactive-label {
    display: inline-block;
    background: #F0EDF4;
    color: #83798F;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #958CA2;
    font-size: 12px;
    padding: 25px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# EMOTION CONFIGURATION
# ==========================================

emotions = {
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😄",
    "sadness": "😢",
    "surprise": "😲",
    "neutral": "😐"
}


# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">
    <div style="font-size:14px; font-weight:600;">
        🧠 EMOTIONSENSE
    </div>
    <h1>Understand your emotions.</h1>
    <p>
        Discover the emotions hidden behind every sentence
        with multi-label emotion classification.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# INPUT SECTION
# ==========================================

st.markdown(
    '<div class="section-title">✍️ Analyze Your Text</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div style="color:#82778F; font-size:13px; margin-bottom:12px;">
    Enter an English sentence below.
</div>
""", unsafe_allow_html=True)


user_text = st.text_area(
    "Your comment",
    key="user_text",
    height=140,
    placeholder="Type your English comment here...",
    label_visibility="collapsed"
)

predict_button = st.button(
    "✨ Analyze Emotions",
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if predict_button:
    if not user_text.strip():
        st.warning("Please enter a comment first.")
    else:
        try:
            with st.spinner("Analyzing text with the trained emotion model..."):
                predictor = load_predictor()
                result = predictor.predict(user_text)

            st.session_state.predictions = result["probabilities"]
            st.session_state.predicted_labels = result["predicted_labels"]
            st.session_state.threshold = result["threshold"]
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.session_state.predictions = None
            st.session_state.predicted_labels = []
            st.session_state.threshold = 0.5


# ==========================================
# RESULT SECTION
# ==========================================

if st.session_state.predictions is not None:
    predictions = st.session_state.predictions
    threshold = st.session_state.threshold
    predicted_labels = st.session_state.predicted_labels

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Your Emotion Profile</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="color:#82778F; font-size:13px; margin-bottom:18px;">
        Here are the predicted emotion probabilities for your text.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if predicted_labels:
        st.markdown(
            "<div style='margin-bottom: 12px;'>"
            + " ".join(
                f'<span class="active-label">{label}</span>'
                for label in predicted_labels
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No emotion exceeded the threshold for this text.")

    active_count = sum(
        score >= threshold for score in predictions.values()
    )

    m1, m2 = st.columns(2)

    with m1:
        st.metric("Active Emotions", f"{active_count} / {len(predictions)}")

    with m2:
        st.metric("Threshold Used", f"{threshold:.2f}")

    st.markdown("### Emotion probabilities")

    cols = st.columns(2)

    for index, (emotion, emoji) in enumerate(emotions.items()):
        probability = predictions.get(emotion, 0.0)
        active = probability >= threshold

        status = (
            '<span class="active-label">● ACTIVE</span>'
            if active
            else '<span class="inactive-label">○ INACTIVE</span>'
        )

        card = f"""
        <div class="emotion-card">
            <div style="display:flex; justify-content:space-between;
                        align-items:center;">
                <div class="emotion-name">
                    {emoji} &nbsp; {emotion.capitalize()}
                </div>
                <div class="emotion-score">
                    {probability:.1%}
                </div>
            </div>
            <div class="progress-bg">
                <div class="progress-fill"
                     style="width:{probability * 100:.2f}%;">
                </div>
            </div>
            <div style="margin-top:12px;">
                {status}
            </div>
        </div>
        """

        with cols[index % 2]:
            st.markdown(card, unsafe_allow_html=True)

    st.markdown("---")

    st.caption(
        "Model inference uses the trained DistilBERT checkpoint and the same preprocessing/tokenization configuration as the training pipeline."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    EmotionSense | Multi-label Emotion Classification
    <br>
    Machine Learning for Text Project
</div>
""", unsafe_allow_html=True)
