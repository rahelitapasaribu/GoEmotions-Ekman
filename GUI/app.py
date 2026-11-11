
import sys
from pathlib import Path

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.inference import EmotionPredictor

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="EmotionSense",
    page_icon=":material/psychology:",
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

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_predictor():
    return EmotionPredictor()

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
# CUSTOM CSS
# ==========================================

st.markdown("""
<link rel="stylesheet"
href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

/* GENERAL */

.stApp {
    background: #F7F5FB;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* HERO SECTION */

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        120deg,
        #292039,
        #49365F,
        #8067A9
    );
    padding: 42px 40px;
    border-radius: 24px;
    color: white;
    margin-bottom: 35px;
    box-shadow: 0 12px 35px rgba(67, 47, 93, 0.16);
}

.hero:before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    border: 45px solid rgba(255,255,255,0.06);
    right: -55px;
    top: -115px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.035);
    right: 80px;
    bottom: -125px;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-label {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    color: #D9C9F2;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 19px;
}

.hero h1 {
    color: white;
    font-size: 39px;
    font-weight: 800;
    margin: 0 0 12px 0;
    letter-spacing: -1px;
}

.hero p {
    color: #E0D7EC;
    font-size: 14px;
    max-width: 570px;
    line-height: 1.8;
    margin: 0;
}

.hero-icon {
    position: absolute;
    right: 100px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 95px;
    color: rgba(255,255,255,0.15);
}

/* SECTION HEADINGS */

.section-title {
    color: #403451;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #958AA3;
    font-size: 13px;
    margin-bottom: 18px;
}

.section-icon {
    color: #8061A9;
    margin-right: 7px;
}

/* INPUT SECTION */

.stTextArea textarea {
    background: #FFFFFF !important;
    border: 1px solid #E6DFEF !important;
    border-radius: 13px !important;
    color: #403451 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 15px !important;
}

.stTextArea textarea:focus {
    border-color: #947BC0 !important;
    box-shadow: 0 0 0 2px rgba(148,123,192,0.12) !important;
}

.stTextArea textarea::placeholder {
    color: #B0A7BC !important;
}

.stButton > button {
    background: linear-gradient(100deg, #72569F, #9579BF);
    color: white;
    border: none;
    border-radius: 11px;
    height: 47px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #594477;
    color: white;
    transform: translateY(-2px);
}

/* SUMMARY CARDS */

.summary-card {
    background: white;
    border: 1px solid #EBE5F2;
    border-radius: 16px;
    padding: 20px 22px;
    min-height: 105px;
    box-shadow: 0 4px 14px rgba(65,48,87,0.035);
}

.summary-label {
    color: #968BA4;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
}

.summary-value {
    color: #49375E;
    font-size: 27px;
    font-weight: 800;
}

.summary-icon {
    color: #947BC0;
    font-size: 17px;
    margin-right: 6px;
}

/* DETECTED EMOTIONS */

.detected-heading {
    color: #49375E;
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 5px;
}

.detected-subtitle {
    color: #958AA3;
    font-size: 12px;
    margin-bottom: 18px;
}

.detected-card {
    background: linear-gradient(
        135deg,
        #FFFFFF,
        #F3EDF9
    );
    border: 1px solid #DCCEEB;
    border-left: 5px solid #8D6DB6;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 13px;
    min-height: 132px;
    box-shadow: 0 5px 15px rgba(91,65,123,0.07);
    transition: 0.2s;
}

.detected-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(91,65,123,0.12);
}

.detected-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.detected-name {
    color: #49375E;
    font-size: 17px;
    font-weight: 800;
}

.detected-emoji {
    font-size: 23px;
    margin-right: 9px;
    vertical-align: middle;
}

.detected-score {
    color: #76549F;
    font-size: 26px;
    font-weight: 800;
}

.detected-status {
    display: inline-block;
    background: #E9F3EC;
    color: #43805B;
    border-radius: 20px;
    padding: 5px 10px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.detected-progress {
    height: 8px;
    border-radius: 20px;
    background: #E9E1F1;
    margin-top: 15px;
    overflow: hidden;
}

.detected-progress-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #B49AD5, #7955A6);
}

/* ALL EMOTION PROBABILITIES */

.probability-card {
    background: white;
    border: 1px solid #ECE7F2;
    border-radius: 14px;
    padding: 17px;
    margin-bottom: 12px;
    min-height: 105px;
}

.probability-card.active {
    border-color: #DCCEEB;
    background: #FCFAFE;
}

.probability-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.probability-name {
    color: #62566F;
    font-size: 14px;
    font-weight: 700;
}

.probability-score {
    color: #796A8A;
    font-size: 19px;
    font-weight: 800;
}

.probability-emoji {
    font-size: 19px;
    margin-right: 9px;
    vertical-align: middle;
}

.probability-card.active .probability-name {
    color: #49375E;
}

.probability-progress {
    height: 6px;
    background: #F0EDF4;
    border-radius: 20px;
    margin-top: 13px;
    overflow: hidden;
}

.probability-progress-fill {
    height: 100%;
    background: #B9A9CB;
    border-radius: 20px;
}

.probability-card.active .probability-progress-fill {
    background: linear-gradient(90deg, #B49AD5, #7955A6);
}

.probability-status {
    margin-top: 10px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

/* EMPTY RESULT */

.empty-result {
    background: #F0ECF5;
    border: 1px dashed #CFC2DE;
    padding: 23px;
    border-radius: 14px;
    color: #796B89;
    font-size: 13px;
    line-height: 1.7;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #A198AD;
    font-size: 11px;
    padding: 30px 10px 10px;
}

/* RESPONSIVE */

@media (max-width: 700px) {

    .hero {
        padding: 30px 24px;
    }

    .hero h1 {
        font-size: 29px;
    }

    .hero-icon {
        display: none;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HERO SECTION
# ==========================================

st.html("""
<div class="hero">

    <div class="hero-content">

        <div class="hero-label">
            <i class="bi bi-cpu"></i>
            EMOTIONSENSE
        </div>

        <h1>Understand your emotions.</h1>

        <p>
            Discover the emotions behind every sentence.
            Analyze text and explore its emotional patterns
            with EMOTIONSENSE multi-label classification.
        </p>

    </div>

    <i class="bi bi-brain hero-icon"></i>

</div>
""")

# ==========================================
# INPUT SECTION
# ==========================================

st.markdown("""
<div class="section-title">
    <i class="bi bi-pencil-square section-icon"></i>
    Analyze Your Text
</div>

<div class="section-subtitle">
    Enter an English sentence to discover its emotional patterns.
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
    "Analyze Emotions  →",
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

            with st.spinner("Analyzing emotional patterns..."):

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

    # Determine active emotions using threshold

    active_emotions = sorted(
        [
            (emotion, score)
            for emotion, score in predictions.items()
            if score >= threshold
        ],
        key=lambda x: x[1],
        reverse=True
    )

    # Sort all emotion probabilities

    all_emotions = sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # ======================================
    # RESULT HEADER
    # ======================================

    st.markdown("---")

    st.markdown("""
    <div class="section-title">
        <i class="bi bi-graph-up-arrow section-icon"></i>
        Emotion Analysis Results
    </div>

    <div class="section-subtitle">
        An overview of the emotional patterns detected in your text.
    </div>
    """, unsafe_allow_html=True)

    # ======================================
    # SUMMARY CARDS
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        st.html(f"""
        <div class="summary-card">

            <div class="summary-label">
                <i class="bi bi-check-circle summary-icon"></i>
                Detected Emotions
            </div>

            <div class="summary-value">
                {len(active_emotions)} / {len(predictions)}
            </div>

        </div>
        """)

    with col2:

        st.html(f"""
        <div class="summary-card">

            <div class="summary-label">
                <i class="bi bi-sliders summary-icon"></i>
                Threshold Used
            </div>

            <div class="summary-value">
                {threshold:.2f}
            </div>

        </div>
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ======================================
    # DETECTED EMOTIONS
    # ======================================

    st.markdown("""
    <div class="detected-heading">
        <i class="bi bi-stars"></i>
        Detected Emotions
    </div>

    <div class="detected-subtitle">
        All emotions with probabilities equal to or above the threshold.
    </div>
    """, unsafe_allow_html=True)

    if active_emotions:

        detected_cols = st.columns(2)

        for index, (emotion, probability) in enumerate(active_emotions):

            emotion_emoji = emotions.get(emotion, "●")

            with detected_cols[index % 2]:

                st.html(f"""
                <div class="detected-card">

                    <div class="detected-card-top">

                        <div>

                            <div class="detected-name">
                                <span class="detected-emoji">
                                    {emotion_emoji}
                                </span>
                                {emotion.capitalize()}
                            </div>

                            <div style="margin-top:9px;">

                                <span class="detected-status">
                                    <i class="bi bi-check-circle-fill"></i>
                                    DETECTED
                                </span>

                            </div>

                        </div>

                        <div class="detected-score">
                            {probability:.1%}
                        </div>

                    </div>

                    <div class="detected-progress">

                        <div class="detected-progress-fill"
                             style="width:{max(0, min(100, probability * 100)):.2f}%;">
                        </div>

                    </div>

                </div>
                """)

    else:

        st.html("""
        <div class="empty-result">

            <i class="bi bi-info-circle"></i>

            No emotion exceeded the classification threshold.
            Try entering another sentence.

        </div>
        """)

    # ======================================
    # ALL EMOTION PROBABILITIES
    # ======================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">
        <i class="bi bi-bar-chart-line section-icon"></i>
        Emotion Probabilities
    </div>

    <div class="section-subtitle">
        Probability scores for all seven emotion categories.
    </div>
    """, unsafe_allow_html=True)

    probability_cols = st.columns(2)

    for index, (emotion, probability) in enumerate(all_emotions):

        emotion_emoji = emotions.get(emotion, "●")

        is_active = probability >= threshold

        card_class = (
            "probability-card active"
            if is_active
            else "probability-card"
        )

        status_text = (
            "DETECTED"
            if is_active
            else "BELOW THRESHOLD"
        )

        status_color = (
            "#43805B"
            if is_active
            else "#988BA8"
        )

        with probability_cols[index % 2]:

            st.html(f"""
            <div class="{card_class}">

                <div class="probability-top">

                    <div class="probability-name">

                        <span class="probability-emoji">
                            {emotion_emoji}
                        </span>

                        {emotion.capitalize()}

                    </div>

                    <div class="probability-score">
                        {probability:.1%}
                    </div>

                </div>

                <div class="probability-progress">

                    <div class="probability-progress-fill"
                         style="width:{max(0, min(100, probability * 100)):.2f}%;">
                    </div>

                </div>

                <div class="probability-status"
                     style="color:{status_color};">

                    {status_text}

                </div>

            </div>
            """)

    # ======================================
    # MODEL INFORMATION
    # ======================================

    st.markdown("---")

    st.caption(
        "Model inference uses the trained DistilBERT checkpoint "
        "and the same preprocessing/tokenization configuration "
        "as the training pipeline."
    )

# ==========================================
# FOOTER
# ==========================================

st.html("""
<div class="footer">

    <i class="bi bi-cpu"></i>

    EmotionSense | Multi-label Emotion Classification

    <br><br>

    Machine Learning for Text Project

</div>
""")