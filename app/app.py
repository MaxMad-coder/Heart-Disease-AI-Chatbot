"""
AI-Powered Medical Chatbot for Heart Disease Prediction
Streamlit Application — Main Entry Point
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chatbot.chatbot import ChatSession, State
from models.predict import (
    build_patient_explanation,
    generate_shap_explanation,
    load_meta,
    load_model,
    load_shap_importance,
    predict,
)
from preprocessing.preprocessor import decode_features, prepare_single_input

# ── Streamlit config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="💓 Heart Disease AI Chatbot",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Global */
    .main { background: #0f1117; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    
    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 18px 18px 4px 18px;
        padding: 12px 16px; margin: 8px 0 8px 60px;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .chat-bot {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0; border-radius: 18px 18px 18px 4px;
        padding: 12px 16px; margin: 8px 60px 8px 0;
        border: 1px solid #333; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #333; border-radius: 12px;
        padding: 20px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card h2 { color: #667eea; font-size: 2em; margin: 0; }
    .metric-card p  { color: #aaa; margin: 4px 0 0; }
    
    /* Disclaimer */
    .disclaimer {
        background: linear-gradient(135deg, #2d1b1b, #1a0a0a);
        border: 1px solid #cc4444; border-radius: 10px;
        padding: 12px 16px; color: #ff9999;
        font-size: 0.85em;
    }
    
    /* Result cards */
    .result-high {
        background: linear-gradient(135deg, #3d0000, #7a0000);
        border: 2px solid #f44336; border-radius: 15px;
        padding: 24px; text-align: center; color: white;
    }
    .result-low {
        background: linear-gradient(135deg, #003d00, #007a00);
        border: 2px solid #4caf50; border-radius: 15px;
        padding: 24px; text-align: center; color: white;
    }
    .result-high h1, .result-low h1 { font-size: 2em; margin: 0 0 10px; }
    
    /* Gauge text */
    .probability-label { font-size: 3em; font-weight: bold; }
    
    /* FAQ */
    .faq-q { color: #667eea; font-weight: bold; margin-top: 16px; }
    .faq-a { color: #ccc; margin-left: 16px; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e; color: #e0e0e0;
        border-radius: 8px 8px 0 0; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* Input fields */
    .stTextInput > div > input {
        background: #1a1a2e; color: white; border: 1px solid #333;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 10px;
        padding: 8px 24px; font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.5);
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────

def init_session():
    defaults: dict = {
        "chat_session":   None,
        "model":          None,
        "meta":           {},
        "chat_history":   [],   # list of {role, content}
        "current_page":   "🏠 Home",
        "assessment_done": False,
        "prediction_result": None,
        "patient_data":   {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── Load model (cached) ────────────────────────────────────────────────────────

@st.cache_resource
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None

@st.cache_resource
def get_meta():
    return load_meta()

@st.cache_resource
def get_shap():
    return load_shap_importance()

model     = get_model()
meta      = get_meta()
shap_vals = get_shap()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💓 Heart Disease AI")
    st.markdown("---")
    pages = [
        "🏠 Home",
        "🤖 Risk Assessment Chatbot",
        "📊 Prediction Insights",
        "📚 Health Education",
        "ℹ️ About",
    ]
    for page in pages:
        if st.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.current_page = page

    st.markdown("---")
    if meta:
        st.markdown("### 🏆 Best Model")
        st.markdown(f"**{meta.get('best_model', 'N/A')}**")
        st.metric("ROC-AUC", f"{meta.get('roc_auc', 0):.3f}")
        st.metric("Accuracy", f"{meta.get('accuracy', 0):.3f}")
        st.metric("F1 Score", f"{meta.get('f1', 0):.3f}")

    st.markdown("---")
    st.markdown("""
    <div class='disclaimer'>
    ⚠️ <strong>Medical Disclaimer</strong><br>
    This tool is for <em>educational purposes only</em>. 
    Not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)

# ── Helper: render chat history ────────────────────────────────────────────────

def render_chat():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">👤 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bot">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

def bot_say(text: str):
    st.session_state.chat_history.append({"role": "bot", "content": text})

def user_say(text: str):
    st.session_state.chat_history.append({"role": "user", "content": text})

# ── Page: Home ─────────────────────────────────────────────────────────────────

def page_home():
    st.markdown("""
    <div style="text-align:center; padding:40px 0 20px;">
        <h1 style="font-size:3em;">💓 AI Heart Disease Prediction</h1>
        <p style="font-size:1.2em; color:#aaa;">
            An intelligent medical chatbot powered by machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("1,025", "Patient Records"),
        ("13", "Clinical Features"),
        (f"{meta.get('accuracy', 0.88)*100:.1f}%", "Model Accuracy"),
        ("6", "ML Models Trained"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(
                f'<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## ✨ Key Features")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.info("🤖 **AI Chatbot**\nConversational interface to collect patient data interactively.")
    with f2:
        st.info("📊 **ML Prediction**\nMuliple trained models ranked by ROC-AUC; best model selected automatically.")
    with f3:
        st.info("🔍 **Explainable AI**\nSHAP values explain which factors drive each prediction.")
    f4, f5, f6 = st.columns(3)
    with f4:
        st.info("📈 **Visual Insights**\nConfusion matrices, ROC curves, and feature importance charts.")
    with f5:
        st.info("❤️ **Health Education**\nLearn about heart disease, risk factors, and prevention.")
    with f6:
        st.info("🔒 **Privacy First**\nNo patient data is stored — all processing is in-memory.")

    st.markdown("---")
    st.markdown("""
    <div class='disclaimer' style='max-width:700px; margin:0 auto;'>
    ⚠️ <strong>Important Medical Disclaimer:</strong> This application is developed for 
    educational and research purposes only. The predictions provided by this system should 
    not be used as a substitute for professional medical diagnosis or advice. Always consult 
    a qualified healthcare professional for medical concerns. The creators of this tool are 
    not responsible for any medical decisions made based on its output.
    </div>
    """, unsafe_allow_html=True)

# ── Page: Risk Assessment Chatbot ─────────────────────────────────────────────

def page_chatbot():
    st.markdown("## 🤖 Heart Disease Risk Assessment")

    if model is None:
        st.error("⚠️ Model not found. Run `python models/train_model.py` first.")
        return

    # Init chat session
    if st.session_state.chat_session is None:
        st.session_state.chat_session = ChatSession()
        bot_say(
            "👋 Welcome to the **Heart Disease Risk Assessment Chatbot**!\n\n"
            "I can help you:\n"
            "- **Assess your heart disease risk** (click 'Start Assessment')\n"
            "- **Answer health questions** about heart disease\n\n"
            "Click **Start Assessment** below to begin, or type any question!"
        )

    session: ChatSession = st.session_state.chat_session

    # Control buttons
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if st.button("▶️ Start Assessment", use_container_width=True):
            reply = session.start_assessment()
            bot_say(reply)
            st.rerun()
    with c2:
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.chat_session  = None
            st.session_state.chat_history  = []
            st.session_state.assessment_done = False
            st.session_state.prediction_result = None
            st.rerun()

    st.markdown("---")

    # Chat display
    chat_container = st.container()
    with chat_container:
        render_chat()

    # Input
    st.markdown("---")
    with st.form(key="chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            user_input = st.text_input(
                "Your response:",
                placeholder="Type your answer or a health question…",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send ➤")

    if submitted and user_input.strip():
        user_say(user_input.strip())

        reply = session.process(user_input.strip())

        if reply == "__PREDICT__":
            # Run prediction
            input_df = prepare_single_input(session.patient)
            result   = predict(model, input_df)
            session.prediction = result
            session.state = State.CHAT
            st.session_state.prediction_result = result
            st.session_state.patient_data      = dict(session.patient)
            st.session_state.assessment_done   = True

            # Generate SHAP explanation
            shap_row   = generate_shap_explanation(input_df, model)
            explanation = build_patient_explanation(shap_row)

            prob_pct = result["probability"] * 100
            color    = "🔴" if result["prediction"] == 1 else "🟢"
            reply = (
                f"## {color} {result['risk_label']}\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| **Probability** | {prob_pct:.1f}% |\n"
                f"| **Confidence** | {result['confidence']} |\n\n"
                f"---\n\n"
                f"### 🔍 Key Factors\n{explanation}\n\n"
                f"---\n\n"
                f"⚕️ **What to do next:**\n"
                + (
                    "Please consult a cardiologist promptly for a proper medical evaluation."
                    if result["prediction"] == 1 else
                    "Continue maintaining a healthy lifestyle. Regular check-ups are still recommended."
                )
                + "\n\n"
                f"You can now ask me questions about your results or heart health in general.\n\n"
                f"👉 Also check the **Prediction Insights** tab for detailed visualizations."
            )

        bot_say(reply)
        st.rerun()

# ── Page: Prediction Insights ──────────────────────────────────────────────────

def page_insights():
    st.markdown("## 📊 Prediction Insights")

    ASSETS = ROOT / "assets"

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Your Prediction", "📈 Model Performance", "🔍 SHAP / Feature Importance", "🧪 EDA"]
    )

    # ── Tab 1: current patient prediction ─────────────────────────────────────
    with tab1:
        if not st.session_state.assessment_done or st.session_state.prediction_result is None:
            st.info("ℹ️ Complete the Risk Assessment chatbot first to see your personalised prediction.")
        else:
            result = st.session_state.prediction_result
            pdata  = st.session_state.patient_data
            prob   = result["probability"]

            # Result banner
            cls = "result-high" if result["prediction"] == 1 else "result-low"
            icon = "⚠️" if result["prediction"] == 1 else "✅"
            st.markdown(
                f'<div class="{cls}"><h1>{icon} {result["risk_label"]}</h1></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])

            with col1:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    title={"text": "Heart Disease Probability (%)", "font": {"size": 16, "color": "white"}},
                    delta={"reference": 50, "increasing": {"color": "#f44336"}, "decreasing": {"color": "#4caf50"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "white"},
                        "bar":  {"color": "#f44336" if prob > 0.5 else "#4caf50"},
                        "bgcolor": "#1a1a2e",
                        "bordercolor": "#333",
                        "steps": [
                            {"range": [0,  30], "color": "#1b5e20"},
                            {"range": [30, 60], "color": "#f57f17"},
                            {"range": [60, 100], "color": "#b71c1c"},
                        ],
                        "threshold": {"value": 50, "line": {"color": "white", "width": 2}},
                    },
                    number={"suffix": "%", "font": {"color": "white"}},
                ))
                fig.update_layout(
                    height=300, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font={"color": "white"},
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Patient data summary
                st.markdown("### 📋 Your Clinical Data")
                decoded = decode_features(pdata)
                df_patient = pd.DataFrame(
                    list(decoded.items()), columns=["Feature", "Your Value"]
                )
                st.dataframe(df_patient, use_container_width=True, height=300)

            # SHAP waterfall for this patient
            st.markdown("### 🔍 Factor Contributions")
            input_df = prepare_single_input(pdata)
            shap_row = generate_shap_explanation(input_df, model) if model else shap_vals

            if shap_row:
                sv_df = pd.DataFrame(
                    [(k, v) for k, v in shap_row.items()],
                    columns=["Feature", "SHAP Value"]
                ).sort_values("SHAP Value")
                sv_df["Color"] = sv_df["SHAP Value"].apply(
                    lambda x: "#f44336" if x > 0 else "#4caf50"
                )
                fig2 = go.Figure(go.Bar(
                    x=sv_df["SHAP Value"], y=sv_df["Feature"],
                    orientation="h",
                    marker_color=sv_df["Color"],
                    text=[f"{v:+.3f}" for v in sv_df["SHAP Value"]],
                    textposition="outside",
                ))
                fig2.update_layout(
                    title="SHAP Values (Red = increases risk, Green = decreases risk)",
                    xaxis_title="SHAP Value", paper_bgcolor="#0f1117",
                    plot_bgcolor="#0f1117", font={"color": "white"},
                    height=450,
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("""
            <div class='disclaimer'>
            ⚠️ This prediction is for educational screening purposes only and 
            does not constitute a medical diagnosis. Please consult a qualified 
            healthcare professional.
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: model performance ───────────────────────────────────────────────
    with tab2:
        st.markdown("### 🏆 Model Comparison")
        if meta.get("comparison"):
            comp = pd.DataFrame(meta["comparison"]).T
            numeric_cols = comp.select_dtypes(include="number").columns
            st.dataframe(
                comp[numeric_cols].style.background_gradient(cmap="YlGn", axis=0),
                use_container_width=True,
            )

        col1, col2 = st.columns(2)
        img_paths = {
            "model_comparison.png": ("Model Comparison", col1),
            "roc_curves.png":       ("ROC Curves",       col2),
            "confusion_matrix.png": ("Confusion Matrix", col1),
        }
        for fname, (title, col) in img_paths.items():
            fp = ASSETS / fname
            if fp.exists():
                with col:
                    st.markdown(f"#### {title}")
                    st.image(str(fp))

    # ── Tab 3: SHAP ───────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🔍 Global Feature Importance (SHAP)")
        for fname, title in [
            ("shap_summary.png",  "SHAP Summary (Bar)"),
            ("shap_beeswarm.png", "SHAP Beeswarm"),
            ("feature_importance.png", "Model Feature Importance"),
        ]:
            fp = ASSETS / fname
            if fp.exists():
                st.markdown(f"#### {title}")
                st.image(str(fp))
                st.markdown("")

        if shap_vals:
            sv_sorted = dict(sorted(shap_vals.items(), key=lambda x: x[1], reverse=True))
            fig = go.Figure(go.Bar(
                x=list(sv_sorted.values()), y=list(sv_sorted.keys()),
                orientation="h", marker_color="#667eea",
                text=[f"{v:.4f}" for v in sv_sorted.values()],
                textposition="outside",
            ))
            fig.update_layout(
                title="Mean |SHAP| Feature Importance",
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font={"color": "white"}, height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: EDA ────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🧪 Exploratory Data Analysis")
        for fname, title in [
            ("target_distribution.png",  "Target Distribution & Class Balance"),
            ("correlation_heatmap.png",  "Correlation Heatmap"),
            ("feature_distributions.png","Feature Distributions"),
            ("outliers.png",             "Outlier Detection"),
        ]:
            fp = ASSETS / fname
            if fp.exists():
                st.markdown(f"#### {title}")
                st.image(str(fp))
                st.markdown("")

# ── Page: Health Education ────────────────────────────────────────────────────

def page_education():
    st.markdown("## 📚 Heart Health Education")

    tab1, tab2, tab3 = st.tabs(["❤️ About Heart Disease", "🛡️ Prevention", "❓ FAQs"])

    with tab1:
        st.markdown("""
### What is Heart Disease?

Heart disease (cardiovascular disease) is an umbrella term for a range of conditions 
affecting the heart and blood vessels. It is the **leading cause of death globally**, 
responsible for approximately **17.9 million deaths per year** (WHO, 2023).

#### Common Types
- **Coronary Artery Disease (CAD)** — plaque buildup in arteries restricts blood flow
- **Heart Failure** — the heart cannot pump enough blood to meet the body's needs
- **Arrhythmia** — irregular heart rhythms
- **Valvular Heart Disease** — damage to one or more heart valves
- **Cardiomyopathy** — disease of the heart muscle

#### Key Risk Factors
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.warning("**Non-modifiable Risk Factors**\n- Age (risk increases with age)\n- Family history of heart disease\n- Gender (men at higher risk at younger ages)\n- Ethnicity")
        with col2:
            st.error("**Modifiable Risk Factors**\n- High blood pressure (hypertension)\n- High cholesterol\n- Smoking\n- Diabetes\n- Obesity\n- Physical inactivity\n- Stress")

    with tab2:
        st.markdown("### 🛡️ Heart Disease Prevention Strategies")
        strategies = {
            "🥦 Healthy Diet": [
                "Follow a Mediterranean-style diet",
                "Increase fruits, vegetables, and whole grains",
                "Reduce saturated fats, trans fats, and sodium",
                "Limit sugar-sweetened beverages and processed foods",
                "Choose lean proteins (fish, beans, poultry)",
            ],
            "🏃 Regular Exercise": [
                "Aim for 150 minutes/week of moderate aerobic activity",
                "Or 75 minutes/week of vigorous aerobic activity",
                "Include muscle-strengthening activities 2+ days/week",
                "Avoid prolonged sitting — take breaks every 30 minutes",
                "Even walking 30 minutes/day significantly reduces risk",
            ],
            "🚭 Quit Smoking": [
                "Smoking doubles the risk of heart disease",
                "Benefits of quitting begin within hours",
                "Risk approaches non-smoker levels within 1-5 years",
                "Seek support: nicotine replacement, counselling, medications",
            ],
            "💊 Manage Medical Conditions": [
                "Monitor and control blood pressure (target < 130/80 mmHg)",
                "Manage cholesterol levels with diet, exercise, and medication",
                "Control blood sugar if diabetic",
                "Take prescribed medications consistently",
            ],
            "😴 Lifestyle": [
                "Get 7-9 hours of quality sleep per night",
                "Manage stress through meditation, yoga, or therapy",
                "Limit alcohol consumption",
                "Maintain a healthy weight (BMI 18.5-24.9)",
                "Schedule regular health check-ups",
            ],
        }
        for title, points in strategies.items():
            with st.expander(title, expanded=False):
                for p in points:
                    st.markdown(f"✅ {p}")

    with tab3:
        st.markdown("### ❓ Frequently Asked Questions")
        faqs = [
            ("Can heart disease be reversed?", "While full reversal is difficult, significant lifestyle changes (diet, exercise, medications) can slow progression and in some cases partially reverse certain forms like CAD."),
            ("At what age should I start worrying about heart disease?", "Risk assessment should start at age 20 with regular cholesterol and blood pressure checks. Those with risk factors should be monitored earlier."),
            ("Are women at lower risk than men?", "Men tend to develop heart disease earlier, but it's the #1 killer of women too. Women often experience atypical symptoms like fatigue, jaw pain, or nausea."),
            ("How accurate is this AI prediction?", f"Our best model achieved ROC-AUC of {meta.get('roc_auc', 0.88):.3f}. However, it's a screening tool only — not a diagnostic instrument."),
            ("What is a normal cholesterol level?", "Total cholesterol < 200 mg/dL is desirable. LDL < 100 mg/dL is optimal. HDL > 60 mg/dL is protective."),
            ("What symptoms need emergency attention?", "Sudden chest pain, shortness of breath, pain radiating to arm/jaw, cold sweats, dizziness, or sudden numbness — call emergency services immediately."),
        ]
        for q, a in faqs:
            with st.expander(f"❓ {q}"):
                st.write(a)

# ── Page: About ───────────────────────────────────────────────────────────────

def page_about():
    st.markdown("## ℹ️ About This Project")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### 🎯 Project Overview
**AI-Powered Medical Chatbot for Heart Disease Prediction** is an end-to-end 
machine learning project that demonstrates the practical application of AI 
in healthcare screening.

### 🗂️ Dataset
- **Source:** Cleveland Heart Disease Dataset
- **Records:** 302 (after de-duplication from 1,025)
- **Features:** 13 clinical parameters
- **Target:** Binary classification (Heart Disease / No Disease)
- **Balance:** ~54% positive, ~46% negative
        """)

    with col2:
        st.markdown("""
### 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Frontend | Streamlit |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Data | Pandas, NumPy |
| Model Persistence | Joblib |
| Testing | Pytest |
        """)

    st.markdown("---")
    st.markdown("### 🤖 ML Pipeline Summary")
    steps = {
        "1. EDA": "Distribution analysis, correlation heatmap, outlier detection, class balance check.",
        "2. Preprocessing": "De-duplication, stratified train/test split (80:20), StandardScaler normalisation.",
        "3. Model Training": "6 models trained: Logistic Regression, Random Forest, Decision Tree, XGBoost, SVM, KNN.",
        "4. Evaluation": "Accuracy, Precision, Recall, F1, ROC-AUC, 5-fold cross-validation.",
        "5. Hyperparameter Tuning": f"GridSearchCV on best model ({meta.get('best_model','N/A')}). CV AUC ≈ 0.91.",
        "6. Explainability": "SHAP KernelExplainer for global + per-patient feature attribution.",
    }
    for step, desc in steps.items():
        st.markdown(f"**{step}:** {desc}")

    st.markdown("---")
    st.markdown("### 📁 Project Structure")
    st.code("""
heart_disease_project/
├── data/
│   └── heart.csv
├── models/
│   ├── train_model.py      # Training pipeline
│   ├── predict.py          # Inference module
│   └── best_model.pkl      # Saved model
├── preprocessing/
│   └── preprocessor.py     # Feature engineering
├── chatbot/
│   └── chatbot.py          # Conversation engine
├── app/
│   └── app.py              # Streamlit frontend
├── assets/                 # Saved plots & SHAP values
├── tests/
│   └── test_pipeline.py    # Unit tests
├── requirements.txt
├── Dockerfile
└── README.md
    """, language="")

# ── Router ─────────────────────────────────────────────────────────────────────

page = st.session_state.current_page

if page == "🏠 Home":
    page_home()
elif page == "🤖 Risk Assessment Chatbot":
    page_chatbot()
elif page == "📊 Prediction Insights":
    page_insights()
elif page == "📚 Health Education":
    page_education()
elif page == "ℹ️ About":
    page_about()
