"""
Heart Disease Medical Chatbot — Conversation Engine
Manages multi-turn state: data collection → prediction → Q&A.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

# ── Conversation states ────────────────────────────────────────────────────────

class State(Enum):
    WELCOME          = auto()
    COLLECT_AGE      = auto()
    COLLECT_SEX      = auto()
    COLLECT_CP       = auto()
    COLLECT_TRESTBPS = auto()
    COLLECT_CHOL     = auto()
    COLLECT_FBS      = auto()
    COLLECT_RESTECG  = auto()
    COLLECT_THALACH  = auto()
    COLLECT_EXANG    = auto()
    COLLECT_OLDPEAK  = auto()
    COLLECT_SLOPE    = auto()
    COLLECT_CA       = auto()
    COLLECT_THAL     = auto()
    PREDICT          = auto()
    CHAT             = auto()

# ── Prompts / questions ────────────────────────────────────────────────────────

QUESTIONS: dict[str, dict] = {
    "age": {
        "prompt": "🔢 What is your **age** in years? (e.g., 54)",
        "type": "int", "min": 1, "max": 120,
    },
    "sex": {
        "prompt": (
            "⚧ What is your **biological sex**?\n"
            "- Type `1` for **Male**\n"
            "- Type `0` for **Female**"
        ),
        "type": "int", "options": [0, 1],
    },
    "cp": {
        "prompt": (
            "💔 What type of **chest pain** do you experience?\n"
            "- `0` — Typical Angina\n"
            "- `1` — Atypical Angina\n"
            "- `2` — Non-anginal Pain\n"
            "- `3` — Asymptomatic (no pain)"
        ),
        "type": "int", "options": [0, 1, 2, 3],
    },
    "trestbps": {
        "prompt": "🩺 What is your **resting blood pressure** (mm Hg)? (e.g., 120)",
        "type": "int", "min": 50, "max": 250,
    },
    "chol": {
        "prompt": "🧪 What is your **serum cholesterol** level (mg/dl)? (e.g., 200)",
        "type": "int", "min": 50, "max": 600,
    },
    "fbs": {
        "prompt": (
            "🍬 Is your **fasting blood sugar** > 120 mg/dl?\n"
            "- `1` — Yes\n"
            "- `0` — No"
        ),
        "type": "int", "options": [0, 1],
    },
    "restecg": {
        "prompt": (
            "📉 What are your **resting ECG results**?\n"
            "- `0` — Normal\n"
            "- `1` — ST-T wave abnormality\n"
            "- `2` — Left ventricular hypertrophy"
        ),
        "type": "int", "options": [0, 1, 2],
    },
    "thalach": {
        "prompt": "❤️ What is your **maximum heart rate** achieved during exercise? (e.g., 150)",
        "type": "int", "min": 60, "max": 250,
    },
    "exang": {
        "prompt": (
            "🚶 Do you experience **exercise-induced angina** (chest pain during exercise)?\n"
            "- `1` — Yes\n"
            "- `0` — No"
        ),
        "type": "int", "options": [0, 1],
    },
    "oldpeak": {
        "prompt": (
            "📊 What is your **ST depression** value induced by exercise "
            "relative to rest? (e.g., 1.5; use 0.0 if none)"
        ),
        "type": "float", "min": 0.0, "max": 10.0,
    },
    "slope": {
        "prompt": (
            "📈 What is the **slope of the peak exercise ST segment**?\n"
            "- `0` — Upsloping\n"
            "- `1` — Flat\n"
            "- `2` — Downsloping"
        ),
        "type": "int", "options": [0, 1, 2],
    },
    "ca": {
        "prompt": (
            "🫀 How many **major vessels** (0-4) are colored by fluoroscopy?\n"
            "Enter a number from 0 to 4."
        ),
        "type": "int", "options": [0, 1, 2, 3, 4],
    },
    "thal": {
        "prompt": (
            "🧬 What is your **thalassemia** status?\n"
            "- `0` — Normal\n"
            "- `1` — Fixed Defect\n"
            "- `2` — Reversible Defect\n"
            "- `3` — Reversible Defect (severe)"
        ),
        "type": "int", "options": [0, 1, 2, 3],
    },
}

STATE_ORDER = [
    State.COLLECT_AGE, State.COLLECT_SEX, State.COLLECT_CP,
    State.COLLECT_TRESTBPS, State.COLLECT_CHOL, State.COLLECT_FBS,
    State.COLLECT_RESTECG, State.COLLECT_THALACH, State.COLLECT_EXANG,
    State.COLLECT_OLDPEAK, State.COLLECT_SLOPE, State.COLLECT_CA,
    State.COLLECT_THAL,
]
STATE_TO_FIELD = {
    State.COLLECT_AGE:      "age",
    State.COLLECT_SEX:      "sex",
    State.COLLECT_CP:       "cp",
    State.COLLECT_TRESTBPS: "trestbps",
    State.COLLECT_CHOL:     "chol",
    State.COLLECT_FBS:      "fbs",
    State.COLLECT_RESTECG:  "restecg",
    State.COLLECT_THALACH:  "thalach",
    State.COLLECT_EXANG:    "exang",
    State.COLLECT_OLDPEAK:  "oldpeak",
    State.COLLECT_SLOPE:    "slope",
    State.COLLECT_CA:       "ca",
    State.COLLECT_THAL:     "thal",
}

# ── Knowledge base for health Q&A ─────────────────────────────────────────────

HEALTH_KB: list[dict] = [
    {
        "patterns": ["what is heart disease", "heart disease definition", "explain heart disease"],
        "response": (
            "❤️ **Heart disease** (cardiovascular disease) refers to a range of conditions "
            "affecting the heart and blood vessels. The most common is **coronary artery disease** "
            "(CAD), where plaque builds up in the arteries reducing blood flow to the heart. "
            "Others include heart failure, arrhythmia, and valvular disease."
        ),
    },
    {
        "patterns": ["symptoms", "signs", "warning signs", "feel"],
        "response": (
            "⚠️ **Common heart disease symptoms** include:\n"
            "- Chest pain or pressure (angina)\n"
            "- Shortness of breath\n"
            "- Rapid or irregular heartbeat\n"
            "- Fatigue or weakness\n"
            "- Dizziness or fainting\n"
            "- Swelling in legs, ankles, or feet\n\n"
            "⚠️ **If you experience sudden severe chest pain**, call emergency services immediately."
        ),
    },
    {
        "patterns": ["reduce risk", "prevent", "prevention", "lower risk", "protect", "lifestyle"],
        "response": (
            "✅ **Ways to reduce your heart disease risk:**\n"
            "- 🥦 Eat a heart-healthy diet (low saturated fats, more fruits & vegetables)\n"
            "- 🏃 Exercise regularly (150 min/week moderate activity)\n"
            "- 🚭 Quit smoking and avoid secondhand smoke\n"
            "- 🍷 Limit alcohol consumption\n"
            "- ⚖️ Maintain a healthy weight\n"
            "- 💊 Manage blood pressure, cholesterol, and diabetes\n"
            "- 😴 Get 7-9 hours of sleep\n"
            "- 🧘 Manage stress (meditation, yoga)"
        ),
    },
    {
        "patterns": ["cardiologist", "doctor", "consult", "see a doctor", "specialist"],
        "response": (
            "🏥 **You should consult a cardiologist if you:**\n"
            "- Experience chest pain, shortness of breath, or palpitations\n"
            "- Have a family history of heart disease\n"
            "- Have high blood pressure, high cholesterol, or diabetes\n"
            "- Are over 40 with multiple risk factors\n"
            "- Received a high-risk prediction from this tool\n\n"
            "🔴 Always seek professional medical evaluation for proper diagnosis and treatment."
        ),
    },
    {
        "patterns": ["cholesterol", "chol", "ldl", "hdl"],
        "response": (
            "🧪 **About Cholesterol:**\n"
            "- **LDL** ('bad' cholesterol) > 130 mg/dL is a risk factor\n"
            "- **HDL** ('good' cholesterol) > 60 mg/dL is protective\n"
            "- **Total cholesterol** ideally < 200 mg/dL\n"
            "Diet, exercise, and medications (statins) help manage cholesterol."
        ),
    },
    {
        "patterns": ["blood pressure", "hypertension", "bp", "trestbps"],
        "response": (
            "🩺 **Blood Pressure Guide:**\n"
            "- Normal: < 120/80 mm Hg\n"
            "- Elevated: 120-129 / < 80\n"
            "- Stage 1 Hypertension: 130-139 / 80-89\n"
            "- Stage 2 Hypertension: ≥ 140 / ≥ 90\n\n"
            "Uncontrolled hypertension significantly increases heart disease risk."
        ),
    },
    {
        "patterns": ["chest pain", "angina", "cp"],
        "response": (
            "💔 **Chest Pain Types:**\n"
            "- **Typical Angina**: Classic squeezing/pressure, often with exertion\n"
            "- **Atypical Angina**: Less typical presentation, may radiate to arm or jaw\n"
            "- **Non-anginal**: Chest discomfort unrelated to heart\n"
            "- **Asymptomatic**: No chest pain (can still have heart disease)\n\n"
            "⚠️ Any new or worsening chest pain warrants immediate medical evaluation."
        ),
    },
    {
        "patterns": ["st depression", "oldpeak", "ecg", "ekg", "electrocardiogram"],
        "response": (
            "📉 **ST Depression (oldpeak):**\n"
            "ST depression on an ECG during exercise stress testing indicates reduced "
            "blood flow to the heart muscle. Values > 2.0 mm are generally considered "
            "significant and may indicate coronary artery disease."
        ),
    },
    {
        "patterns": ["thalassemia", "thal"],
        "response": (
            "🧬 **Thalassemia & Heart Disease:**\n"
            "Thalassemia is a blood disorder affecting hemoglobin. In heart disease prediction:\n"
            "- **Normal**: No thalassemia abnormality\n"
            "- **Fixed Defect**: Permanent defect in blood flow\n"
            "- **Reversible Defect**: Defect only during stress (higher risk indicator)"
        ),
    },
    {
        "patterns": ["exercise", "workout", "physical activity"],
        "response": (
            "🏃 **Exercise & Heart Health:**\n"
            "Regular physical activity is one of the most effective ways to protect your heart:\n"
            "- Aim for **150 min/week** of moderate aerobic activity\n"
            "- Or **75 min/week** of vigorous activity\n"
            "- Include strength training 2+ days/week\n"
            "- Even brisk walking for 30 min/day makes a significant difference\n\n"
            "Always consult your doctor before starting a new exercise program if you have heart concerns."
        ),
    },
    {
        "patterns": ["diet", "food", "eat", "nutrition"],
        "response": (
            "🥦 **Heart-Healthy Diet Tips:**\n"
            "- Increase fruits, vegetables, whole grains, legumes\n"
            "- Choose lean proteins (fish, poultry, beans)\n"
            "- Limit saturated fats, trans fats, and processed foods\n"
            "- Reduce sodium intake (< 2,300 mg/day)\n"
            "- The **Mediterranean diet** is highly recommended for heart health\n"
            "- Limit sugary beverages and alcohol"
        ),
    },
    {
        "patterns": ["accuracy", "reliable", "trust", "how accurate", "model"],
        "response": (
            "📊 **About This Prediction Model:**\n"
            "This tool uses a machine learning model trained on clinical heart disease data. "
            "It is intended as an **educational screening tool** only.\n\n"
            "⚠️ **Important limitations:**\n"
            "- ML models have inherent uncertainty\n"
            "- This does NOT replace clinical diagnosis\n"
            "- A negative result does not guarantee absence of heart disease\n"
            "- Always consult a qualified healthcare professional"
        ),
    },
    {
        "patterns": ["hello", "hi", "hey", "greet"],
        "response": (
            "👋 Hello! I'm your **Heart Health Assistant**.\n"
            "I can help you:\n"
            "- Assess your heart disease risk (click 'Start Assessment' tab)\n"
            "- Answer questions about heart health\n"
            "- Explain your prediction results\n\n"
            "What would you like to know?"
        ),
    },
    {
        "patterns": ["thank", "thanks", "appreciate"],
        "response": (
            "🙏 You're welcome! Remember, your heart health is important. "
            "Please consult a healthcare professional for personalized medical advice. Stay healthy! ❤️"
        ),
    },
]


# ── Chatbot class ──────────────────────────────────────────────────────────────

@dataclass
class ChatSession:
    state:      State         = State.WELCOME
    patient:    dict          = field(default_factory=dict)
    prediction: dict | None  = None
    history:    list[dict]   = field(default_factory=list)

    def reset(self) -> None:
        self.state      = State.COLLECT_AGE
        self.patient    = {}
        self.prediction = None

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    # ── input validation ───────────────────────────────────────────────────────

    def _validate(self, field_key: str, raw: str) -> tuple[bool, Any, str]:
        spec = QUESTIONS[field_key]
        raw  = raw.strip()
        try:
            if spec["type"] == "int":
                val = int(float(raw))
            else:
                val = float(raw)
        except ValueError:
            return False, None, f"⚠️ Please enter a valid number."

        if "options" in spec and val not in spec["options"]:
            opts = ", ".join(str(o) for o in spec["options"])
            return False, None, f"⚠️ Please enter one of: {opts}"
        if "min" in spec and val < spec["min"]:
            return False, None, f"⚠️ Value must be ≥ {spec['min']}"
        if "max" in spec and val > spec["max"]:
            return False, None, f"⚠️ Value must be ≤ {spec['max']}"
        return True, val, ""

    # ── state machine step ─────────────────────────────────────────────────────

    def process(self, user_input: str) -> str:
        """Advance the state machine and return a bot reply."""

        # ── free-form Q&A after prediction ────────────────────────────────────
        if self.state == State.CHAT:
            return self._answer_health_question(user_input)

        # ── collecting patient data ────────────────────────────────────────────
        if self.state in STATE_TO_FIELD:
            field_key = STATE_TO_FIELD[self.state]
            ok, val, err = self._validate(field_key, user_input)
            if not ok:
                return err + f"\n\n{QUESTIONS[field_key]['prompt']}"

            self.patient[field_key] = val
            next_state = self._next_state()
            self.state = next_state

            if self.state == State.PREDICT:
                return "__PREDICT__"  # signal to app layer to run model
            else:
                return QUESTIONS[STATE_TO_FIELD[self.state]]["prompt"]

        return "Type your question or click 'Start Assessment' to begin."

    def _next_state(self) -> State:
        idx = STATE_ORDER.index(self.state)
        if idx + 1 < len(STATE_ORDER):
            return STATE_ORDER[idx + 1]
        return State.PREDICT

    def start_assessment(self) -> str:
        self.reset()
        self.state = State.COLLECT_AGE
        return (
            "🏥 **Heart Disease Risk Assessment**\n\n"
            "I'll ask you 13 clinical questions. Please answer as accurately as possible.\n"
            "Your data is processed locally and not stored.\n\n"
            "---\n\n"
            + QUESTIONS["age"]["prompt"]
        )

    def get_first_question(self) -> str:
        return QUESTIONS["age"]["prompt"]

    # ── health Q&A ─────────────────────────────────────────────────────────────

    def _answer_health_question(self, query: str) -> str:
        q = query.lower()
        for item in HEALTH_KB:
            if any(p in q for p in item["patterns"]):
                return item["response"]

        # Prediction context
        if self.prediction and any(k in q for k in ["result", "prediction", "risk", "score", "probability"]):
            p = self.prediction
            return (
                f"📊 **Your Prediction Result:**\n"
                f"- **Risk Level:** {p['risk_label']}\n"
                f"- **Probability:** {p['probability']*100:.1f}%\n"
                f"- **Confidence:** {p['confidence']}\n\n"
                f"Please consult a healthcare professional for medical advice."
            )

        return (
            "🤔 I'm not sure about that specific question. "
            "Here are some topics I can help with:\n"
            "- *What is heart disease?*\n"
            "- *What are the symptoms?*\n"
            "- *How can I reduce my risk?*\n"
            "- *What do my cholesterol levels mean?*\n"
            "- *Should I see a cardiologist?*\n\n"
            "For personalised medical advice, please consult a qualified healthcare professional."
        )
