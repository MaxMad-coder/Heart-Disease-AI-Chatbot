# 💓 AI-Powered Medical Chatbot for Heart Disease Prediction

An end-to-end machine learning project featuring a conversational chatbot interface for heart disease risk assessment, built with Python, Scikit-learn, XGBoost, SHAP, and Streamlit.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🤖 AI Chatbot | Conversational interface collecting 13 clinical parameters |
| 🏆 Multi-Model ML | 6 classifiers trained and compared (LR, RF, DT, XGBoost, SVM, KNN) |
| 🔍 Explainable AI | SHAP values explain every individual prediction |
| 📊 Rich Visualisations | ROC curves, confusion matrices, feature importance, EDA charts |
| 📚 Health Education | Built-in knowledge base for heart health Q&A |
| 🔒 Privacy First | No patient data stored — all processing is in-memory |

---

## 📊 Model Performance

| Model | Accuracy | F1 | ROC-AUC | CV-AUC |
|-------|----------|-----|---------|--------|
| **Logistic Regression** ⭐ | 0.803 | 0.824 | 0.881 | **0.912** |
| Random Forest | 0.754 | 0.776 | 0.859 | 0.904 |
| SVM | 0.771 | 0.794 | 0.842 | 0.898 |
| KNN | 0.787 | 0.812 | 0.838 | 0.877 |
| XGBoost | 0.721 | 0.746 | 0.832 | 0.903 |
| Decision Tree | 0.803 | 0.818 | 0.802 | 0.732 |

*Best model selected by ROC-AUC after 5-fold CV.*

---

## 🗂️ Project Structure

```
heart_disease_project/
├── data/
│   └── heart.csv                  # Cleveland Heart Disease Dataset
├── models/
│   ├── train_model.py             # Full training pipeline (EDA → SHAP)
│   ├── predict.py                 # Single-patient inference module
│   └── best_model.pkl             # Saved trained pipeline (auto-generated)
├── preprocessing/
│   └── preprocessor.py            # Feature engineering & validation
├── chatbot/
│   └── chatbot.py                 # State-machine conversation engine
├── app/
│   └── app.py                     # Streamlit multi-page frontend
├── assets/                        # Auto-generated plots & SHAP values
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── outliers.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   ├── shap_beeswarm.png
│   └── shap_values.json
├── tests/
│   └── test_pipeline.py           # 44 unit tests (pytest)
├── requirements.txt
├── Dockerfile
├── main.py                        # Unified entry point
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd heart_disease_project
```

### 2. Place the dataset

```bash
cp heart.csv data/heart.csv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python main.py --train
# or directly:
python models/train_model.py
```

This will:
- Run EDA and save all plots to `assets/`
- Train 6 ML models and compare them
- Tune the best model with GridSearchCV
- Compute SHAP values
- Save the model to `models/best_model.pkl`

### 5. Launch the app

```bash
python main.py --app
# or directly:
streamlit run app/app.py
```

Open http://localhost:8501 in your browser.

### 6. Run tests

```bash
python main.py --test
# or:
pytest tests/ -v
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t heart-disease-ai .

# Run
docker run -p 8501:8501 heart-disease-ai
```

---

## ☁️ Cloud Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, set **Main file path** to `app/app.py`
4. Add a `requirements.txt` at the root
5. Deploy!

> **Note:** Add a pre-run hook or startup script to train the model before the app loads.

### Hugging Face Spaces

1. Create a new Space (Streamlit SDK)
2. Upload all project files
3. The Space auto-installs `requirements.txt` and runs `app/app.py`

### Render

1. Push to GitHub
2. New Web Service → connect repo
3. **Build Command:** `pip install -r requirements.txt && python models/train_model.py`
4. **Start Command:** `streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0`

---

## 📋 Dataset

| Property | Value |
|----------|-------|
| Source | Cleveland Heart Disease Dataset (UCI ML Repository) |
| Records | 302 unique (after de-duplication from 1,025) |
| Features | 13 clinical parameters |
| Target | Binary: 0 = No Disease, 1 = Heart Disease |
| Missing values | None |

### Feature Descriptions

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | Sex (1=Male, 0=Female) |
| `cp` | Chest pain type (0-3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1=Yes) |
| `restecg` | Resting ECG results (0-2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1=Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (0-2) |
| `ca` | Number of major vessels (0-4) |
| `thal` | Thalassemia (0=Normal, 1=Fixed, 2=Reversible) |

---

## 🏥 Medical Disclaimer

> ⚠️ **This application is for educational and research purposes only.**
> It does not constitute medical advice, diagnosis, or treatment.
> Always consult a qualified healthcare professional for medical decisions.
> The authors are not responsible for any medical decisions made based on this tool's output.

---

## 🛠️ Technologies

- **Python 3.11** — Core language
- **Scikit-learn** — ML pipeline, preprocessing, model evaluation
- **XGBoost** — Gradient boosting classifier
- **SHAP** — Model explainability
- **Streamlit** — Web frontend
- **Plotly** — Interactive charts
- **Matplotlib / Seaborn** — Static visualisations
- **Joblib** — Model serialisation
- **Pytest** — Unit testing (44 tests, 100% pass rate)
