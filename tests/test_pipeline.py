"""
Heart Disease Prediction — Unit Tests
Run: pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessor import (
    FEATURE_COLS,
    TARGET_COL,
    build_pipeline,
    decode_features,
    load_data,
    prepare_single_input,
    split_data,
)
from chatbot.chatbot import ChatSession, State

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return load_data(ROOT / "data" / "heart.csv")


@pytest.fixture(scope="module")
def sample_patient() -> dict:
    return {
        "age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2,
    }


# ── Data loading ───────────────────────────────────────────────────────────────

class TestDataLoading:
    def test_load_returns_dataframe(self, sample_df):
        assert isinstance(sample_df, pd.DataFrame)

    def test_columns_present(self, sample_df):
        for col in FEATURE_COLS + [TARGET_COL]:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_no_missing_after_load(self, sample_df):
        assert sample_df[FEATURE_COLS].isnull().sum().sum() == 0

    def test_target_is_binary(self, sample_df):
        assert set(sample_df[TARGET_COL].unique()).issubset({0, 1})

    def test_minimum_rows(self, sample_df):
        assert len(sample_df) >= 100


# ── Preprocessing ──────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_split_sizes(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df)
        n = len(sample_df)
        assert abs(len(X_test) / n - 0.2) < 0.05

    def test_feature_columns_preserved(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df)
        assert list(X_train.columns) == FEATURE_COLS

    def test_no_data_leakage(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df)
        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_prepare_single_input_shape(self, sample_patient):
        df = prepare_single_input(sample_patient)
        assert df.shape == (1, len(FEATURE_COLS))
        assert list(df.columns) == FEATURE_COLS

    def test_decode_features_returns_dict(self, sample_patient):
        result = decode_features(sample_patient)
        assert isinstance(result, dict)
        assert len(result) == len(FEATURE_COLS)

    def test_decode_features_values_are_strings(self, sample_patient):
        result = decode_features(sample_patient)
        for v in result.values():
            assert isinstance(v, str)


# ── Pipeline ───────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_has_scaler_and_model(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        pipe = build_pipeline(LogisticRegression())
        assert "scaler" in pipe.named_steps
        assert "model"  in pipe.named_steps

    def test_pipeline_fits_and_predicts(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        X_train, X_test, y_train, y_test = split_data(sample_df)
        pipe = build_pipeline(LogisticRegression(max_iter=1000))
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})

    def test_pipeline_probabilities_sum_to_one(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        X_train, X_test, y_train, y_test = split_data(sample_df)
        pipe = build_pipeline(LogisticRegression(max_iter=1000))
        pipe.fit(X_train, y_train)
        probas = pipe.predict_proba(X_test)
        assert np.allclose(probas.sum(axis=1), 1.0)


# ── Prediction module ──────────────────────────────────────────────────────────

class TestPrediction:
    @pytest.fixture(scope="class")
    def trained_model(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        X_train, X_test, y_train, y_test = split_data(sample_df)
        pipe = build_pipeline(LogisticRegression(max_iter=1000))
        pipe.fit(X_train, y_train)
        return pipe

    def test_predict_returns_dict(self, trained_model, sample_patient):
        from models.predict import predict
        input_df = prepare_single_input(sample_patient)
        result   = predict(trained_model, input_df)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, trained_model, sample_patient):
        from models.predict import predict
        input_df = prepare_single_input(sample_patient)
        result   = predict(trained_model, input_df)
        for key in ("prediction", "probability", "confidence", "risk_label"):
            assert key in result

    def test_predict_probability_in_range(self, trained_model, sample_patient):
        from models.predict import predict
        input_df = prepare_single_input(sample_patient)
        result   = predict(trained_model, input_df)
        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_binary_output(self, trained_model, sample_patient):
        from models.predict import predict
        input_df = prepare_single_input(sample_patient)
        result   = predict(trained_model, input_df)
        assert result["prediction"] in (0, 1)

    def test_confidence_levels(self, trained_model, sample_df):
        from models.predict import predict
        X_train, X_test, y_train, y_test = split_data(sample_df)
        pipe = trained_model
        for _, row in X_test.head(10).iterrows():
            input_df = pd.DataFrame([row])
            result   = predict(pipe, input_df)
            assert result["confidence"] in ("High", "Moderate", "Low")


# ── Chatbot ────────────────────────────────────────────────────────────────────

class TestChatbot:
    def test_initial_state_is_welcome(self):
        session = ChatSession()
        assert session.state == State.WELCOME

    def test_start_assessment_changes_state(self):
        session = ChatSession()
        session.start_assessment()
        assert session.state == State.COLLECT_AGE

    def test_valid_age_advances_state(self):
        session = ChatSession()
        session.start_assessment()
        reply = session.process("55")
        assert session.state == State.COLLECT_SEX
        assert "PREDICT" not in reply

    def test_invalid_age_stays_in_state(self):
        session = ChatSession()
        session.start_assessment()
        reply = session.process("abc")
        assert session.state == State.COLLECT_AGE
        assert "valid" in reply.lower() or "⚠️" in reply

    def test_out_of_range_age_rejected(self):
        session = ChatSession()
        session.start_assessment()
        reply = session.process("200")
        assert session.state == State.COLLECT_AGE

    def test_full_collection_emits_predict_signal(self):
        """Walk through all 13 questions and expect __PREDICT__."""
        session = ChatSession()
        session.start_assessment()
        answers = ["55", "1", "2", "120", "200", "0", "0", "150", "0", "1.5", "1", "0", "2"]
        last = None
        for ans in answers:
            last = session.process(ans)
        assert last == "__PREDICT__"

    def test_patient_data_populated_after_collection(self):
        session = ChatSession()
        session.start_assessment()
        answers = ["55", "1", "2", "120", "200", "0", "0", "150", "0", "1.5", "1", "0", "2"]
        for ans in answers:
            session.process(ans)
        assert len(session.patient) == 13

    def test_health_question_answering(self):
        session = ChatSession()
        session.state = State.CHAT
        reply = session.process("what is heart disease")
        assert len(reply) > 20
        assert "heart" in reply.lower()

    def test_unknown_question_gives_fallback(self):
        session = ChatSession()
        session.state = State.CHAT
        reply = session.process("xyzzy qwerty")
        assert len(reply) > 10

    def test_reset_clears_patient_data(self):
        session = ChatSession()
        session.start_assessment()
        session.process("55")
        session.reset()
        assert session.patient == {}
        assert session.prediction is None

    def test_add_message_history(self):
        session = ChatSession()
        session.add_message("user", "hello")
        session.add_message("bot",  "hi!")
        assert len(session.history) == 2
        assert session.history[0]["role"] == "user"


# ── Input validation edge cases ────────────────────────────────────────────────

class TestInputValidation:
    @pytest.mark.parametrize("field,value,expected_ok", [
        ("age",      "55",   True),
        ("age",      "-1",   False),
        ("age",      "200",  False),
        ("sex",      "1",    True),
        ("sex",      "2",    False),
        ("sex",      "abc",  False),
        ("cp",       "3",    True),
        ("cp",       "5",    False),
        ("trestbps", "120",  True),
        ("trestbps", "20",   False),
        ("oldpeak",  "1.5",  True),
        ("oldpeak",  "-1",   False),
        ("ca",       "4",    True),
        ("ca",       "5",    False),
    ])
    def test_validate_field(self, field, value, expected_ok):
        session = ChatSession()
        ok, _, _ = session._validate(field, value)
        assert ok == expected_ok, f"Field {field}={value}: expected ok={expected_ok}, got {ok}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
