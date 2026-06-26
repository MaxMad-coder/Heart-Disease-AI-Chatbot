"""
Heart Disease AI Chatbot — Main Entry Point
Usage:
    python main.py             # Train model + launch Streamlit app
    python main.py --train     # Train model only
    python main.py --app       # Launch app only (model must exist)
    python main.py --test      # Run unit tests
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def train():
    print("🔬 Training ML pipeline …")
    result = subprocess.run(
        [sys.executable, str(ROOT / "models" / "train_model.py")],
        check=False,
    )
    if result.returncode != 0:
        print("❌ Training failed.")
        sys.exit(1)
    print("✅ Model training complete.")


def run_app():
    model_path = ROOT / "models" / "best_model.pkl"
    if not model_path.exists():
        print("⚠️  Model not found. Training first …")
        train()

    print("🚀 Launching Streamlit app at http://localhost:8501")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(ROOT / "app" / "app.py"),
            "--server.port=8501",
            "--browser.gatherUsageStats=false",
        ],
        check=False,
    )


def run_tests():
    print("🧪 Running unit tests …")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-v", "--tb=short"],
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease AI Chatbot")
    parser.add_argument("--train", action="store_true", help="Train model only")
    parser.add_argument("--app",   action="store_true", help="Launch app only")
    parser.add_argument("--test",  action="store_true", help="Run unit tests")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.app:
        run_app()
    elif args.test:
        run_tests()
    else:
        # Default: train then launch
        train()
        run_app()
