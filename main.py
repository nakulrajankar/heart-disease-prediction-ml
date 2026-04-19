from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_dataset, preprocess_data
from src.train_model import build_models, train_all
from src.evaluate import evaluate_model
from src.utils import save_model

DATA_PATH = "data/heart.csv"


def main():
    print("📥 Loading Data...")
    df = load_dataset(DATA_PATH)

    print("⚙️ Preprocessing...")
    X, y, scaler = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🤖 Training Models...")
    models = build_models()
    trained_models = train_all(models, X_train, y_train)

    print("\n📊 Results:\n")

    best_model = None
    best_score = 0

    for name, model in trained_models.items():
        acc, report = evaluate_model(model, X_test, y_test)

        print(f"🔹 {name.upper()} Accuracy: {acc:.4f}")
        print(report)

        if acc > best_score:
            best_score = acc
            best_model = model

    # 👇 ADD HERE (IMPORTANT)
    print(f"\n🏆 Best Model Accuracy: {best_score:.4f}")
    save_model(best_model, "models/trained_model.pkl")
    print("✅ Model Saved Successfully!")


if __name__ == "__main__":
    main()