import importlib.metadata
import os
import pickle

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from optuna.visualization.matplotlib import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("water_potability.csv")

x_train, x_test, y_train, y_test = train_test_split(
    df.drop("Potability", axis=1),
    df["Potability"],
    test_size=0.2,
    random_state=20,
)


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")
    return best_model


def save_plots(study):
    os.makedirs("plots", exist_ok=True)
    plot_configs = [
        ("optimization_history.png", plot_optimization_history),
        ("param_importances.png", plot_param_importances),
        ("slice.png", plot_slice),
        ("contour.png", plot_contour),
    ]
    for filename, plot_fn in plot_configs:
        plot_fn(study)
        plt.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        print(f"Guardado: plots/{filename}")
        plt.close("all")


def generate_requirements():
    packages = ["mlflow", "optuna", "xgboost", "scikit-learn", "pandas", "numpy"]
    with open("requirements.txt", "w") as f:
        for pkg in packages:
            version = importlib.metadata.version(pkg)
            f.write(f"{pkg}=={version}\n")
    print("Guardado: requirements.txt")


def optimize_model():
    experiment_name = "xgboost_potabilidad_agua"
    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(experiment_name)
    if existing and existing.lifecycle_stage == "deleted":
        client.restore_experiment(existing.experiment_id)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        }
        run_name = (
            f"XGBoost n_est={params['n_estimators']} lr={params['learning_rate']:.4f} depth={params['max_depth']}"
        )

        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            xg = XGBClassifier(**params, random_state=20)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=20)
            scores = cross_val_score(xg, x_train, y_train, cv=cv, scoring="f1_macro")
            f1 = scores.mean()

            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", f1)

            xg.fit(x_train, y_train)
            mlflow.sklearn.log_model(xg, "model")

        return f1

    storage = optuna.storages.RDBStorage("sqlite:///optuna_lab8.db")
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="xgboost_potabilidad_agua",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10)

    best_model = get_best_model(experiment_id)

    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"Mejor trial: {study.best_trial.number}")
    print(f"Mejor F1: {study.best_value:.4f}")
    print(f"Mejores hiperparámetros: {study.best_params}")

    print("\nGenerando plots...")
    save_plots(study)

    print("\nGenerando requirements.txt...")
    generate_requirements()

    return best_model


if __name__ == "__main__":
    optimize_model()
