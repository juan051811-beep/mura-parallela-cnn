from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def obtener_modelos_grid():
    return {
        "SVM": {
            "modelo": SVC(),
            "parametros": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            }
        },
        "Random Forest": {
            "modelo": RandomForestClassifier(random_state=42),
            "parametros": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None]
            }
        },
        "Logistic Regression": {
            "modelo": LogisticRegression(max_iter=2000, random_state=42),
            "parametros": {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs"]
            }
        }
    }
