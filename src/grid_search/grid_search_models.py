from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def crear_modelo(nombre, parametros):
    if nombre == "SVM":
        return SVC(**parametros)

    if nombre == "Random Forest":
        return RandomForestClassifier(random_state=42, **parametros)

    if nombre == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=42, **parametros)

    raise ValueError(f"Modelo no reconocido: {nombre}")


def obtener_parametros_grid():
    return {
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None]
        },
        "Logistic Regression": {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs"]
        }
    }
