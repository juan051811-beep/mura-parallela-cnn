import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


X = np.load("features_sequential.npy") #cargamos las features
y = np.load("labels_sequential.npy")

print(f"Dataset completo: {X.shape[0]} imágenes")

X_train, X_test, y_train, y_test = train_test_split( #dividimos 80-20
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]} imágenes")
print(f"Prueba: {X_test.shape[0]} imágenes")


modelo = LogisticRegression( #entrenamos con el mejor modelo dado por grid search
    C=0.1,
    solver="lbfgs",
    max_iter=1000
)

print("\nEntrenando modelo...")
modelo.fit(X_train, y_train)

print("Realizando predicciones...")
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) #metricas
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nResultados:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

reporte = classification_report(y_test, y_pred) #guardamos el reporte de clasificacion

with open("classification_report_lr.txt", "w", encoding="utf-8") as f:
    f.write(reporte)

print("\nClassification report guardado.")

matriz = confusion_matrix(y_test, y_pred) #matriz de confusion

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["Normal", "Anormal"]
)

disp.plot(cmap="Blues")
plt.title("Matriz de confusión - Logistic Regression")
plt.savefig("confusion_matrix_lr.png")
plt.close()

print("Matriz de confusión guardada.")


with open("metrics_lr.txt", "w", encoding="utf-8") as f: #guardamos las metricas
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")

print("Métricas guardadas.")
