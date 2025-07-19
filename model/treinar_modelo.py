import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

dados = pd.DataFrame()
for arquivo in os.listdir("data"):
    if arquivo.endswith(".csv"):
        caminho = os.path.join("data", arquivo)
        df = pd.read_csv(caminho)
        dados = pd.concat([dados, df], ignore_index=True)

# Separar features e rótulo
X = dados.drop("label", axis=1)
y = dados["label"]

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Treinar modelo SVM
print("Treinando modelo SVM...")
modelo = SVC(kernel="rbf", gamma="scale", probability=True)
modelo.fit(X_train, y_train)

# Avaliação
y_pred = modelo.predict(X_test)
print("\n📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred))
print(f"Acurácia geral: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Salvar modelo
os.makedirs("model", exist_ok=True)
joblib.dump(modelo, "model/classificador_gestos.pkl")
print("Modelo salvo com sucesso em: model/classificador_gestos.pkl")
