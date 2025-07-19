import joblib

modelo = joblib.load("model/classificador_gestos.pkl")

def classificar_gesto(landmarks):
    if hasattr(modelo, "predict_proba"):
        probas = modelo.predict_proba([landmarks])[0]
        pred = probas.argmax()
        conf = probas[pred]
        return pred, conf
    else:
        pred = modelo.predict([landmarks])[0]
        return pred, 1.0
