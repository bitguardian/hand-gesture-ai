def extrair_landmarks(hand_landmarks, img_shape):
    h, w, _ = img_shape
    pontos = []

    for lm in hand_landmarks.landmark:
        pontos.append(lm.x)
    for lm in hand_landmarks.landmark:
        pontos.append(lm.y)

    return pontos
