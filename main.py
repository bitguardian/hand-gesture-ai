import cv2
import mediapipe as mp
from utils.feature_extractor import extrair_landmarks
from classify import classificar_gesto

# Inicializa o MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Acesso à webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extrair_landmarks(hand_landmarks, frame.shape)

            if len(features) == 42:
                numero, confianca = classificar_gesto(features)
                texto = f"Gesto: {numero} ({confianca*100:.1f}%)"
                cv2.putText(frame, texto, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                print(texto)

    cv2.imshow("Detector de Números com IA", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
