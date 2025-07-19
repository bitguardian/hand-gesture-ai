import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

label = input("Digite o número do gesto (0 a 5): ")
arquivo = f"data/gesto_{label}.csv"

# Cria cabeçalho
if not os.path.exists(arquivo):
    with open(arquivo, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
        writer.writerow(header)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("[i] Pressione 's' para salvar o frame. 'q' para sair.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                row = []
                for lm in hand_landmarks.landmark:
                    row.append(lm.x)
                for lm in hand_landmarks.landmark:
                    row.append(lm.y)
                row.append(label)
                with open(arquivo, mode='a', newline='') as f:
                    csv.writer(f).writerow(row)
                print("✔️ Frame salvo!")

    cv2.imshow("Coleta de Dados", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
