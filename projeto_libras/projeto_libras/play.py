import json
import cv2
import numpy as np
import tensorflow as tf
from collections import Counter, deque

IMG_SIZE = 224
try:
    model = tf.keras.models.load_model("gesture_model.keras")
    with open("class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)
except Exception as exc:
    raise SystemExit(f"Não consegui carregar o modelo. Execute python train.py primeiro.\nDetalhe: {exc}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Não consegui abrir a webcam. Feche Zoom/Meet e tente novamente.")

recent = deque(maxlen=7)
while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao ler a webcam.")
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = max(0, w // 2 - side // 2), max(0, h // 2 - side // 2)
    side = min(side, w - x1, h - y1)
    roi = frame[y1:y1 + side, x1:x1 + side]

    img = cv2.cvtColor(cv2.resize(roi, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    probs = model.predict(img[None].astype("float32"), verbose=0)[0]
    idx = int(np.argmax(probs))
    label = class_names[idx]
    confidence = float(probs[idx])
    recent.append(label)
    stable = Counter(recent).most_common(1)[0][0]

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), (0, 255, 0), 3)
    cv2.putText(frame, f"Sinal: {stable} ({confidence:.0%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    for i, cls in enumerate(class_names):
        bar_width = int(float(probs[i]) * 180)
        y = 65 + i * 26
        cv2.rectangle(frame, (10, y), (10 + bar_width, y + 14), (0, 200, 255), -1)
        cv2.putText(frame, cls, (200, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1)
    cv2.putText(frame, "Q = sair", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Reconhecimento de sinais de Libras", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
