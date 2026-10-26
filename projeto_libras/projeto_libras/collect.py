import os
import time
import cv2
from class_names import GESTURES

DATA_DIR = "data"
IMG_SIZE = 224
MAX_PER_CLASS = 300
SAVE_EVERY_N_FRAMES = 3
COUNTDOWN = 3.0
DEBOUNCE = 0.5  # ignora repeticao de tecla (segurar) por 0.5s

# 1-9 e 0 para os dez sinais; N para "nada"; L alterna a iluminacao (claro/escuro).
KEY_TO_CLASS = {
    ord("1"): GESTURES[0], ord("2"): GESTURES[1],
    ord("3"): GESTURES[2], ord("4"): GESTURES[3],
    ord("5"): GESTURES[4], ord("6"): GESTURES[5],
    ord("7"): GESTURES[6], ord("8"): GESTURES[7],
    ord("9"): GESTURES[8], ord("0"): GESTURES[9],
    ord("n"): "nada",
}
CLASSES = GESTURES + ["nada"]

for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

def count_images(cls):
    folder = os.path.join(DATA_DIR, cls)
    return len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Não consegui abrir a webcam. Feche Zoom/Meet e tente novamente.")

recording = None
countdown_start = None
frame_count = 0
last_key_time = 0
light = "claro"  # iluminacao atual; aperte L para alternar
window = "Coleta de sinais de Libras"
print("Clique na janela do vídeo. 1-9/0 escolhem sinais; N = nada; L = alterna claro/escuro; Q = sair.")

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

    status = "PARADO - aperte 1-9/0 ou N"
    color = (0, 255, 0)

    if recording and countdown_start is not None:
        remaining = COUNTDOWN - (time.time() - countdown_start)
        if remaining > 0:
            status = f"{recording}: prepare a mao... {int(remaining) + 1}"
            color = (0, 255, 255)
        else:
            countdown_start = None

    if recording and countdown_start is None:
        status = f"GRAVANDO: {recording}"
        color = (0, 0, 255)
        frame_count += 1
        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            if count_images(recording) < MAX_PER_CLASS:
                img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                filename = f"{int(time.time() * 1000)}_{light}.jpg"  # sufixo guarda a iluminacao
                cv2.imwrite(os.path.join(DATA_DIR, recording, filename), img)
            else:
                print(f"{recording}: chegou a {MAX_PER_CLASS} fotos.")
                recording = None

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), color, 3)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"ILUMINACAO: {light.upper()} (L alterna)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "1-9/0 = sinais | N = nada | L = luz | Q = sair",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imshow(window, frame)
    key = cv2.waitKey(10) & 0xFF

    if key == 255:
        continue
    if key == ord("q"):
        break
    if key == ord("l") and time.time() - last_key_time > DEBOUNCE:
        last_key_time = time.time()
        light = "escuro" if light == "claro" else "claro"
        print(f"Iluminação: {light}")
    elif key in KEY_TO_CLASS and time.time() - last_key_time > DEBOUNCE:
        last_key_time = time.time()
        chosen = KEY_TO_CLASS[key]
        if recording == chosen:
            print(f"Pausado: {chosen}")
            recording, countdown_start = None, None
        else:
            recording, countdown_start = chosen, time.time()
            print(f"Preparando: {chosen} (3 segundos)")
    elif key == ord(" ") and time.time() - last_key_time > DEBOUNCE:
        if recording:
            recording, countdown_start = None, None
            print("Coleta pausada.")

cap.release()
cv2.destroyAllWindows()
print("Imagens por classe:")
for cls in CLASSES:
    print(f"  {cls}: {count_images(cls)}")
