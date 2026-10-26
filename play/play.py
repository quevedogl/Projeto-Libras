import json
import random
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# =========================
# CONFIGURAÇÕES
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "gesture_model.keras"
CLASSES_PATH = BASE_DIR / "class_names.json"

IMG_SIZE = 224
CAMERA_INDEX = 0
COUNTDOWN_SECONDS = 3
RESULT_SECONDS = 3

MOVES = ["pedra", "papel", "tesoura"]

# A chave vence o valor.
BEATS = {
    "pedra": "tesoura",
    "papel": "pedra",
    "tesoura": "papel",
}

WINDOW_NAME = "Pedra, Papel e Tesoura com CNN"


# =========================
# CARREGAMENTO DO MODELO
# =========================

def load_model_and_classes():
    """Carrega o modelo treinado e os nomes das classes."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}\n"
            "Execute primeiro o treinamento do modelo."
        )

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de classes não encontrado: {CLASSES_PATH}"
        )

    model = tf.keras.models.load_model(str(MODEL_PATH))

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if isinstance(class_names, dict):
        class_names = list(class_names.values())

    missing_moves = set(MOVES) - set(class_names)

    if missing_moves:
        raise ValueError(
            f"O modelo precisa ter as classes: {', '.join(MOVES)}. "
            f"Classes ausentes: {', '.join(sorted(missing_moves))}"
        )

    return model, class_names


# =========================
# PREDIÇÃO DOS GESTOS
# =========================

def predict_gesture(model, roi, class_names):
    """Identifica o gesto e retorna a confiança da previsão."""

    image = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32)

    probabilities = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))

    gesture = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return gesture, confidence, probabilities


# =========================
# LÓGICA DO JOGO
# =========================

def determine_winner(user_move, pc_move, score):
    """Compara as jogadas e atualiza o placar."""

    if user_move == pc_move:
        score["empates"] += 1
        return f"Empate! PC: {pc_move}"

    if BEATS[user_move] == pc_move:
        score["voce"] += 1
        return f"Você ganhou! ({user_move} x {pc_move})"

    score["pc"] += 1
    return f"PC ganhou! ({user_move} x {pc_move})"


def draw_interface(
    frame,
    roi_coords,
    gesture,
    confidence,
    probabilities,
    class_names,
    message,
    score,
    countdown=None,
):
    """Desenha a interface do jogo na imagem da webcam."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi_coords

    # Área onde o usuário posiciona a mão.
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    # Gesto identificado.
    cv2.putText(
        frame,
        f"Modelo: {gesture} ({confidence:.0%})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    # Barras de probabilidade.
    for index, class_name in enumerate(class_names):
        probability = float(probabilities[index])
        bar_width = int(probability * 150)
        y = 55 + index * 25

        cv2.rectangle(
            frame,
            (10, y),
            (10 + bar_width, y + 14),
            (0, 200, 255),
            -1,
        )

        cv2.putText(
            frame,
            f"{class_name}: {probability:.0%}",
            (170, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Contagem regressiva.
    if countdown is not None:
        cv2.putText(
            frame,
            str(countdown),
            (width // 2 - 25, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 255),
            5,
        )

    # Mensagem do jogo.
    cv2.putText(
        frame,
        message,
        (10, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Placar.
    scoreboard = (
        f"Você {score['voce']} x {score['pc']} PC "
        f"(Empates: {score['empates']})"
    )

    cv2.putText(
        frame,
        scoreboard,
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "ESPACO: jogar | Q: sair",
        (width - 280, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


# =========================
# EXECUÇÃO DO JOGO
# =========================

def main():
    model, class_names = load_model_and_classes()

    # Mantém a abertura da webcam semelhante ao código original.
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir a webcam.")

    recent_predictions = deque(maxlen=7)

    score = {
        "voce": 0,
        "pc": 0,
        "empates": 0,
    }

    state = "idle"
    state_start_time = 0
    message = "Aperte ESPACO para jogar"
    pc_move = None

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Falha ao capturar imagem da webcam.")
                break

            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # Define a região central de captura.
            side = min(height, width) // 2 + 60

            side = min(side, height, width)

            x1 = (width - side) // 2
            y1 = (height - side) // 2
            x2 = x1 + side
            y2 = y1 + side

            roi = frame[y1:y2, x1:x2]

            # Classifica o gesto.
            gesture, confidence, probabilities = predict_gesture(
                model,
                roi,
                class_names,
            )

            recent_predictions.append(gesture)

            # Suaviza a previsão usando os últimos frames.
            stable_gesture = Counter(
                recent_predictions
            ).most_common(1)[0][0]

            countdown = None

            # Durante a contagem regressiva.
            if state == "count":
                elapsed = time.time() - state_start_time

                countdown = COUNTDOWN_SECONDS - int(elapsed)

                if elapsed >= COUNTDOWN_SECONDS:
                    user_move = stable_gesture

                    if user_move not in MOVES:
                        message = "Não reconheci sua mão! Tente novamente."
                    else:
                        pc_move = random.choice(MOVES)

                        message = determine_winner(
                            user_move,
                            pc_move,
                            score,
                        )

                    state = "result"
                    state_start_time = time.time()

                else:
                    message = f"Prepare-se... {countdown}"

            # Após mostrar o resultado, aguarda para permitir outra rodada.
            elif state == "result":
                if time.time() - state_start_time >= RESULT_SECONDS:
                    state = "idle"
                    message = "Aperte ESPACO para jogar"

            # Desenha a interface.
            draw_interface(
                frame=frame,
                roi_coords=(x1, y1, x2, y2),
                gesture=gesture,
                confidence=confidence,
                probabilities=probabilities,
                class_names=class_names,
                message=message,
                score=score,
                countdown=countdown,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32 and state == "idle":
                state = "count"
                state_start_time = time.time()
                recent_predictions.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Erro: {error}")