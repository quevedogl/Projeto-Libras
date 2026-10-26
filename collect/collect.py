import os
import time
import cv2

# ==========================================
# CONFIGURAÇÕES
# ==========================================

CLASSES = ["pedra", "papel", "tesoura", "nada"]

KEYS = {
    ord("1"): "pedra",
    ord("p"): "pedra",

    ord("2"): "papel",
    ord("a"): "papel",

    ord("3"): "tesoura",
    ord("t"): "tesoura",

    ord("4"): "nada",
    ord("n"): "nada",
}

IMG_SIZE = 224
MAX_PER_CLASS = 150
SAVE_EVERY_N_FRAMES = 3
COUNTDOWN = 3.0
DEBOUNCE = 0.5

# Diretório de dados relativo ao arquivo atual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

WINDOW_NAME = "Coleta de Gestos"

# ==========================================
# PREPARAÇÃO DOS DIRETÓRIOS
# ==========================================

for gesture in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)


def count_images(gesture):
    """Conta as imagens JPG existentes para um gesto."""
    folder = os.path.join(DATA_DIR, gesture)

    return sum(
        1
        for filename in os.listdir(folder)
        if filename.lower().endswith(".jpg")
    )


# Mantém a contagem em memória para evitar consultar
# todos os arquivos repetidamente a cada quadro.
image_counts = {
    gesture: count_images(gesture)
    for gesture in CLASSES
}

# ==========================================
# INICIALIZAÇÃO DA WEBCAM
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise SystemExit(
        "Não foi possível abrir a webcam. "
        "Feche outros aplicativos que estejam usando a câmera "
        "ou tente cv2.VideoCapture(1)."
    )

# ==========================================
# VARIÁVEIS DE CONTROLE
# ==========================================

recording = None
countdown_start = None

last_key_time = 0
frame_count = 0

print("\n======================================")
print("       COLETA DE GESTOS - LIBRAS")
print("======================================")
print("1 ou P: Pedra")
print("2 ou A: Papel")
print("3 ou T: Tesoura")
print("4 ou N: Nada")
print("Q: Encerrar coleta")
print("======================================")
print("Clique na janela da webcam para utilizar as teclas.\n")

# ==========================================
# LOOP PRINCIPAL
# ==========================================

try:
    while True:

        # Captura um quadro da webcam
        ok, frame = cap.read()

        if not ok:
            print("Falha ao capturar imagem da webcam.")
            break

        # Espelha a imagem para facilitar o posicionamento
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # Define a região quadrada de captura
        side = min(height, width) // 2 + 60

        x1 = max(0, (width - side) // 2)
        y1 = max(0, (height - side) // 2)

        x2 = min(width, x1 + side)
        y2 = min(height, y1 + side)

        roi = frame[y1:y2, x1:x2]

        status = "PARADO - pressione 1, 2, 3 ou 4"
        color = (0, 255, 0)

        # ======================================
        # CONTAGEM REGRESSIVA
        # ======================================

        if recording is not None and countdown_start is not None:

            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN - elapsed

            if remaining > 0:
                status = (
                    f"{recording.upper()} - prepare-se: "
                    f"{int(remaining) + 1}"
                )
                color = (0, 255, 255)

            else:
                countdown_start = None
                frame_count = 0

                print(f"Iniciando coleta: {recording}")

        # ======================================
        # COLETA DAS IMAGENS
        # ======================================

        if recording is not None and countdown_start is None:

            status = f"GRAVANDO: {recording.upper()}"
            color = (0, 0, 255)

            if image_counts[recording] >= MAX_PER_CLASS:

                print(
                    f"Coleta de '{recording}' concluída: "
                    f"{image_counts[recording]} imagens."
                )

                recording = None
                countdown_start = None

                status = "COLETA CONCLUIDA"
                color = (0, 255, 0)

            else:
                frame_count += 1

                if frame_count % SAVE_EVERY_N_FRAMES == 0:

                    # Redimensiona a região de interesse
                    image = cv2.resize(
                        roi,
                        (IMG_SIZE, IMG_SIZE)
                    )

                    # Gera um nome único para a imagem
                    timestamp = time.time_ns()

                    filename = f"{timestamp}.jpg"

                    filepath = os.path.join(
                        DATA_DIR,
                        recording,
                        filename
                    )

                    # Salva a imagem e verifica o resultado
                    saved = cv2.imwrite(filepath, image)

                    if saved:
                        image_counts[recording] += 1

                        print(
                            f"\r{recording}: "
                            f"{image_counts[recording]}/{MAX_PER_CLASS}",
                            end="",
                            flush=True
                        )

                    else:
                        print(
                            f"\nErro ao salvar imagem: {filepath}"
                        )

        # ======================================
        # INTERFACE VISUAL
        # ======================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        # Exibe o progresso de cada classe
        for index, gesture in enumerate(CLASSES):

            text = (
                f"[{index + 1}] {gesture}: "
                f"{image_counts[gesture]}/{MAX_PER_CLASS}"
            )

            cv2.putText(
                frame,
                text,
                (10, 65 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imshow(WINDOW_NAME, frame)

        # ======================================
        # CONTROLE DO TECLADO
        # ======================================

        key = cv2.waitKey(10) & 0xFF

        # Ignora quadros sem entrada do teclado
        if key == 255:
            continue

        # Encerra o programa
        if key == ord("q"):
            print("\nEncerrando a coleta...")
            break

        # Verifica se a tecla corresponde a um gesto
        if key in KEYS:

            current_time = time.time()

            # Evita repetições rápidas de teclas
            if current_time - last_key_time < DEBOUNCE:
                continue

            last_key_time = current_time

            gesture = KEYS[key]

            # Pausa se o mesmo gesto for selecionado
            if recording == gesture:

                recording = None
                countdown_start = None

                print(f"\nColeta pausada: {gesture}")

            else:

                # Não inicia a coleta se a classe já estiver completa
                if image_counts[gesture] >= MAX_PER_CLASS:

                    print(
                        f"\nA classe '{gesture}' já possui "
                        f"{MAX_PER_CLASS} imagens."
                    )

                    recording = None
                    countdown_start = None

                else:

                    recording = gesture
                    countdown_start = time.time()

                    print(
                        f"\nPreparando a coleta de '{gesture}'..."
                    )

# ==========================================
# FINALIZAÇÃO SEGURA
# ==========================================

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("\n\n======================================")
    print("       RESUMO DA COLETA")
    print("======================================")

    for gesture in CLASSES:
        print(
            f"{gesture.capitalize()}: "
            f"{count_images(gesture)} imagens"
        )

    print(f"\nDiretório dos dados: {DATA_DIR}")
    print("Programa encerrado.")