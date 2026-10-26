import cv2

print("Testando webcam...")

# Tenta abrir a câmera usando DirectShow, do Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    cap.release()
    raise SystemExit(1)

print("Câmera aberta! Tentando capturar imagem...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Não foi possível capturar o quadro.")
        break

    cv2.imshow("Teste da Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Teste encerrado.")