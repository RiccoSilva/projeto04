import cv2
import numpy as np
import tensorflow as tf

# Verifica a versão do TensorFlow
print(f'TensorFlow version: {tf.__version__}')

# Tenta importar a função `load_model`
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from tensorflow.compat.v1.keras.models import load_model

# Caminho para o modelo salvo (ajuste conforme necessário)
caminho_modelo = 'modelo_libras.h5'

# Carrega o modelo
try:
    modelo_libras = load_model(caminho_modelo)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

# Inicializa a webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

# Inicializa o rastreador de mãos
from cvzone.HandTrackingModule import HandDetector
rastreador = HandDetector(detectionCon=0.8, maxHands=2)

# Mapeamento das letras para índices do modelo
letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

while True:
    sucesso, imagem = webcam.read()
    
    if not sucesso:
        print("Não foi possível capturar o vídeo.")
        break

    hands, imagem_maos = rastreador.findHands(imagem)

    if hands:
        for hand in hands:
            lmList = hand['lmList']
            bbox = hand['bbox']
            x, y, w, h = bbox
            roi = imagem[y:y+h, x:x+w]
            roi = cv2.resize(roi, (224, 224))
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)

            try:
                predicao = modelo_libras.predict(roi)
                letra = letras[np.argmax(predicao)]
                cv2.putText(imagem_maos, letra, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Erro ao fazer a predição: {e}")

    cv2.imshow("Reconhecimento de LIBRAS", imagem_maos)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
