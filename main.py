import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
import os
from gpiozero import LED
from gpiozero.pins.lgpio import LGPIOFactory
import numpy as np

factory = LGPIOFactory()
gpio_verde = LED(5, pin_factory=factory)
gpio_vermelho = LED(6, pin_factory=factory)

gpio_verde.on()
gpio_vermelho.on()
time.sleep(2)

# Carrega modelo
model = YOLO("pesos/wV3.pt")

# Inicializa câmera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    #2304x1296
    main={"size": (2304, 1296), "format": "RGB888"}
))
picam2.start()

# Foco manual fixo no ferramental
picam2.set_controls({
    "AfMode": 0,
    "LensPosition": 6.0  # Ajuste fino aqui
})

# Travar exposição
picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False
})

def processa_imagem():
    frame = picam2.capture_array()

    # Inferência YOLO
    results = model(frame, verbose=False)

    
    # Realizar a detecção de objetos e armazenar em um objeto do tipo dict (dicionário python)
    detections = []
    detection = {}

    for result in results:
        for box in result.boxes:
            #print(f"Box: {box}")  
            #print(f"cls: {box.cls}, conf: {box.conf}, bbox: {box.xyxy}") 
            
            cls = box.cls.item() 
            conf = box.conf.item() 
            # bbox = [float(coord) for coord in box.xyxy[0]]

            detection = {
                "class": cls,
                "confidence": conf,
                # "bbox": bbox
            }
            
            detections.append(detection)

    return detections # retorno da função

def controle_leds():
    result = processa_imagem()

    #  Coleta das classes detectadas
    classes = [det['class'] for det in result]
    print(type(classes[0]))
	
    # ID da classe "Part"
    PART_CLASS_ID = 0.0   # <-- altere aqui se necessário

    # Verifica se "Part" foi detectada
    if PART_CLASS_ID in classes:
        print("Peca detectada: LED verde ligado, LED vermelho desligado!")
        gpio_verde.on()       # acende verde
        gpio_vermelho.off()   # apaga vermelho
    else:
        print("Peca NAO detectada: LED vermelho ligado, LED verde desligado!")
        gpio_verde.off()      # apaga verde
        gpio_vermelho.on()    # acende vermelho

    #time.sleep(2)
    #gpio_verde.off()      # apaga verde
    #gpio_vermelho.off()    # acende vermelho

def main():
    # incio = time.time()
    # camera = setup()
    notas = []
    n = 0
    while True:
        inicio = time.time()
        controle_leds()
        # time.sleep(10)
        fim = time.time()
        notas.append(fim-inicio)
        n+=1
        #if n==30: break
        # print(f'Tempo de execução: {fim - inicio:.2f} s')

    with open('lista.txt', 'w') as f:
        for tempo in notas:
            f.write(str(tempo) + '\n')

if __name__ == '__main__':
    main()
