import cv2
from ultralytics import YOLO
import torch
import time
import psutil

# Last ferdig trent YOLO-modell
model = YOLO('yolov8n.pt')

# Flytt modellen til CPU hvis CUDA er tilgjengelig
if torch.cuda.is_available():
    model = model.to(torch.device('cpu'))

# path til video som brukes
vidPath = "C:/Users/lostc/Desktop/car.MP4"

# åpner videoen(vidPath) eller bruk (0) for camera
videoCapture = cv2.VideoCapture(1)

# ser om det er noe å vise
if not videoCapture.isOpened():
    print("Feil ved åpning av videofil.")
    exit()

# definerer variabler som skal brukes til å regne gjennomsnittet
totalInferenceTime = 0
totalDisplayTime = 0
totalFPS = 0
totalCpuPercent = 0
frameCount = 0


cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 1280, 720)  # bytter størrelse på vinduet

# Looper gjennom videoframes
while videoCapture.isOpened():
    # leser frames (bilder) fra en video eller kamera og behandler dem en etter en
    
    ableToReadFrames, frame = videoCapture.read()
    # ableToReadFrame vil være true hvis det er noe å ''lese'' og false ellers
    if ableToReadFrames:

        startInferenceTime = time.perf_counter() # begynner å måler tid for inferanse (deteksjon)
        
        detectObjects = model(frame) # gjør selve deteksjonen på video eller fra kamera
        endInferenceTime = time.perf_counter() # stopper tid for deteksjonen
        
        inferenceTime = endInferenceTime - startInferenceTime # regner ut tiden brukt til inferanse (deteksjon)
        totalInferenceTime += inferenceTime # legger tiden i total inferansetid

       
        
        markedFrame = detectObjects[0].plot() # frames som har detekterte objekter blir markert

        # viser FPS på bildet, tatt fra ChatGPT
        fps = 1 / inferenceTime # regner ut FPS
        totalFPS += fps  # legger til FPS-verdien for hver ramme
        cv2.putText(markedFrame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # viser resultatet i eget vindu
        cv2.imshow("YOLOv8 Inference", markedFrame)
        endDisplayTime = time.perf_counter() # stopper tiden det tar å vise

        
       

         # Skriv ut prosesseringstid mens deteksjonen skjer
        print(f"Prosesseringstid (inferanse) for ramme {frameCount}: {inferenceTime:.4f} sekunder")
      

        frameCount += 1 # en tracker som holder oversikt over antall rammer mens deteksjonen kjører

        # måler CPU utnyttelse for hver iterasjon tatt fra ChatGPT
        cpuPercent = psutil.cpu_percent()
        totalCpuPercent += cpuPercent

        
        key = cv2.waitKey(1)
        if key == 27:  # avslutter når 'esc' blir trykket
            break

# regner ut gjennomsnittene, fått hjelp av ChatGPT
if frameCount > 0:
    avgInferenceTime = totalInferenceTime / frameCount
    avgDisplayTime = totalDisplayTime / frameCount
    avgFPS = totalFPS / frameCount  # Beregn gjennomsnittlig FPS
    print(" ")
    print(f"Gjennomsnittlig prosesseringstid for alle rammer: {avgInferenceTime:.4f} sekunder") # gjennomsnittlig inferansetid
    print(f"Gjennomsnittlig tid for visning av deteksjonen: {avgDisplayTime:.4f} sekunder") # gjennomsnittlig visningstid
    print(f"Gjennomsnittlig FPS: {avgFPS:.2f}") # gjennomsnittlig FPS
else:
    print("Ingen rammer lest inn.")

# regner ut gjennomsnittlig CPU-ytelse
if frameCount > 0:
    avgCpuPercent = totalCpuPercent / frameCount
    print(f"Gjennomsnittlig CPU-ytelse: {avgCpuPercent}%")

videoCapture.release()
cv2.destroyAllWindows()
