import cv2  # Importer OpenCV for videobehandling
from ultralytics import YOLO  # Importer YOLO-modellen fra ultralytics-biblioteket
import torch  # Importer PyTorch for å jobbe med YOLO-modellen
import time  # Importer tid-modulen for å måle tid
import gpustat  # Importer gpustat for å overvåke GPU-bruk

# laster inn ferdigtrent YOLO-modell
model = YOLO('yolov8n.pt')

# path til video som brukes
vidPath = "C:/Users/lostc/Desktop/car.MP4"

# åpner videoen(vidPath) eller bruk (0) for camera
videoCapture = cv2.VideoCapture(0)

# ser om det er noe å vise
if not videoCapture.isOpened():
    print("Feil ved åpning av videofil.")
    exit()

# definerer variabler som skal brukes til å regne gjennomsnittet
totalInferenceTime = 0
totalFPS = 0
frameCount = 0

# justerer størrelse på window
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 1280, 720)  # størrelse på vinduet

# Loop gjennom videoframes
while videoCapture.isOpened():
    # leser frames (bilder) fra en video eller kamera og behandler dem en etter en
    
    # ableToReadFrame vil være true hvis det er noe å ''lese'' og false ellers
    ableToReadFrames, frame = videoCapture.read()  # frame er bildet som blir lest fra videoen eller kameraet
    
    if ableToReadFrames:
        
        startInferenceTime = time.perf_counter()  # begynner å måler tid for inferanse (deteksjon)
        
        detectObjects = model(frame)  # gjør selve deteksjonen også kalt inferanse
        endInferenceTime = time.perf_counter()  # stopper tid for deteksjonen
        
        inferenceTime = endInferenceTime - startInferenceTime  # regner ut tiden brukt til inferanse (deteksjon)
        totalInferenceTime += inferenceTime  # legger tiden i total inferansetid
        startDisplayTime = time.perf_counter()  # starter tiden for å vise deteksjonen
        
        markedFrame = detectObjects[0].plot()  # frames som har detekterte objekter blir markert

        # Viser FPS på bildet, tatt fra ChatGPT
        fps = 1 / inferenceTime  # regner ut FPS
        totalFPS += fps  # legger til FPS-verdien for hver ramme
        cv2.putText(markedFrame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # skriver FPS på bildet

        # viser resultatet i eget vindu
        cv2.imshow("YOLOv8 Inference", markedFrame) 
        endDisplayTime = time.perf_counter()  # stopper tiden det tar å vise

        displayTime = endDisplayTime - startDisplayTime  # regner ut tiden brukt til visning
       
        # Skriv ut prosesseringstid mens deteksjonen skjer
        print(f"Prosesseringstid (inferanse) for ramme {frameCount}: {inferenceTime:.4f} sekunder")
        print(f"Tid vise deteksjon for ramme {frameCount}: {displayTime:.4f} sekunder")

        frameCount += 1  # en tracker som holder oversikt over antall rammer mens deteksjonen kjører

        # 
        key = cv2.waitKey(1)
        if key == 27:  # hvis 'esc' blir trykket, så avslutter det
            break

# regner ut gjennomsnittene, ved hjelp av ChatGPT
if frameCount > 0:
    avgInferenceTime = totalInferenceTime / frameCount  # Gjennomsnittlig inferansetid
    avgFPS = totalFPS / frameCount  # Gjennomsnittlig FPS
    print(" ")
    print(f"Gjennomsnittlig prosesseringstid for alle rammer: {avgInferenceTime:.4f} sekunder")
    print(f"Gjennomsnittlig FPS: {avgFPS:.2f}")
else:
    print("Ingen rammer lest inn.")

# bruker gpustat for å få GPU informasjon
gpuStats = gpustat.GPUStatCollection.new_query()

# skriver ut GPU ytelse
for gpu in gpuStats.gpus:
    print(f"GPU {gpu.index}: {gpu.name}, GPU-ytelse: {gpu.utilization}%")

videoCapture.release()
cv2.destroyAllWindows()
