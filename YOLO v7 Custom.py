import cv2
import time
import sys
import numpy as np
import math
import pandas as pd
import tkinter
from tkinter import *
import torch
import os


model_name = 'config_files/bestigo.onnx'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\n\nDevice used:", device)


upload = 0
azul = (255, 191, 0)
verde = (0, 255, 0)
verm = (0, 0, 255)
roxo = (153, 51, 153)
rosa = (255, 0, 255)
amarelo = (24, 117, 255)
roxodark = (130, 0, 75)

idsdict = {}
idsdict["ID"] = ["Posição", "Velocidade", "Classe", "Número", "Sentido", "Tempo"]
classdict = {0:'Caminhão', 1:'Carro', 2:'Moto', 3:'Onibus', 4:'Roda'}

rfframe = -5

arqvideo = 'C01.mp4'
fpsvid = 30
side = 2
side2 = 1

framen = 0
video = cv2.VideoCapture('Videos/'+arqvideo)
_, frame = video.read()
framew = 1000
frameh = int(frame.shape[0]/(frame.shape[1] / framew))
dim = (framew, frameh)
framer = cv2.resize(frame, dim)
clone = framer.copy()
cv2.imshow(arqvideo, framer)
cv2.waitKey(1)
fpsseg = 1/int(fpsvid)
car = 0
truck = 0
moto = 0
car2 = 0
truck2 = 0
moto2 = 0
bus = 0
bus2 = 0
ids = 0

precords = []
rodas = []


inplinha = 0.50
inplinha2 = 0.66
cofaixa = 0.55
cofaixa2 = 0.55
posx = int((inplinha * framer.shape[0]))
posx2 = int((inplinha2 * framer.shape[0]))
faixa = int((cofaixa * framer.shape[1]))
faixa2 = int((cofaixa2 * framer.shape[1]))
dist12 = 10

tempinic = time.time()


while True:

    start_time = time.time()
    rodas = []
    curcords = []
    _, frame = video.read()
    if frame is None:
        print("End of stream")
        break
    clone = frame.copy()
    dimmodel = (640, 640)
    frame = cv2.resize(frame, dimmodel)
    posx = int((inplinha * frame.shape[0]))
    posx2 = int((inplinha2 * frame.shape[0]))
    faixa = int((cofaixa * frame.shape[1]))
    faixa2 = int((cofaixa2 * frame.shape[1]))
    framen += 1
    model.to(device)
    results = model(frame)
    boxes = []
    scores = []
    class_ids = []
    for result in results.xyxyn[0]:
        boxes.append((int(result[0]*frame.shape[1]), int(result[1]*frame.shape[0]), int(result[2]*frame.shape[1]- (result[0]*frame.shape[1])), int(result[3]*frame.shape[0] - (result[1]*frame.shape[0]))))
        scores.append(float(result[4]))
        class_ids.append(int(result[5]))
    cv2.line(frame, (0, posx), (frame.shape[0], posx), verde, 3)
    cv2.line(frame, (0, posx2), (frame.shape[0], posx2), verde, 3)
    cv2.line(frame, (faixa, posx + 15), (faixa, posx - 15), roxo, 2)
    cv2.line(frame, (faixa2, posx2 + 15), (faixa2, posx2 - 15), roxo, 2)
    boxcount = -1
    for box in boxes:
        boxcount += 1
        if class_ids[boxcount] == 4 and scores[boxcount] > 0.5:
            (x, y, w, h) = box
            x1 = x + int(w/2)
            y1 = y + int(h/2)
            centercord = (x1, y1)
            rodas.append(centercord)
            
    boxcount = -1
    for box in boxes:
        eix = 2
        boxcount += 1
        (x, y, w, h) = box
        if scores[boxcount] > 0.5:
            if class_ids[boxcount] == 1:
                texto = "Carro"
                cor = verde
            elif class_ids[boxcount] == 2:
                texto = "Moto"
                cor = azul
            elif class_ids[boxcount] == 0:
                texto = "Caminhao"
                cor = verm
            elif class_ids[boxcount] == 3:
                texto = "Onibus"
                cor = rosa
            else:
                texto = ""
                cor = amarelo
            cv2.rectangle(frame, (x,y), (x + w, y + h), cor, 2)
            cv2.putText(frame, str(texto), (x+w+5, y+h), 0, 0.5, cor, 2)
            x1 = x + int(w/2)
            y1 = y + int(h)

            centercord = (x1, y1)
            if class_ids[boxcount] != 4:
                curcords.append(centercord)
            for precord in precords:
                (xp, yp) = precord
                dist = math.hypot(x1 - xp, y1 - yp)
                if dist < 25 and class_ids[boxcount] != 4:
                                        
                    if yp <= posx and y1 > posx: #Faixa 1 - Cima -> Baixo
                        if side == 1:
                            if x1 > faixa and xp > faixa:
                                ids += 1
                                if class_ids[boxcount] == 1: # Conta A
                                    car += 1
                                    ident = car
                                elif class_ids[boxcount] == 2:
                                    moto += 1
                                    ident = moto
                                elif class_ids[boxcount] == 3:
                                    bus += 1
                                    ident = bus
                                elif class_ids[boxcount] == 0:
                                    truck += 1
                                    ident = truck     
                                else:
                                    ident = "X"
                                cv2.line(frame, (0, posx), (frame.shape[0], posx), verm, 3)
                                tempoent = framen
                                idsdict[ids] = [centercord, tempoent, class_ids[boxcount], ident, "Sentido A", round(framen*fpsseg, 2)]
                                
                                
                        if side2 == 1:
                            if x1 < faixa and xp < faixa:
                                ids += 1
                                if class_ids[boxcount] == 1: # Conta B
                                    car2 += 1
                                    ident = car2
                                elif class_ids[boxcount] == 2:
                                    moto2 += 1
                                    ident = moto2
                                elif class_ids[boxcount] == 3:
                                    bus2 += 1
                                    ident = bus2
                                elif class_ids[boxcount] == 0:
                                    truck2 += 1
                                    ident = truck2
                                else:
                                    ident = "X"
                                cv2.line(frame, (0, posx), (frame.shape[0], posx), azul, 3)
                                tempoent = framen
                                idsdict[ids] = [centercord, tempoent, class_ids[boxcount], ident, "Sentido B", round(framen*fpsseg, 2)]


                    elif yp <= posx2 and y1 > posx2: #Faixa 2 - Esquerda -> Direita
                        if side == 1:
                            if x1 > faixa2:
                                for check in idsdict: # Velocidade
                                    if idsdict[check][0] == precord:
                                        frametime = (framen - idsdict[check][1]) * fpsseg
                                        idsdict[check][1] = round((dist12 / frametime * 3.6), 2)
                                        idsdict[check][0] = centercord
                                
                        if side2 == 1:
                            if x1 < faixa2:
                                for check in idsdict: # Velocidade
                                    if idsdict[check][0] == precord:
                                        frametime = (framen - idsdict[check][1]) * fpsseg
                                        idsdict[check][1] = round((dist12 / frametime * 3.6), 2)
                                        idsdict[check][0] = centercord

                    elif yp >= posx and y1 < posx: #Faixa 1 - Direita -> Esquerda
                        if side == 2:
                            if x1 > faixa:
                                for check in idsdict: # Velocidade
                                    if idsdict[check][0] == precord:
                                        frametime = (framen - idsdict[check][1]) * fpsseg
                                        idsdict[check][1] = round((dist12 / frametime * 3.6), 2)
                                        idsdict[check][0] = centercord
                                
                        if side2 == 2:
                            if x1 < faixa:
                                for check in idsdict: # Velocidade
                                    if idsdict[check][0] == precord:
                                        frametime = (framen - idsdict[check][1]) * fpsseg
                                        idsdict[check][1] = round((dist12 / frametime * 3.6), 2)
                                        idsdict[check][0] = centercord

                    elif yp >= posx2 and y1 < posx2: #Faixa 2 - Direita -> Esquerda
                        if side == 2:
                            if x1 > faixa2 and xp > faixa2:
                                ids += 1
                                if class_ids[boxcount] == 1: # Conta A
                                    car += 1
                                    ident = car
                                elif class_ids[boxcount] == 2:
                                    moto += 1
                                    ident = moto
                                elif class_ids[boxcount] == 3:
                                    bus += 1
                                    ident = bus
                                elif class_ids[boxcount] == 0:
                                    truck += 1
                                    ident = truck
                                else:
                                    ident = "X"
                                cv2.line(frame, (0, posx2), (frame.shape[0], posx2), verm, 3)
                                tempoent = framen
                                idsdict[ids] = [centercord, tempoent, class_ids[boxcount], ident, "Sentido A", round(framen*fpsseg, 2)]
                                   
                        if side2 == 2:
                            if x1 < faixa2 and xp < faixa2:
                                ids += 1
                                if class_ids[boxcount] == 1: # Conta B
                                    car2 += 1
                                    ident = car2
                                elif class_ids[boxcount] == 2:
                                    moto2 += 1
                                    ident = moto2
                                elif class_ids[boxcount] == 3:
                                    bus2 += 1
                                    ident = bus2
                                elif class_ids[boxcount] == 0:
                                    truck2 += 1
                                    ident = truck2
                                else:
                                    ident = "X"
                                cv2.line(frame, (0, posx2), (frame.shape[0], posx2), azul, 3)
                                tempoent = framen
                                idsdict[ids] = [centercord, tempoent, class_ids[boxcount], ident, "Sentido B", round(framen*fpsseg, 2)]

                    else:
                        for check in idsdict:
                            if idsdict[check][0] == precord:
                                idsdict[check][0] = centercord 

                    break
        
                   

    for curcord in curcords:
        cv2.circle(frame, curcord, 1, verm, 2)

    for roda in rodas:
        cv2.circle(frame, roda, 1, verde, 2)

    framer = cv2.resize(frame, dim)
    cv2.putText(framer, "B: ", (160, framer.shape[0] - 30), 0, 1, verde, 2)
    cv2.putText(framer, str(car), (200, framer.shape[0] - 30), 0, 1, verm, 2)
    cv2.putText(framer, str(moto), (290, framer.shape[0] - 30), 0, 1, verm, 2)
    cv2.putText(framer, str(truck), (380, framer.shape[0] - 30), 0, 1, verm, 2)
    cv2.putText(framer, str(bus), (470, framer.shape[0] - 30), 0, 1, verm, 2)
    cv2.putText(framer, "A: ", (160, framer.shape[0] - 60), 0, 1, verde, 2)
    cv2.putText(framer, str(car2), (200, framer.shape[0] - 60), 0, 1, azul, 2)
    cv2.putText(framer, str(moto2), (290, framer.shape[0] - 60), 0, 1, azul, 2)
    cv2.putText(framer, str(truck2), (380, framer.shape[0] - 60), 0, 1, azul, 2)
    cv2.putText(framer, str(bus2), (470, framer.shape[0] - 60), 0, 1, azul, 2)
    cv2.putText(framer, "C", (200, framer.shape[0] - 90), 0, 1, roxo, 2)
    cv2.putText(framer, "M", (290, framer.shape[0] - 90), 0, 1, roxo, 2)
    cv2.putText(framer, "T", (380, framer.shape[0] - 90), 0, 1, roxo, 2)
    cv2.putText(framer, "B", (470, framer.shape[0] - 90), 0, 1, roxo, 2)  
    end_time = time.time()
    fpsshow = 1/np.round(end_time - start_time, 3)
    cv2.putText(framer, str(int(fpsshow)), (int(framer.shape[0]/10)+100,int(framer.shape[0]/10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
    cv2.imshow(arqvideo, framer)
    precords = curcords.copy()
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == 32:
        key = cv2.waitKey(0)
        if key == 27:
            break
        

tempfim = time.time()
durac = tempfim - tempinic
print(durac)
cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()

for check in idsdict:
    if isinstance(idsdict[check][1], int) is True:
        idsdict[check][1] = "N/A"
    if check != "ID":
        if idsdict[check][2] < 10:
            idsdict[check][2] = classdict[idsdict[check][2]]    
    del(idsdict[check][0])
        

df = pd.DataFrame(idsdict)
print(df)
df.to_excel('Results/'+arqvideo+'.xlsx', index=False)
pd.read_excel('Results/'+arqvideo+'.xlsx')


