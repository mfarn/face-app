from facenet_pytorch import MTCNN, InceptionResnetV1
import torch as torch
from load_data import Load_Data 
from torch.utils.data import DataLoader as DataLoader
from PIL import Image as Image
import cv2 as cv
import time
import os

class Find_Face:

    mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # lista de rostos
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    embedding_list = Load_Data.load_data()[0]
    name_list = Load_Data.load_data()[1]

    def identify(self):

        cam = cv.VideoCapture(0)

        while True:
            ret, frame = cam.read()
            if not ret:
                print("fail to grab frame, try again")
                break

            img = Image.fromarray(frame)
            img_cropped_list, prob_list = mtcnn(img, return_prob=True)

            if img_cropped_list is not None:
                boxes, _ = mtcnn.detect(img)

                for i, prob in enumerate(prob_list):
                    if prob>0.90:
                        emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                        dist_list = [] # lista com distancias, menor distancia determina a pessoa

                        for idx, emb_db in enumerate(embedding_list):
                            dist = torch.dist(emb, emb_db).item()
                            dist_list.append(dist)

                        min_dist = min(dist_list) # pegar menor distancia
                        min_dist_idx = dist_list.index(min_dist)
                        name = name_list[min_dist_idx]

                        box = boxes[i] 
                        
                        original_frame = frame.copy()

                        if min_dist < 0.90:
                            frame = cv.putText(frame, name, (int(box[0]),int(box[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv.LINE_AA)

                        frame = cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

            cv.imshow("IMG", frame)