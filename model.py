from facenet_pytorch import MTCNN, InceptionResnetV1
import torch as torch
from get_loader import Load_Data 
from torch.utils.data import DataLoader as DataLoader
from PIL import Image as Image
import cv2 as cv

class Face_Recognition:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # apenas um rosto
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.data = torch.load(root_dir)
        self.embedding_list = self.data[0]
        self.name_list = self.data[1]
        self.cam = cv.VideoCapture(0)   

    def identify_person(self):

        while True:
            ret, frame = self.cam.read()
            if not ret:
                print("Failed to grab a frame, please try again!")
                break

            img = Image.fromarray(frame)
            img_cropped_list, prob_list = self.mtcnn(img, return_prob=True)

            if img_cropped_list is not None:
                boxes, _ = self.mtcnn.detect(img)

                for i, prob in enumerate(prob_list):
                    if prob>0.90:
                        emb = self.resnet(img_cropped_list[i].unsqueeze(0)).detach()

                        dist_list = [] # lista com distancias, menor distancia determina a pessoa

                        for idx, emb_db in enumerate(self.embedding_list):
                            dist = torch.dist(emb, emb_db).item()
                            dist_list.append(dist)

                        min_dist = min(dist_list) # pegar menor distancia
                        min_dist_idx = dist_list.index(min_dist)
                        name = self.name_list[min_dist_idx]

                        box = boxes[i] 
                        
                        original_frame = frame.copy()

                        if min_dist < 0.90:
                            frame = cv.putText(frame, name, (int(box[0]),int(box[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv.LINE_AA)

                        frame = cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

            cv.imshow("IMG", frame)

            cv.imshow("IMG", frame)
    
            k = cv.waitKey(1)
            if k%256==27: #ESC
                print('Esc pressed, closing...')
                break

        self.cam.release()
        cv.destroyAllWindows()

        