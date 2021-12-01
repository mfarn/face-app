from facenet_pytorch import MTCNN, InceptionResnetV1
import torch as torch
from torchvision import datasets
from torch.utils.data import DataLoader as DataLoader
from PIL import Image as Image
import cv2 as cv
import time
import os

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # apenas um rosto
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # lista de rostos
resnet = InceptionResnetV1(pretrained='vggface2').eval()

dataset = datasets.ImageFolder('/Users/mateus/Desktop/Code/Python/MachineLearning/face-app/photos-mini')
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}


def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = []  # lista de nomes correspondentes às fotos
embedding_list = []  # lista de embedding matrix apos conversao das fotos usando resnet

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True)
    if face is not None and prob > 0.92:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])

# salvar os dados
data = [embedding_list, name_list]
torch.save(data, '/Users/mateus/Desktop/Code/Python/MachineLearning/face-app/data.pt')

load_data = torch.load('/Users/mateus/Desktop/Code/Python/MachineLearning/face-app/data.pt')
embedding_list = load_data[0]
name_list = load_data[1]

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
    
    k = cv.waitKey(1)
    if k%256==27: #ESC
        print('Esc pressed, closing...')
        break

    elif k%256==32: #space to save image
        print('Enter your name: ')
        name = input()

        # criar past se nao existir
        if not os.path.exists('/Users/mateus/Desktop/Code/Python/MachineLearning/face-app/photos-mini/'+name):
            os.mkdir('/Users/mateus/Desktop/Code/Python/MachineLearning/face-app/photos-mini/'+name)

        img_name = "photos-mini/{}/{}.jpg".format(name, int(time.time()))
        cv.imwrite(img_name, original_frame)
        print(" saved: {}".format(img_name))

cam.release()
cv.destroyAllWindows()