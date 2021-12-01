from facenet_pytorch import MTCNN, InceptionResnetV1
import torch as torch
from torchvision import datasets
from torch.utils.data import DataLoader as DataLoader
from PIL import Image as Image
import cv2 as cv
import time
import os

class Load_Data:

    def save_data(self): 
        
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
    
    def load_data(self):

        load_data = torch.load('/Users/mateus/Desktop/Code/Python/MachineLearning/face-app/data.pt')
        embedding_list = load_data[0]
        name_list = load_data[1]

        return [embedding_list, name_list]