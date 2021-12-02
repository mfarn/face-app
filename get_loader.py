from facenet_pytorch import MTCNN, InceptionResnetV1
import torch as torch
from torchvision import datasets
from torch.utils.data import DataLoader as DataLoader
from PIL import Image as Image


class Load_Data:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # apenas um rosto
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.dataset = datasets.ImageFolder(root_dir+'/photos-mini')

    def Load(self):
        idx_to_class = {i: c for c, i in self.dataset.class_to_idx.items()}

        def collate_fn(x):
            return x[0]

        loader = DataLoader(self.dataset, collate_fn=collate_fn)

        name_list = []  # lista de nomes correspondentes às fotos
        embedding_list = []  # lista de embedding matrix apos conversao das fotos usando resnet

        for img, idx in loader:
            face, prob = self.mtcnn0(img, return_prob=True)
            if face is not None and prob > 0.92:
                emb = self.resnet(face.unsqueeze(0))
                embedding_list.append(emb.detach())
                name_list.append(idx_to_class[idx])

        # salvar os dados
        data = [embedding_list, name_list]
        torch.save(data, self.root_dir+'/data.pt')
        return data