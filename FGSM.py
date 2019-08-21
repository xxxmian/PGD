# requirements
import torch
import torch.nn as nn
import json
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
# set args
eps = 0.007
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# prepare data
class_idx = json.load(open('./data/imagenet_class_index.json'))
idx2label = [class_idx[str(i)][1] for i in range(len(class_idx))]
transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor()]
)
def image_folder_custom_label(root, transform, custom_label):
    old_data = dsets.ImageFolder(root, transform=transform)
    old_class = old_data.classes
    new_data = dsets.ImageFolder(root='./data/imagenet', transform=transform,
                                 target_transform=lambda x:custom_label.index(old_class[x]))
    new_data.class_to_idx = [{custom_label[i]:i} for i in range(len(custom_label))]
    new_data.classes = custom_label
    return new_data
normal_data = image_folder_custom_label(root='./data/imagenet', transform=transform, custom_label=idx2label)
normal_loader= DataLoader(normal_data)

def imgshow(img, title):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


print('True image and true label')
for image, label in normal_loader:
    imgshow(torchvision.utils.make_grid(image.cpu().data, normalize=True),
           normal_data.classes[label])
# download inception v3
model = models.inception_v3(pretrained=True).to(device)
# print true image and predicted label

print('True iamge and predicted label')
model.eval()
for image, label in normal_loader:
    image = image.to(device)
    output = model(image)
    pre = torch.argmax(output)
    imgshow(torchvision.utils.make_grid(image.cpu().data, normalize=True),
            normal_data.classes[pre])
# attack
def fgsm_attack(model, image, label, eps):
    image = image.to(device)
    label = label.to(device)
    image.requires_grad = True
    model.zero_grad()
    output = model(image).to(device)
    loss = nn.CrossEntropyLoss()
    cost = loss(output, label).to(device)
    cost.backward()
    attimg = image + eps*image.grad.sign()
    attimg = torch.clamp(attimg,0,1)
    return attimg

for image, label in iter(normal_loader):
    attimg = fgsm_attack(model, image, label, eps=eps)
    pre = torch.argmax(model(attimg.to(device)))
    if pre != label:
        imgshow(torchvision.utils.make_grid(attimg.cpu().data, normalize=True),
                normal_data.classes[pre])
    else:
        print("attack failed!")
