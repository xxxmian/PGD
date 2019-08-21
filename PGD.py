import numpy as np
import json
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.autograd import Variable

# device
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# data
class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor()
])

def image_folder_custom_label(root, transform, custom_label):
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}
    for i, item in enumerate(idx2label):
        label2idx[item] = i
    new_data=dsets.ImageFolder(root=root, transform=transform,
                               target_transform=lambda x:custom_label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.classes_to_idx = label2idx
    return new_data
normal_data = image_folder_custom_label(root='./data/imagenet',transform=transform,custom_label=idx2label)
normal_loader = data.DataLoader(normal_data,batch_size=1,shuffle=False)

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize=(5,15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
normal_iter = iter(normal_loader)


model = models.inception_v3(pretrained=True).to(device)
print("True Image and Predicted Label")
model.eval()
correct=0
total=0

for images,labels in normal_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, pre = torch.max(outputs.data, 1)
    total = total + 1
    correct = correct + (pre==labels).sum()
    imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True),
           [normal_data.classes[i] for i in pre])
    print("Accuracy of test text:%f %%" % (100*float(correct)/total))

def pgd_attack(model, images,labels,eps=0.3,alpha=2/225,iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    ori_images = images.data
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad=False

    for i in range(iters):
        images.requires_grad = True

        outputs = model(images)
        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images+alpha*images.grad.sign()
        eta = torch.clamp(adv_images-ori_images, min=-eps,max=eps)
        images=torch.clamp(ori_images+eta,min=0,max=1).detach_()

    return images
print("Attack Image and Predicted Label")
model.eval()
correct = 0
total = 0
for images,labels in normal_loader:
    attimages=pgd_attack(model, images, labels)
    labels = labels.to(device)
    outputs = model(attimages)
    _, pre = torch.max(outputs.data, 1)

    total = total + 1
    correct = correct + (pre == labels).sum()
    imshow(torchvision.utils.make_grid(attimages.cpu().data, normalize=True),[normal_data.classes[i] for i in pre])
print('Accuracy of test text: %f %%' % (100 * float(correct) / total))


