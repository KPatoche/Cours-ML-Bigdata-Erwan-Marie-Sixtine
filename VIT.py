# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:28:06 2025

@author: erwan
"""

import os
from pathlib import Path
print("Racine du projet :", os.getcwd())
base_dir = Path.cwd()
os.chdir("C:/Users/erwan/Documents/M2+/BigDataLC/Projet Erwan-Marie-Sixtine")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import PIL
import timm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#On importe un modèle de vision transformer qui fonctionne en
model = timm.create_model('vit_base_patch16_224', pretrained=True)

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder("train", transform=train_transform)
test_dataset = datasets.ImageFolder("test", transform=test_transform)

classes_voulues = ['Apple Scab Leaf', 'Apple leaf' , "Apple rust leaf"]
test_dataset.classes

indices = [i for i, (_, label) in enumerate(train_dataset.samples)
            if train_dataset.classes[label] in classes_voulues]

train_data_apple = Subset(train_dataset,indices)
train_loader = DataLoader(train_data_apple, batch_size=32, shuffle=True)

indices = [i for i, (_, label) in enumerate(test_dataset.samples)
            if test_dataset.classes[label] in classes_voulues]

test_data_apple = Subset(test_dataset,indices)
test_loader = DataLoader(test_data_apple, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Époque [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")


model.eval()
correct = 0
total = 0

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

print(f"Précision sur le test set: {100 * correct / total:.2f}%")
cm = confusion_matrix(all_labels, all_preds)
print(cm)


class_names = test_loader.dataset.dataset.classes  # ou train_dataset.classes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()