# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:28:06 2025

@author: erwan
"""

######
# Le code ci dessous nous a permis de modifier le nom des images pour le standardiser
# Cependant certaines images avaient des noms illisibles sur Python, ces dernières ont du être renommées à la main
######

import os
from pathlib import Path
print("Racine du projet :", os.getcwd())
base_dir = os.getcwd()
os.chdir("C:/Users/erwan/Documents/M2+/BigDataLC/Projet Erwan-Marie-Sixtine")

#On va renommer toutes les images de train
train_lien = str(Path.cwd()) + "/train"
for element in os.listdir(train_lien):
    # Vérifie si c'est un dossier
    if os.path.isdir(os.path.join(train_lien, element)):
        image_lien = train_lien + "/" + element
        print(image_lien)
        for i, filename in enumerate(os.listdir(image_lien)):
            print(filename)
            os.rename(image_lien + "/" + filename, image_lien + "/" + "train" + element + str(i) + ".jpg")

test_lien = str(Path.cwd()) + "/test"
for element in os.listdir(test_lien):
    # Vérifie si c'est un dossier
    if os.path.isdir(os.path.join(test_lien, element)):
        image_lien = test_lien + "/" + element
        print(image_lien)
        for i, filename in enumerate(os.listdir(image_lien)):
            print(filename)
            os.rename(image_lien + "/" + filename, image_lien + "/" + "test" + element + str(i) + ".jpg")

#Importation des packages nécéssaires à l'utilisation du VIT

import os
import torch
import torch.nn as nn
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

#Definition de la racine du projet
#Ce chemin doit être modifié sur l'ordinateur de la personne qui l'utilise
os.chdir("C:/Users/erwan/Documents/M2+/BigDataLC/Projet Erwan-Marie-Sixtine")

# Transformation des images pour être lisibles par Vit16 224
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

#On charge les images depuis les dossiers train et test
train_dataset = datasets.ImageFolder("train", transform=train_transform)
test_dataset = datasets.ImageFolder("test", transform=test_transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#On regarde le nombre de classes que l'on a, ici toutes celles présentent mais on peut également choisir à la main
classes_voulues = train_dataset.classes
num_classes = len(train_dataset.classes)

#On importe le modèle "Vit patch16 224" du package timm, on précise le nombre de neurones sur la couche de sortie pour bien avoir une classification selon notre nombre de classes.
#Par défaut il y a 1000 classes
model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=num_classes)

#Si présence de cuda on charge sur le GPU
#Si vous n'avez pas de GPU, il est conseillé de le faire une machine en disposant ou à défaut sur google colab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#On définit notre loss et notre optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


#On freeze les paramètres du modèles (environ 81 millions), puis on garde les 3 dernières couches comme entrainable
for param in model.parameters():
    param.requires_grad = False

for param in model.blocks[-1].parameters():
    param.requires_grad = True

for param in model.head.parameters():
    param.requires_grad = True


#On fait sur 10 epochs
num_epochs = 10        
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
        
        
class_names = test_loader.dataset.classes  # ou train_dataset.classes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()



#####

#On veut détecter si une feuille est malade ou non, on a fusionné les feuilles de différentes espèces pour voir si une feuilles est malade ou non
#Le code ci-dessous est presque le même que le précendent, à la différence des définitions des jeux d'entrainements et de test

#####

#On charge les images depuis les dossiers train et test
trainmal_dataset = datasets.ImageFolder("trainmal", transform=train_transform)
testmal_dataset = datasets.ImageFolder("testmal", transform=test_transform)


trainmal_loader = DataLoader(trainmal_dataset, batch_size=32, shuffle=True)
testmal_loader = DataLoader(testmal_dataset, batch_size=32, shuffle=True)

#On regarde le nombre de classes que l'on a, ici toutes celles présentent mais on peut également choisir à la main
classes_voulues = trainmal_dataset.classes
num_classes = len(trainmal_dataset.classes)

#On importe le modèle "Vit patch16 224" du package timm, on précise le nombre de neurones sur la couche de sortie pour bien avoir une classification selon notre nombre de classes.
#Par défaut il y a 1000 classes
model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=num_classes)

#Si présence de cuda on charge sur le GPU
#Si vous n'avez pas de GPU, il est conseillé de le faire une machine en disposant ou à défaut sur google colab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#On définit notre loss et notre optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


#On freeze les paramètres du modèles (environ 81 millions), puis on garde les 3 dernières couches comme entrainable
for param in model.parameters():
    param.requires_grad = False

for param in model.blocks[-1].parameters():
    param.requires_grad = True

for param in model.head.parameters():
    param.requires_grad = True


#On fait sur 10 epochs
num_epochs = 10        
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
        
    for images, labels in trainmal_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(trainmal_loader)
    print(f"Époque [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
        
        
model.eval()
correct = 0
total = 0
        
all_labels = []
all_preds = []
        
with torch.no_grad():
    for images, labels in testmal_loader:
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
        
        
class_names = testmal_loader.dataset.classes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()
