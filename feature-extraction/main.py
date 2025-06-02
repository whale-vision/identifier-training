from dataloader import TripletWhaleDataset
# from accuracy import calculateAccuracy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import sys
from datetime import datetime


WHALE_SPECIES = sys.argv[1]

print(WHALE_SPECIES)


DATASET = "/db/lpxsf2/extractor_datasets/" + WHALE_SPECIES
OUTPUT = "/db/lpxsf2/outputs/" + WHALE_SPECIES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=1)


def createModel():
	model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

	model.fc = nn.Sequential(nn.Linear(
		in_features=model.fc.in_features, out_features=128, bias=False), L2_norm())
	model = nn.DataParallel(model)
	
	model = model.to(device)
	model.eval()
	
	return model
    


# Hyperparameters
batch_size = 32
embedding_size = 128
lr = 0.001
num_epochs = 10000


# Load the dataset
train_val_dataset = TripletWhaleDataset(DATASET, transform=data_transforms)

train_ratio = 0.9
dataset_size = len(train_val_dataset)

train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size


# Create the model
model = createModel()
model.to(device)

# Define the triplet loss function
triplet_loss = nn.TripletMarginLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)



train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

highest_accuracy = 0.0

for epoch in range(num_epochs):
    # Create dataloaders for each split
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.train()

    for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()

        # anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        anchor, positive, negative = anchor.to("cpu"), positive.to("cpu"), negative.to("cpu")


    # Evaluation on validation set
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(val_dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)

            anchor, positive, negative = anchor.to("cpu"), positive.to("cpu"), negative.to("cpu")

            val_loss += loss.item()

        filePath = OUTPUT + f'.pth'
        torch.save(model.state_dict(), filePath)
        

        print(f'Epoch: {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_dataloader):.4f}, End time: {datetime.now()}', flush=True)

    # if (epoch + 1) % 10 == 0:
    #     accuracy = calculateAccuracy(filePath, DATASET)

    #     print(f'Accuracy: {accuracy:.4f}', flush=True)

    #     if accuracy > highest_accuracy:
    #         torch.save(model.state_dict(), OUTPUT + f'_best.pth')
    #         highest_accuracy = accuracy

    #     if accuracy >= 1:
    #         break


print('Training complete!')