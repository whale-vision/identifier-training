import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet101
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getAllWhales(directory):
    whaleDict = {}

    # Walk through the directory tree
    for root, _, files in os.walk(directory):
        # Get the relative path from the root directory
        relativePath = os.path.relpath(root, directory)
        # Split the relative path to find the first subdirectory (whale name)
        pathParts = relativePath.split(os.sep)

        if len(pathParts) > 0 and pathParts[0] != ".":
            whaleName = pathParts[0]  # First subdirectory is the whale name

            # Initialize the list for this whale if it doesn't exist
            if whaleName not in whaleDict:
                whaleDict[whaleName] = []

            # Add each file to the whale's list
            for file in files:
                fullPath = os.path.join(root, file)

                whaleDict[whaleName].append((whaleName, fullPath, []))

    return whaleDict


TARGET_WIDTH = 224
TARGET_HEIGHT = 224

def loadImageForCropping(imageData):
	img = Image.open(imageData)
	width = img.size[0]
	height = img.size[1]

	img.convert('RGB')
	img = img.resize((TARGET_WIDTH, TARGET_HEIGHT))
	img = np.array(img).astype(np.float32)
	img /= 255.0

    # reshape from (h, w, c) to (c, h, w)
	img = np.swapaxes(img, 0, 2)
	img = np.swapaxes(img, 1, 2)
	
	return img, width, height

def cropImage(imagePath, predictor):
    try:
        predictionImage, orgWidth, orgHeight = loadImageForCropping(imagePath)
        tensorImage = torch.from_numpy(predictionImage)
        
        tensorImage = tensorImage.to(device)
        outputs = predictor([tensorImage])
        
        if len(outputs[0]['boxes']) == 0:
            return
		
        box = outputs[0]["boxes"][0]
        predictedClass = outputs[0]["labels"][0]
        type = "fluke" if predictedClass==1 else "flank"

        # Inflate the boxes back up to the original image size
        box[0] = box[0] * (orgWidth / TARGET_WIDTH)
        box[1] = box[1] * (orgHeight / TARGET_HEIGHT)
        box[2] = box[2] * (orgWidth / TARGET_WIDTH)
        box[3] = box[3] * (orgHeight / TARGET_HEIGHT)
        
        # Inflate the boxes by 1% to help with predictions segmented slightly too tight.
        inflationX = (box[2] - box[0]) * 0.01
        inflationY = (box[3] - box[1]) * 0.01
        
        box[0] = max(0, box[0] - inflationX)
        box[1] = max(0, box[1] - inflationY)
        box[2] = min(orgWidth, box[2] + inflationX)
        box[3] = min(orgHeight, box[3] + inflationY)

        return {
            "path": imagePath,
            "croppingDimensions": box.detach().cpu().tolist(),
            "type": type,
        }
    
    except Exception as e:
        print("WARNING: Failed to segment image: ", imagePath)
        print(e)


def cropIfNeeded(image, imageData, predictor):
    if imageData[2] == []:
        croppedData = cropImage(imageData[1], predictor)
        dimensions = croppedData["croppingDimensions"]
        imageData[2].extend(dimensions)

    box = imageData[2]
    croppedImage = image.crop((box[0], box[1], box[2], box[3]))

    return croppedImage


def loadSegmentationModel(segmentationModelPath):
	model = fasterrcnn_resnet50_fpn(pretrained=True)
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3) 

	model.load_state_dict(torch.load(segmentationModelPath, map_location=torch.device(device)))
	model.to(device)
	model.eval()

	return model


# Define the Triplet Whale Dataset
class TripletWhaleDataset(Dataset):
    def __init__(self, directory, transform=None):
        allWhales = getAllWhales(directory)

        self.transform = transform

        self.cropping = loadSegmentationModel("/db/lpxsf2/outputs/segmentation.pth")

        self.whalesByName = allWhales
        self.whales = []

        for whale in self.whalesByName:
            whales = []

            for imageData in self.whalesByName[whale]:
                try:
                    cropIfNeeded(Image.open(imageData[1]).convert('RGB'), imageData, self.cropping)

                    if (imageData[2] != []):
                        self.whales.append(imageData)
                        whales.append(imageData)

                except Exception as e:
                    print("WARNING: Failed to crop image: ", imageData[1], e)

            if len(whales) < 0:
                self.whalesByName[whale] = whales




    def __len__(self):
        return len(self.whales)

    def __getitem__(self, index):
        anchor = self.whales[index]
        positive = random.choice(self.whalesByName[anchor[0]])

        negative = random.choice(self.whales)
        while negative[0] == anchor[0]:
            negative = random.choice(self.whales)

        anchor_img = cropIfNeeded(Image.open(anchor[1]).convert('RGB'), anchor, self.cropping)
        positive_img = cropIfNeeded(Image.open(positive[1]).convert('RGB'), positive, self.cropping)
        negative_img = cropIfNeeded(Image.open(negative[1]).convert('RGB'), negative, self.cropping)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img
