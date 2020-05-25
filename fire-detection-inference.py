import torch, torchvision
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os, glob

from PIL import Image, ImageDraw, ImageFont

# Applying Transforms to the Data
mean = [0.485, 0.456, 0.406]
variance = [0.229, 0.224, 0.225]

image_transforms = {'test':transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean, variance)
    ])
}

dataset = 'data'
test_dir = os.path.join(dataset, 'test')
# Load data from folders
data = {
    'test': datasets.ImageFolder(root=test_dir,
        transform=image_transforms['test'])
}

# Get a mapping of the indices to the class names, in order to see the output classes of the test images
idx_to_class = {v:k for k, v in data['test'].class_to_idx.items()}

def computeTestAccuracy(model, loss_criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_acc = 0.0
    test_loss = 0.0

    # No gradient tracking needed for testing
    with torch.no_grad():
        # set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)

            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/test_data_size 
    avg_test_acc = test_acc/test_data_size

    print("Test accuracy : " + str(avg_test_acc))

def predict(model, test_image_path):
    img_name = test_image_path.split('/')[-1]
    test_image = Image.open(test_image_path)
    imwidth, imheight = test_image.size
    # print(imwidth, imheight)
    transform = image_transforms['test']
    # plt.imshow(test_image)
    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        # print(out)
        ps = torch.exp(out)
        topk, topclass = ps.topk(2, dim=1)
        # print(topk)
        # print(topclass)
        for i in range(2):
            # print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])
            myclass = idx_to_class[topclass.cpu().numpy()[0][i]]
            myscore = topk.cpu().numpy()[0][i]
            font = ImageFont.truetype("arial.ttf", int(0.03*imwidth))
            d = ImageDraw.Draw(test_image)
            d.text((10,0.05*imheight*i), '{}: {}, Score: {:.2f}'.format(i+1, myclass, myscore),font=font, fill=(0,0,255))
        test_image.save('data/output/' + img_name)


# Test a particular model on a test image
test_dir = 'data/images'
model = torch.load('pre-trained-model/forest-fire_model_24.pt', map_location=torch.device('cpu'))

for img in glob.glob(test_dir + '/*.jpg'):
    predict(model, img)

# Load Dat from folders (Calculate Test  accuracy)
# computeTestAccuracy(model, loss_func)