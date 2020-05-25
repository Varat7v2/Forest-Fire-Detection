import torch, torchvision
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

# APPLYING TRANSFORMS TO THE DATA
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_transforms = {
    'train':transforms.Compose([
        transforms.RandomResizedCrop(size=256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
    'valid':transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
    'test':transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
}

# LOAD THE DATA
# set train and validation direcotry paths
dataset = 'animals-10'
train_dir = os.path.join(dataset, 'train')
valid_dir = os.path.join(dataset, 'valid')
test_dir = os.path.join(dataset, 'test')

# SET BATCH SIZE
batch_size = 32
# SET NUMBER OF CLASSES
num_classes = len(os.listdir(valid_dir))
print(num_classes)

# Load data from folders
data = {
    'train': datasets.ImageFolder(root=train_dir,
        transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_dir,
        transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_dir,
        transform=image_transforms['test'])
}

# Get a mapping of the indices to the class names, in order to see the output classes of the test images
idx_to_class = {v:k for k, v in data['train'].class_to_idx.items()}
print(idx_to_class)

# size of data to be used for calculating average loss and accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

# create iterators for data loaded using dataloader module
train_data_loader = DataLoader(data['train'],
    batch_size=batch_size, shuffle=True)
valid_data_loader = DataLoader(data['valid'],
    batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(data['test'],
    batch_size=batch_size, shuffle=True)

# print the train, validation and test set data sizes
print(train_data_size, valid_data_size, test_data_size)

#  Load a pretrained model --> ResNet50
resnet50 = models.resnet50(pretrained=True)
# resnet50 = resnet50.to('cuda')

# freeze model parameters
for param in resnet50.parameters():
    param.requires_grad = False

# change the final layer of Resnet50 for transfer learning
fc_inputs = resnet50.fc.in_features

resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes),
    nn.LogSoftmax(dim=1) #for using NLLLoss()
    )

# convert model to be used on GPU
resnet50 = resnet50.to('cuda')

# Define optimizer and loss function
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())

# def predict(model, test_image_name):
#     transform = image_transforms['test']

#     test_image = Image.open(test_image_name)
#     plt.imshow(test_image)
#     test_image_tensor = transform(test_image)

#     if torch.cuda.is_available():
#         test_image_tensor = test_image_tensor.view(1,3,224,224).cuda()
#     else:
#         test_image_tensor = test_image_tensor.view(1,3,224,224)

#     with torch.no_grad():
#         model.eval()
#         # model outputs log probabilites
#         out = model(test_image_tensor)
#         ps = torch.exp(out)
#         topk, topclass = ps.topk(1, dim=1)
#         print("Output class: ", idx_to_class[topclass.cpu().numpy()[0][0]])

def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        # set to training mode
        model.train()

        # loss and accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # clean existing gradients
            optimizer.zero_grad()

            # forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # compute loss
            loss = loss_criterion(outputs, labels)

            # backpropagate the gradients
            loss.backward()

            # update the parameters
            optimizer.step()

            # compute the total loss for the batch and add it to train_loss
            train_loss += loss.item()*inputs.size(0)

            # compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item()*(inputs.size(0))
            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy:{:.4f}".format(
            #   i, loss.item(), acc.item()))

        # Validation - No gradient tracking needed
        with torch.no_grad():
            # set to evaluation mode
            model.eval()

            # validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass
                outputs = model(inputs)

                # compute loss
                loss = loss_criterion(outputs, labels)
                # compute the total loss for the batch and add it to valid loss
                valid_loss += loss.item()*inputs.size(0)
                # calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                # convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                # compute the total accuracy in the whole batch nd add to valid_acc
                valid_acc += acc.item()*inputs.size(0)
                
                # print validation loss and accruacy at each step
                # print("Validation Batch number:{:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc/float(train_data_size)

        # find average validation loss and accuracy
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / float(valid_data_size)

        history.append([avg_train_loss, avg_valid_acc, avg_valid_loss, avg_valid_acc])

        epoch_end = time.time()

        print("Epoch: {:03d}, Training Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss{:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, (epoch_end-epoch_start)))

        # save if the model has best accuracy till now
        torch.save(model, 'models/' + dataset+'_model_'+str(epoch)+'.pt')
    return model, history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print the model to be trained
summary(resnet50, input_size=(3,244,244), batch_size=batch_size, device='cuda')

# TRAIN THE MODEL FOR 'X' EPOCHS
num_epochs = 25
trained_model, history = train_and_validate(resnet50, loss_func, optimizer, num_epochs)

torch.save(history, 'models/' + dataset+'_history.pt')

# VISUALIZATION OF TRAINING AND VALIDATION LOSS AND ACCURACY
history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.savefig('plots/' + dataset+'_loss_curve.png')
plt.show()

plt.plot(history[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.savefig('plots/' + dataset+'_accuracy_curve.png')
plt.show()