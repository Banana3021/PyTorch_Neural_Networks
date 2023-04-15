import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
AIM: given a list of training and test data of images of cats and dogs, this logistic regression
algorithm aims to analyse whether an image is a cat or a dog.
This algorithm makes use of a 4 layer feedforward neural network.
'''

# hyperparameter values (also guesstimate here)
batch_size = 100 # number of input feature vectors
iter_range = 1000
learning_rate = 0.00001
input_size = 49152 # size of feature vector for each sample
hidden_size = 10 # number of neurons in that unit
num_classes = 2 

# getting the paths to the image folders
training_images = '/Users/ananya/Documents/Grab_AI_Lab/logistic_regression_data/training_set'
test_images = '/Users/ananya/Documents/Grab_AI_Lab/logistic_regression_data/test_set'

# writing out the code to transform the images to be of size 128 by 128
data_transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor()  
])

# creating the train and test data usinhg image folder
train_dataset = datasets.ImageFolder(root=training_images, 
                                  transform=data_transform) 

test_dataset = datasets.ImageFolder(root=test_images, 
                                 transform=data_transform)

# create the data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


feedforward_model = nn.Sequential(nn.Linear(49152, hidden_size),
                      nn.ReLU(),
                      nn.Sigmoid())

# creating the loss function and the optimiser
loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(feedforward_model.parameters(), lr=learning_rate)

# training the model
for images, labels in train_loader:  
    # Move tensors to the configured device
    images = images.reshape(-1, input_size)
    labels = labels
    
    # Forward pass
    outputs = feedforward_model(images)
    loss = loss_function(outputs, labels)
    
    # Backward and optimize
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# checking for correctness against test data
for images, labels in test_loader:
    correct = 0
    total = 0

    images = images.reshape(-1, input_size)
    labels = labels
    outputs = feedforward_model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    print('Accuracy of model: {} %'.format(100 * correct / total))