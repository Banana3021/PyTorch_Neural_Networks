import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

'''
AIM: given a list of training and test data of images of cats and dogs, this logistic regression
algorithm aims to analyse whether an image is a cat or a dog.
This algorithm makes use of a single hidden layer linear neural network.
'''

# hyperparameter values (also guesstimate here)
batch_size = 75 # splitting the data into batch_size samples
iter_range = 1000
learning_rate = 0.001
input_size = 49152 # size of feature vector for each sample
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

# creating the model
log_model = nn.Linear(input_size, num_classes)

# creating the loss function and the optimiser
loss_function = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(log_model.parameters(), lr=learning_rate)

# training the model
for images, labels in train_loader:
    images = images.reshape(-1, input_size)

    # forward pass
    output = log_model(images)
    loss = loss_function(output, labels)

    # backward pass
    optimiser.zero_grad()
    grads = loss.backward()
    optimiser.step()
    
# checking for correctness against test data
for images, labels in test_loader:
    correct = 0
    total = 0

    images = images.reshape(-1, input_size)
    outputs = log_model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

    print('Accuracy of model: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(log_model.state_dict(), 'model.ckpt')


