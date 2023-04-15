import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

'''
AIM: given a list of training and test data about the daily temperatures in World War 2,
get a line of best fit approximating the relationship between min and max tempeatures
'''

# hyperparameter values (guesstimate here, not identifying them 100% correctly)
learning_rate = 0.001
iteration_num = 375

# getting training and test data
weather_file = pd.read_csv('/Users/ananya/Documents/Grab_AI_Lab/linear_regression_data/weather.csv')

# creating some fake training and test data
x = np.array(weather_file.iloc[0:10000].loc[:, "MinTemp"], dtype=np.float32)
y = np.array(weather_file.iloc[0:10000].loc[:, "MaxTemp"], dtype=np.float32)

train_X = torch.from_numpy(x.reshape(len(x), 1))
train_Y = torch.from_numpy(y.reshape(len(y), 1))

# creating the linear regression model
linear_model = nn.Linear(1, 1)

# create the loss and optimiser for gradient descent
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr=learning_rate)

# training model, forward and back propagation
for i in range(iteration_num):

    # forward prop + loss function
    pred = linear_model(train_X)
    loss = loss_function(pred, train_Y)

    # backward prop 
    optimizer.zero_grad()
    grads = loss.backward()
    optimizer.step()

# plot graph
new_pred = linear_model(train_X).detach().numpy()
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, new_pred, label='Fitted line')
plt.xlabel('Min Temperature (°C)')
plt.ylabel('Max Temperature (°C)')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(linear_model.state_dict(), 'model.ckpt')






