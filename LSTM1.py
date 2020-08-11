# LSTM1 for touchData

#This script accepts experimental spike data from a day of touch experiments
#and runs them through a single layer LSTM classifier. Each data point is a
#vector of length N (total number of 3b neurons) with each entry being that neurons
#spike count during a window of time, w. The LSTM seeks to classify each data point
# as occuring during a touch stimulus or during a period of rest (no sitmulus).

# Written by Alexa Aucoin

#import packages
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.io import loadmat

torch.manual_seed(1)

#prepare data
fulldata = loadmat('touchData_w25.mat')
fulldata.keys()
N = fulldata['fulldata'].shape[1]-1
A = fulldata['fulldata']
np.random.shuffle(A)

print(A)
data = A[:,0:49]
tags = (A[:,49:]+1)/2

# Pytorch's LSTM expects
# all of its inputs to be 3D tensors. The semantics of the axes of these
# tensors is important. The first axis is the sequence itself, the second
# indexes instances in the mini-batch, and the third indexes elements of
# the input.
split = round(fulldata['fulldata'].shape[0]*.005)
train_x = np.transpose([data[1:split]],[1,0,2]).tolist()
train_y = tags[1:split].tolist()
test_x = np.transpose([data[split:]] ,[1,0,2]).tolist()
test_y = tags[split:].tolist()

inputs = tuple(zip(train_x,train_y))

# train_x = np.transpose(torch.tensor([data[1:split]]),[1,0,2]).float()
# train_y = tags[1:split]
# test_x = np.transpose(torch.tensor([data[split:]]) ,[1,0,2]).float()
# test_y = data[split:]
# torch.tensor(inputs)
#
# lstm = nn.LSTM(N, 2) #input size N, output size 1
# inputs = train_x.float()
# hidden = (torch.randn(1, 1, 2), torch.randn(1, 1, 2))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)
# print(out)
# print(hidden)

#########################################################
EMBEDDING_DIM = 1
HIDDEN_DIM = 30

# nn.Linear ( feature_size_from_previous_layer , 2)
# criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def prep_seq(seq):
    return torch.tensor([seq]).float()

def prep_tag(seq):
    return torch.tensor(seq).long()


######################################################################
#create the model

class LSTMmodel(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers=1):
        super(LSTMmodel, self).__init__()
        self.hidden_dim=hidden_dim
        #LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers)
        #Final, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = 1
        # get LSTM outputs
        lstm_output, (h,c) = self.lstm(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        lstm_output = lstm_output.view(-1, self.hidden_dim)

        # get final output
        model_output = self.fc(lstm_output)

        return model_output, (h,c)

######################################################################
# Train the model:
num_class = 2
batch_size = 1
hidden_size = 30
num_layers = 1
model = LSTMmodel(N,num_class,hidden_size,num_layers)
state = model.state_dict()
print(state)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inps = torch.tensor([train_x[0]]).float()
    hidden = (torch.randn(1, 1, hidden_size),
              torch.randn(1, 1, hidden_size))
    tag_scores, hidden = model(inps,hidden)
    print(tag_scores)

for epoch in range(10):
    for sequence, tags in inputs:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sequence_in = prep_seq(sequence)
        targets = prep_tag(tags)

        # Step 3. Run our forward pass.
        tag_scores, hidden = model(sequence_in,hidden)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward(retain_graph=True)
        optimizer.step()
    print(epoch)

# See what the scores are after training
with torch.no_grad():
    inps = torch.tensor([train_x[8]]).float()
    tag_scores, hidden = model(inps,hidden)
    print(tag_scores)
