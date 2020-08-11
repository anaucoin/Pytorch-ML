import torch
import torchvision
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.io import loadmat

################################################################################
class AE(nn.Module):
    # def __init__(self, **kwargs):
    #     super().__init__()
    #     self.encoder_hidden_layer = nn.Linear(
    #         in_features=kwargs["input_shape"], out_features=10
    #     )
    #     self.encoder_output_layer = nn.Linear(
    #         in_features=10, out_features=10
    #     )
    #     self.decoder_hidden_layer = nn.Linear(
    #         in_features=10, out_features=10
    #     )
    #     self.decoder_output_layer = nn.Linear(
    #         in_features=10, out_features=kwargs["input_shape"]
    #     )
    #
    # def forward(self, features):
    #     activation = self.encoder_hidden_layer(features)
    #     activation = torch.relu(activation)
    #     code = self.encoder_output_layer(activation)
    #     code = torch.relu(code)
    #     activation = self.decoder_hidden_layer(code)
    #     activation = torch.relu(activation)
    #     activation = self.decoder_output_layer(activation)
    #     reconstructed = torch.relu(activation)
    #     return reconstructed

    #

    def __init__(self, **kwargs):
        super().__init__()
        self.code_size = kwargs["code_size"]
        self.input_shape = kwargs["input_shape"]

        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=(3,1), stride = (1,1))
        self.enc_cnn_2 = nn.Conv2d(10, 10, kernel_size=(3,1), stride= (1,1))
        self.enc_linear_1 = nn.Linear(4 * 4 * 10, 15)
        self.enc_linear_2 = nn.Linear(15, self.code_size)

        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 15)
        self.dec_linear_2 = nn.Linear(15, input_shape)

    def forward(self, features):
        code = self.encode(features)
        out = self.decode(code)
        return out, code

    def encode(self, features):
        code = self.enc_cnn_1(features)
        code = F.selu(F.max_pool2d(code, 2))

        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))

        code = code.view([features.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code

    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out.view([code.size(0), 1, input_shape])
        return out


###############################################################################

torch.manual_seed(1)

numdays = 2
N = [0]*numdays
#prepare data
fulldata = loadmat('touchData_w25.mat')
fulldata.keys()
N[0] = fulldata['fulldata'].shape[1]-1
A = fulldata['fulldata']
np.random.shuffle(A)
data = A[:,0:49]
tags = (A[:,49:]+1)/2

# Pytorch's LSTM expects
# all of its inputs to be 3D tensors. The semantics of the axes of these
# tensors is important. The first axis is the sequence itself, the second
# indexes instances in the mini-batch, and the third indexes elements of
# the input.
split = 128*48
train_x = np.transpose([data[1:split]],[1,0,2]).tolist()
train_y = tags[1:split].tolist()
test_x = np.transpose([data[split:]],[1,0,2]).tolist()
test_y = tags[split:].tolist()


#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu

#inputs = tuple(zip(train_x,train_y))
code_dim = 10
encoder = AE(input_shape=N[0], code_size = code_dim).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()

train_loader = torch.utils.data.DataLoader(
    torch.tensor(train_x), batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    torch.tensor(test_x), batch_size=1, shuffle=False, num_workers=4
)


train_loader


epochs = 1000
for epoch in range(epochs):
    loss = 0
    for batch_features in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, N[0]).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = encoder(batch_features.float())

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features.float())

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))



###############################################################################
# Second day auto-encoder
fulldata = loadmat('071819_touchData_w25.mat')
fulldata.keys()
N[1] = fulldata['fulldata'].shape[1]-1
N[1]
A = fulldata['fulldata']
np.random.shuffle(A)
data = A[:,0:N[1]]
tags = (A[:,N[1]:]+1)/2

split = 128*48
train_x = np.transpose([data[1:split]],[1,0,2]).tolist()
train_y = tags[1:split].tolist()
test_x = np.transpose([data[split:]],[1,0,2]).tolist()
test_y = tags[split:].tolist()

#inputs = tuple(zip(train_x,train_y))
encoder = AE(input_shape=N[1]).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()

train_loader = torch.utils.data.DataLoader(
    torch.tensor(train_x), batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    torch.tensor(test_x), batch_size=1, shuffle=False, num_workers=4
)

epochs = 1000
for epoch in range(epochs):
    loss = 0
    for batch_features in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, N[1]).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = encoder(batch_features.float())

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features.float())

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
