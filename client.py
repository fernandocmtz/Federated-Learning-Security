import flwr as fl
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import CNN
import numpy as np
import ssl

BATCH_SIZE = 32
LABEL_FLIP_PROB = 0.3  # 30% of labels will be flipped (attack)

# Disable SSL verification to avoid dataset download errors
ssl._create_default_https_context = ssl._create_unverified_context  

""" def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)

    # Introduce label flipping for adversarial clients
    for i in range(len(train_data.targets)):
        if np.random.rand() < LABEL_FLIP_PROB:
            train_data.targets[i] = (train_data.targets[i] + 1) % 10  # Flip labels

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader
"""
# CHANGE NEW Load data with optional label flipping for adversarial clients
def load_data(do_label_flip=False):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)

    if do_label_flip:
        # Flip 30% of the labels randomly
        for i in range(len(train_data.targets)):
            if np.random.rand() < 0.3:
                train_data.targets[i] = (train_data.targets[i] + 1) % 10

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


def train(model, train_loader, epochs=1):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

class MNISTClient(fl.client.NumPyClient):
    #def __init__(self):
    def __init__(self, is_attacker=False):
        self.model = CNN()
        self.train_loader = load_data(do_label_flip=is_attacker)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    """def fit(self, parameters, config):
        for param, new_param in zip(self.model.state_dict().values(), parameters):
            param.data = torch.tensor(new_param)
        return train(self.model, self.train_loader), len(self.train_loader.dataset), {}
    """

    def fit(self, parameters, config):
        # Set the model parameters
        for param, new_param in zip(self.model.state_dict().values(), parameters):
            param.data = torch.tensor(new_param)

        # Train the model
        updated_weights = train(self.model, self.train_loader)

        # Convert to list of NumPy arrays
        weights_numpy = [val.cpu().numpy() for _, val in updated_weights.items()]

        return weights_numpy, len(self.train_loader.dataset), {}


"""if __name__ == "__main__":
    # Ensure correct connection using the latest Flower API
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MNISTClient())
"""

if __name__ == "__main__":
    # This client is the attacker
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=MNISTClient(is_attacker=False)
    )

