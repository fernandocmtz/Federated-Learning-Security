import flwr as fl
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import CNN
import numpy as np
import ssl
import argparse


BATCH_SIZE = 32


# Disable SSL verification to avoid dataset download errors
ssl._create_default_https_context = ssl._create_unverified_context  


# CHANGE NEW Load data with optional label flipping for adversarial clients
def load_data(do_label_flip=False):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)

    if do_label_flip:

        # --- count originals before flipping ---
        orig_labels = train_data.targets.clone()

        print("[INFO] Flipping labels in this client...")
        for i in range(len(train_data.targets)):
            
            train_data.targets[i] = (train_data.targets[i] + 1) % 10  # Flip labels 100% of labels
    
    # --- verify ---
        flips = (train_data.targets != orig_labels).sum().item()
        total = len(train_data.targets)
        pct   = 100 * flips / total
        print(f"[POISON]   flipped {flips}/{total} labels  ({pct:.1f} %)")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


def train(model, train_loader, epochs=1):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device) # Move data to GPU
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()


class MNISTClient(fl.client.NumPyClient):
    def __init__(self, cid: int, is_attacker: bool = False):
        self.cid = cid            
        self.model = CNN()
        self.train_loader = load_data(do_label_flip=is_attacker)



    def get_parameters(self, config):
        self.model.to("cpu") # Move model to CPU for parameter retrieval
        # Convert model parameters to NumPy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    

    def fit(self, parameters, config):
        # Set the model parameters
        for param, new_param in zip(self.model.state_dict().values(), parameters):
            param.data = torch.tensor(new_param)
        #for p, new_p in zip(self.model.state_dict().values(), parameters):
        #    p.data = torch.tensor(new_p)

        # Train the model
        updated_weights = train(self.model, self.train_loader)

        # Convert to list of NumPy arrays
        weights_numpy = [val.cpu().numpy() for _, val in updated_weights.items()]

        return weights_numpy, len(self.train_loader.dataset), {"client_id": self.cid}
    
    
    def evaluate(self, parameters, config):
    # Load the server parameters into the local model
        for param, new_param in zip(self.model.state_dict().values(), parameters):
            param.data = torch.tensor(new_param)

        self.model.eval()

        # Load test data
        transform = transforms.Compose([transforms.ToTensor()])
        test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)


        correct = 0
        total = 0
        loss_total = 0.0

        with torch.no_grad():     
            
           
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)  # Move data to GPU
                outputs = self.model(images)  


                loss = F.cross_entropy(outputs, labels)
                loss_total += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = loss_total / total

        return float(avg_loss), total, {"accuracy": float(accuracy)}



# Main function to start the client

if __name__ == "__main__":
    # This client is the attacker
    # Parse command line arguments for client ID and attack flag

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Client ID (e.g., 1 to 10)")
    parser.add_argument("--attack", action="store_true", help="Enable label flipping for this client")
    args = parser.parse_args()

    print(f"🚀 Starting client {args.id} | Attacker: {args.attack}")
    
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=MNISTClient(cid=args.id, is_attacker=args.attack)
        #client=MNISTClient(is_attacker=args.attack)
    )

class MNISTClient(fl.client.NumPyClient):
    def __init__(self, cid: int, is_attacker: bool = False):
    #def __init__(self, cid, is_attacker=False):
        self.cid  = cid          # save 1-based numeric id
        ...                     

    # Flower asks for properties once at startup
    def get_properties(self, config):
        return {"client_id": str(self.cid)}


