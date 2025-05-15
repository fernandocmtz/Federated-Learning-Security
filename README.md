# Federated-Learning-Security
This project explores security threats in Federated Learning (FL) by implementing a Label Flipping Attack and using Median Aggregation as a defense. It includes a CNN model, FL server-client setup, and evaluation metrics like accuracy and convergence. Built with PyTorch, FLWR, NumPy, and Matplotlib. 

# Federated-Learning-Security ― Steps

---

## 1.  Prerequisites

| Item | Version tested |
|------|----------------|
| Python | **3.11** (3.9–3.12 work) |
| pip or conda | up-to-date |
| OS | Windows 10/11, macOS 12+, Ubuntu 20.04+ |

GPU is **not** required; everything fits on CPU.

---

## 2.  Set-up (once)

```bash
# clone the repo (or unzip the hand-in files)
git clone https://github.com/your-username/Federated-Learning-Security.git
cd Federated-Learning-Security

# create a fresh env
python -m venv venv          # or: conda create -n fl_sec python=3.11
source venv/bin/activate     #   (Windows: venv\Scripts\activate)

# install deps
pip install -r requirements.txt

------------------------------------------------------------------------

---Launch the FL experiment---

Start the server (choose a strategy) (Terminal 1)
# FedAvg (baseline, no defence)
python server.py --strategy fedavg

# Median (hard trim, top-k outliers)
python server.py --strategy median

# Weighted-Median (soft weights + low-frequency cut)
python server.py --strategy weighted
--------------------------------------------------------------------------
You can select the number of Rounds, clients and attackers on the server.py file

TOTAL_CLIENTS = 10          # total number of clients
SAMPLE_SIZE   = 10            # clients per round
NUM_ROUNDS    = 12         # number of training rounds

-------------------------------------------------------------------------
The server:
•	listens on 127.0.0.1:8080
•	Gathers the updates per client
•	Prints the plot with the dropped clients
----------------------------------------------------------------------
Launch the ten clients (Terminal 2)

python launch_clients.py
--------------------------------------------------------------
At the end of every round it will print and plot the dropped clients. 


