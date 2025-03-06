import flwr as fl
import numpy as np
import torch
from model import CNN

# Define median aggregation function
def median_aggregation(weights_list):
    return np.median(np.array(weights_list), axis=0)

# Define a custom strategy for federated learning
class RobustFedServer(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters = median_aggregation([parameters for parameters, _ in results])
        return aggregated_parameters, {}

if __name__ == "__main__":
    # Use the correct start_server function
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=RobustFedServer(),
    )
