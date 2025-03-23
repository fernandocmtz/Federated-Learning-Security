import flwr as fl
import numpy as np
import torch
from model import CNN

# -------------------------
# Median Aggregation Utils
# -------------------------

def median_aggregate(results):
    """Aggregate model weights by computing the median for each layer across clients."""
    weights = [
        fl.common.parameters_to_ndarrays(fit_res.parameters)
        for _, fit_res in results
    ]

    layer_medians = []
    for layer_weights in zip(*weights):
        stacked = np.stack(layer_weights, axis=0)
        median_layer = np.median(stacked, axis=0)
        layer_medians.append(median_layer)

    return fl.common.ndarrays_to_parameters(layer_medians)


# -------------------------
# Custom Strategy
# -------------------------

class MedianAggregationStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}
        aggregated_parameters = median_aggregate(results)
        return aggregated_parameters, {}

# -------------------------
# Server Start
# -------------------------

if __name__ == "__main__":
   fl.server.start_server(
    server_address="127.0.0.1:8080",
    strategy=MedianAggregationStrategy(),
    config=fl.server.ServerConfig(num_rounds=10)

)
