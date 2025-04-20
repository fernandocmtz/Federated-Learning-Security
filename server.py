import flwr as fl
import numpy as np
import torch
from model import CNN
import argparse

USE_MEDIAN_DEFENSE = False  # Set to True to enable median aggregation defense


# -------------------------
# Median Aggregation Utils
# -------------------------

def median_aggregate(results):
    """Aggregate weights using median."""
    
    import numpy as np

    # Extract parameter lists from all clients
    weights = [fl.common.parameters_to_ndarrays(res.parameters) for _, res in results]
    num_clients = len(weights)

    # Stack weights layer-by-layer to compute medians
    layer_medians = []
    ignored_counts = []

    for i, layer_weights in enumerate(zip(*weights)):
        stacked = np.stack(layer_weights, axis=0)  # Shape: (num_clients, ...)
        median = np.median(stacked, axis=0)

        # Calculate L2 distance from each client's update to the median
        dists = np.linalg.norm((stacked - median).reshape(stacked.shape[0], -1), axis=1)
        # Calculate the threshold for outlier detection (e.g., median distance)
        # Here we use the median distance as a cutoff for outliers
        threshold = np.percentile(dists, 70)  # Median distance as cutoff

        # Count how many clients are farther than the threshold
        ignored = np.sum(dists > threshold)
        ignored_counts.append(ignored)

        layer_medians.append(median)

    total_ignored = max(ignored_counts)  # Approximate: max clients ignored across layers
    print(f"[Median Aggregation] Ignored approx {total_ignored} outlier client(s) this round.")

    return fl.common.ndarrays_to_parameters(layer_medians)


# -------------------------
# Custom Strategy
# -------------------------

class MedianAggregationStrategy(fl.server.strategy.FedAvg):
    """Custom strategy that uses median aggregation."""
     
    #def __init__(self): Set for 10 clients 
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,             # Use all available clients
            min_fit_clients=10,           # Require 10 clients to run each round
            min_available_clients=10      # Must have at least 10 clients connected
        )

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}
        aggregated_parameters = median_aggregate(results)
        return aggregated_parameters, {}
    
    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}

        losses = [res.loss for _, res in results]
        accuracies = [res.metrics["accuracy"] for _, res in results]

        avg_loss = float(np.mean(losses))
        avg_accuracy = float(np.mean(accuracies))

        return avg_loss, {"accuracy": avg_accuracy}

# -------------------------
# Server Start
# -------------------------

if __name__ == "__main__":
    
    if USE_MEDIAN_DEFENSE:
        print("[Server] Median aggregation defense ENABLED.")
        strategy = MedianAggregationStrategy()
    else:
        print("[Server] FedAvg (no defense) ENABLED.")
        class FedAvgWithEval(fl.server.strategy.FedAvg):
            def aggregate_evaluate(self, rnd, results, failures):
                if not results:
                    return None, {}

                losses = [res.loss for _, res in results]
                accuracies = [res.metrics["accuracy"] for _, res in results]

                avg_loss = float(np.mean(losses))
                avg_accuracy = float(np.mean(accuracies))

                return avg_loss, {"accuracy": avg_accuracy}

        strategy = FedAvgWithEval(
            fraction_fit=1.0,
            min_fit_clients=10,
            min_available_clients=10
        )
        """strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=10,
            min_available_clients=10
        )"""

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10)
    )
    

