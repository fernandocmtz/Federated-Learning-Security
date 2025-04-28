import flwr as fl
import numpy as np
import argparse
import csv
from model import CNN

# -------------------------
# Global tracking
# -------------------------
ignored_client_log = []  # stores ignored client IDs per round (for median aggregation)

# Load attacker IDs from file
try:
    with open("attacker_ids.txt", "r") as f:
        attacker_ids = set(f.read().strip().split(","))
except FileNotFoundError:
    attacker_ids = set()

# -------------------------
# Median Aggregation
# -------------------------

def median_aggregate(results):
    """Aggregate weights using median and track ignored clients."""
    weights = [fl.common.parameters_to_ndarrays(res.parameters) for _, res in results]
    client_ids = [client.cid for client, _ in results]
    layer_medians = []
    ignored_indices = set()

    for i, layer_weights in enumerate(zip(*weights)):
        stacked = np.stack(layer_weights, axis=0)
        median = np.median(stacked, axis=0)

        dists = np.linalg.norm((stacked - median).reshape(stacked.shape[0], -1), axis=1)
        threshold = np.percentile(dists, 70)

        for idx, dist in enumerate(dists):
            if dist > threshold:
                ignored_indices.add(idx)

        layer_medians.append(median)

    ignored_clients = [client_ids[i] for i in ignored_indices]
    ignored_client_log.append(ignored_clients)

    attacker_ignored = [cid for cid in ignored_clients if cid in attacker_ids]
    benign_ignored = [cid for cid in ignored_clients if cid not in attacker_ids]

    print(f"[Median Aggregation] Ignored clients this round: {ignored_clients}")
    print(f"[Tracking] Ignored attackers this round: {attacker_ignored}")
    print(f"[Tracking] Ignored benign clients this round: {benign_ignored}")

    return fl.common.ndarrays_to_parameters(layer_medians)

# -------------------------
# Weighted Median Aggregation
# -------------------------

def weighted_median_aggregate(results):
    weights = [fl.common.parameters_to_ndarrays(res.parameters) for _, res in results]
    layer_weighted_avg = []

    for layer_weights in zip(*weights):
        stacked = np.stack(layer_weights, axis=0)
        median = np.median(stacked, axis=0)

        dists = np.linalg.norm((stacked - median).reshape(stacked.shape[0], -1), axis=1)
        dists += 1e-10

        inv_weights = 1.0 / dists
        normalized_weights = inv_weights / np.sum(inv_weights)

        weighted_avg = np.tensordot(normalized_weights, stacked, axes=1)
        layer_weighted_avg.append(weighted_avg)

    print("[Weighted Median Aggregation] applied weighted average based on inverse distances.")
    return fl.common.ndarrays_to_parameters(layer_weighted_avg)

# -------------------------
# Strategy Classes
# -------------------------

class MedianAggregationStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(fraction_fit=1.0, min_fit_clients=10, min_available_clients=10)

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}
        return median_aggregate(results), {}

    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}
        losses = [res.loss for _, res in results]
        accuracies = [res.metrics["accuracy"] for _, res in results]
        return float(np.mean(losses)), {"accuracy": float(np.mean(accuracies))}

class WeightedMedianAggregationStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(fraction_fit=1.0, min_fit_clients=10, min_available_clients=10)

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}
        return weighted_median_aggregate(results), {}

    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}
        losses = [res.loss for _, res in results]
        accuracies = [res.metrics["accuracy"] for _, res in results]
        return float(np.mean(losses)), {"accuracy": float(np.mean(accuracies))}

class FedAvgWithEval(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}
        losses = [res.loss for _, res in results]
        accuracies = [res.metrics["accuracy"] for _, res in results]
        return float(np.mean(losses)), {"accuracy": float(np.mean(accuracies))}

# -------------------------
# Entry Point
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, choices=["fedavg", "median", "weighted"], required=True)
    args = parser.parse_args()

    if args.strategy == "median":
        print("[Server] Using Median Aggregation Strategy")
        strategy = MedianAggregationStrategy()
    elif args.strategy == "weighted":
        print("[Server] Using Weighted Median Aggregation Strategy")
        strategy = WeightedMedianAggregationStrategy()
    else:
        print("[Server] Using FedAvg (no defense)")
        strategy = FedAvgWithEval(fraction_fit=1.0, min_fit_clients=10, min_available_clients=10)

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10)
    )

    # Print round-by-round
    print("\n=== Ignored clients by round ===")
    for rnd, ignored in enumerate(ignored_client_log, 1):
        print(f"Round {rnd}: {ignored}")

    # Save CSV summary
    with open("ignored_summary.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Round", "Ignored_Total", "Ignored_Attackers", "Ignored_Benigns"])

        for rnd, ignored in enumerate(ignored_client_log, 1):
            ignored_attackers = [cid for cid in ignored if cid in attacker_ids]
            ignored_benign = [cid for cid in ignored if cid not in attacker_ids]

            writer.writerow([
                rnd,
                len(ignored),
                len(ignored_attackers),
                len(ignored_benign),
            ])

    print("✅ CSV report saved as ignored_summary.csv")
