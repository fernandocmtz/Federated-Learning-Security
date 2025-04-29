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
    # convert each client update to ndarray list
    weights = [fl.common.parameters_to_ndarrays(res.parameters) for _, res in results]
    client_hex = [proxy.cid for proxy, _ in results]

    layer_medians = []
    ignored_clients = set()

    for layer_idx, layer_weights in enumerate(zip(*weights)):
        stacked = np.stack(layer_weights, axis=0)
        median = np.median(stacked, axis=0)

        # IQR-based cutoff
        dists = np.linalg.norm((stacked - median).reshape(stacked.shape[0], -1), axis=1)
        q1, q3 = np.percentile(dists, [25, 75])
        cutoff = q3 + 1.5 * (q3 - q1)
        bad_idx = np.where(dists > cutoff)[0]

        ignored_clients.update(client_hex[i] for i in bad_idx)
        layer_medians.append(median)

    return fl.common.ndarrays_to_parameters(layer_medians), list(ignored_clients)


# -------------------------
# Weighted Median Aggregation
# -------------------------

def weighted_median_aggregate(results):
    weights_nd   = [fl.common.parameters_to_ndarrays(res.parameters) for _, res in results]
    client_hex   = [proxy.cid for proxy, _ in results]

    layer_out   = []
    client_w    = np.zeros(len(client_hex))

    for layer_weights in zip(*weights_nd):
        stacked = np.stack(layer_weights, axis=0)       # shape (N, …)
        median  = np.median(stacked, axis=0)

        dists   = np.linalg.norm((stacked - median).reshape(stacked.shape[0], -1), axis=1)

        median_d = np.median(dists)
        scaled   = dists / (median_d + 1e-12)

        gamma  = 35.0                # the larger, the steeper the drop-off
        raw_w  = np.exp(-gamma * scaled)
        raw_w /= raw_w.sum()
        #raw_w   = 1.0 / (1.0 + dists)
        #raw_w  /= raw_w.sum()                           # normalise

        
        # ------ use np.average: no manual broadcasting needed ------
        weighted_avg = np.average(stacked, axis=0, weights=raw_w)
        # -----------------------------------------------------------

        layer_out.append(weighted_avg)
        client_w += raw_w
    
    client_w /= client_w.sum()                # normalise weights


    fair = 1.0 / len(client_hex)
    ignored_hex = [cid for cid, w in zip(client_hex, client_w) if w < 0.8 * fair]# list clients that got very small overall weight

    
    #ignored_hex = [cid for cid, w in zip(client_hex, client_w) if w < 0.40]

    aggregated = fl.common.ndarrays_to_parameters(layer_out)

    print("[DEBUG] client weights:", {cid[:6]: f"{w:.3f}" for cid, w in zip(client_hex, client_w)})

    return aggregated, ignored_hex

"""
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
"""
# -------------------------
# Strategy Classes
# -------------------------

class MedianAggregationStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(fraction_fit=1.0,
                         min_fit_clients=10,
                         min_available_clients=10)
        self.cid_map = {}          # hex → "Client N"
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # build mapping once per round
        for proxy, fit_res in results:
            hex_cid = proxy.cid
            client_id = fit_res.metrics.get("client_id")   # sent by each client
            if client_id is not None:
                self.cid_map[hex_cid] = f"Client {client_id}"

        # median_aggregate returns (params, ignored_hex_list)
        aggregated_params, ignored_hex = median_aggregate(results)

        ignored_readable = [self.cid_map.get(x, x[:6]) for x in ignored_hex]
        print(f"[Median] Ignored this round: {ignored_readable}")
        
        ignored_client_log.append(ignored_readable)

        return aggregated_params, {}
    
    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}
        loss       = np.mean([res.loss for _, res in results])
        accuracy   = np.mean([res.metrics["accuracy"] for _, res in results])
        return float(loss), {"accuracy": float(accuracy)}
    



class WeightedMedianStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cid_map = {}

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # map proxy.cid → "Client N"
        for proxy, fit_res in results:
            cid_hex   = proxy.cid
            num_id    = fit_res.metrics.get("client_id")
            if num_id is not None:
                self.cid_map[cid_hex] = f"Client {num_id}"

        aggregated, ignored_hex = weighted_median_aggregate(results)
        ignored_readable = [self.cid_map.get(x, x[:6]) for x in ignored_hex]

        print(f"[WeightedMedian] very-low-weight clients this round: {ignored_readable}")
        ignored_client_log.append(ignored_readable)

        return aggregated, {}


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
        strategy = WeightedMedianStrategy(fraction_fit=1.0, min_fit_clients=10, min_available_clients=10)
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
