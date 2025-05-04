import flwr as fl
import numpy as np
import argparse, csv, datetime
from model import CNN
import pandas as pd         

ignored_client_log = []
round_metrics       = []     # stores (loss, acc, …) per round
ignored_client_hex_log  = []   # raw hex, for counting attackers
# -------------------------
#Number of Rounds
num_rounds = 7

def median_trim_aggregate(results, trim_frac=0.40): #max 40% of clients ignored
    """Median aggregation with trimming of the worst clients."""
    weights = [fl.common.parameters_to_ndarrays(r.parameters) for _, r in results]
    cids    = [p.cid for p, _ in results]
    k       = max(1, int(np.ceil(trim_frac * len(weights))))   # ≤ 2 for N=10

    # ---- average distance over layers ----
    d_tot = np.zeros(len(weights))
    for layer in zip(*weights):
        med   = np.median(np.stack(layer, axis=0), axis=0)
        dists = np.linalg.norm(
                    (np.stack(layer, axis=0) - med).reshape(len(weights), -1),
                    axis=1)
        d_tot += dists
    d_tot /= len(weights)
    # ---------------------------------------

    worst_idx   = np.argsort(d_tot)[-k:]
    ignored_hex = [cids[i] for i in worst_idx]

    # simply return the per-layer median of all clients
    new_params = fl.common.ndarrays_to_parameters(
        [np.median(np.stack(layer, axis=0), axis=0) for layer in zip(*weights)]
    )
    return new_params, ignored_hex





# -------------------------
# Global tracking
# -------------------------


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
    weights_nd  = [fl.common.parameters_to_ndarrays(r.parameters) for _, r in results]
    client_hex  = [p.cid for p, _ in results]

    layer_out   = []
    client_w    = np.zeros(len(client_hex))

    for layer_weights in zip(*weights_nd):
        stacked = np.stack(layer_weights, axis=0)
        median  = np.median(stacked, axis=0)

        dists   = np.linalg.norm((stacked - median).reshape(len(client_hex), -1), axis=1)

        # -------------  TUNE Acordingly  -------------
        gamma   = 20.0                    # ← steeper drop-off than 5.0
        raw_w   = np.exp(-gamma * dists)  # weight = e^(−γ·d)
        raw_w  /= raw_w.sum()
        # -----------------------------------------------

        client_w += raw_w
        weighted_avg = np.average(stacked, axis=0, weights=raw_w)
        layer_out.append(weighted_avg)

    # normalise over layers
    client_w /= client_w.sum()

    # ----- choose a “low weight” cut-off relative to fairness ----------
    fair        = 1.0 / len(client_hex)      # ideal equal share (=0.10 for N=10)
    low_frac    = 0.7                        # keep 60% of clients
    threshold   = low_frac * fair            # → 0.08 for N=10
    ignored_hex = [cid for cid, w in zip(client_hex, client_w) if w < threshold]
    # ------------------------------------------------------------------

    print("[DEBUG] client weights:", {cid[:6]: f"{w:.3f}" for cid, w in zip(client_hex, client_w)})

    aggregated  = fl.common.ndarrays_to_parameters(layer_out)
    return aggregated, ignored_hex



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
        #aggregated_params, ignored_hex = median_aggregate(results)
        aggregated_params, ignored_hex = median_trim_aggregate(results) 

        ignored_readable = [self.cid_map.get(x, x[:6]) for x in ignored_hex]
        print(f"[Median] Ignored this round: {ignored_readable}")
        
        ignored_client_log.append(ignored_readable)
        ignored_client_hex_log.append(ignored_hex) 

        return aggregated_params, {}
    """
    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}
        loss       = np.mean([res.loss for _, res in results])
        accuracy   = np.mean([res.metrics["accuracy"] for _, res in results])
        return float(loss), {"accuracy": float(accuracy)}
    """
    
    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}
        loss = np.mean([res.loss for _, res in results])
        acc  = np.mean([res.metrics["accuracy"] for _, res in results])
        round_metrics.append((rnd, loss, acc))          # <- NEW
        return float(loss), {"accuracy": float(acc)}

    



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
        #ignored_client_log.append(ignored_readable)
        ignored_client_log.append(ignored_readable)
        ignored_client_hex_log.append(ignored_hex)

        return aggregated, {}
    
    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}
        loss = np.mean([res.loss for _, res in results])
        acc  = np.mean([res.metrics["accuracy"] for _, res in results])
        round_metrics.append((rnd, loss, acc))          

        return float(loss), {"accuracy": float(acc)}



class FedAvgWithEval(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cid_map = {}                 

    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}
        losses      = [res.loss for _, res in results]
        accuracies  = [res.metrics["accuracy"] for _, res in results]
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
        config=fl.server.ServerConfig(num_rounds) #number of rounds
    )
# choose a file name by strategy

# build a fallback mapping from the hex in attacker_ids.txt
if not strategy.cid_map:
    strategy.cid_map = {h: f"Client {i+1}"           # 1 … N
                        for i, h in enumerate(set().union(*ignored_client_hex_log))}

tag = {
    "median":   "Median_flipping",
    "weighted": "Weighted_flipping",
    "fedavg":   "FedAvg_flipping",
}[args.strategy]

ts   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
path = f"{tag}_{ts}.csv"


hex_to_num = {}
if hasattr(strategy, "cid_map"):
    hex_to_num = {h: int(lbl.split()[-1]) for h, lbl in strategy.cid_map.items()}
else:
    # fallback: generate mapping for FedAvg (no defense)
    all_hex = set().union(*ignored_client_hex_log) if ignored_client_hex_log else set()
    hex_to_num = {h: i+1 for i, h in enumerate(all_hex)}

rows = []


for i, (rnd, loss, acc) in enumerate(round_metrics):
    if args.strategy == "fedavg":
        # No clients ignored in FedAvg
        rows.append([rnd, loss, acc, 0, 0, 0])
    else:
        ignored_readable = ignored_client_log[i]
        ignored_hex      = ignored_client_hex_log[i]

        att = [
            h for h in ignored_hex
            if hex_to_num.get(h) is not None and str(hex_to_num[h]) in attacker_ids
        ]
        ben = [
            h for h in ignored_hex
            if hex_to_num.get(h) is not None and str(hex_to_num[h]) not in attacker_ids
        ]

        print("[DEBUG] attacker_ids loaded:", attacker_ids)
        print("[DEBUG] hex_to_num map:", hex_to_num)
        print("[DEBUG] ignored_hex:", ignored_hex)

        rows.append([rnd, loss, acc, len(ignored_readable), len(att), len(ben)])


df = pd.DataFrame(rows, columns=[
        "round", "loss", "accuracy",
        "ignored_total", "ignored_attackers", "ignored_benign"
     ])
df.to_csv(path, index=False)
print(f"✅ CSV report saved as {path}")



