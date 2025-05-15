# ---------- Imports -----------------------------------------------------------
import argparse, datetime, csv
import numpy as np
import flwr as fl
import matplotlib.pyplot as plt
import os
import pandas as pd
from model import CNN         

# ---------- Constants ---------------------------------------------------------
TOTAL_CLIENTS = 10          # total number of clients
SAMPLE_SIZE   = 10            # clients per round
NUM_ROUNDS    = 12         # number of training rounds

# ---------- Global logging ----------------------------------------------------
ignored_client_log      : list[list[str]] = []   # readable names
ignored_client_hex_log  : list[list[str]] = []   # raw hex IDs
round_metrics           : list[tuple[int,float,float]] = []

# -----------------------------------------------------------------------------#
#                               Helper functions                               #
# -----------------------------------------------------------------------------#
    

def plot_weight_profile(rnd, w, thr, ignored_hex, cid_map, hexs, *, block=False):
    xs   = np.arange(len(w))
    lbls = [cid_map.get(h, h[:6]) for h in hexs]
    plt.figure(figsize=(7,3))
    plt.scatter(xs, w, c="tab:blue")
    plt.axhline(thr, ls="--", c="k", lw=.8, label=f"cut ≈ {thr:.3f}")
    plt.scatter([xs[lbls.index(cid_map.get(h,h[:6]))] for h in ignored_hex],
                [w [lbls.index(cid_map.get(h,h[:6]))] for h in ignored_hex],
                c="tab:red", label="ignored")
    plt.xticks(xs, lbls, rotation=45, ha="right", fontsize=7)
    plt.ylabel("normalised client weight")
    plt.title(f"Round {rnd}: client weights")
    plt.legend(); plt.tight_layout(); plt.grid()
    if block: plt.show(); plt.savefig(f"plots/weights_r{rnd:02d}.png"); plt.close()
    else:     plt.savefig(f"plots/weights_r{rnd:02d}.png"); plt.close()

def median_trim_aggregate(
        
    results,
    rnd: int,
    trim_frac: float = 0.35,    # fraction of clients to ignore
    *,
    cid_map: dict[str, str] | None = None,
    plot: bool = False,
):
    """Median aggregation that discards the k worst clients by distance."""
    weights = [fl.common.parameters_to_ndarrays(r.parameters) for _, r in results]
    cids    = [p.cid for p, _ in results]
    k       = max(1, int(np.ceil(trim_frac * len(weights))))

    # average layer‑wise distance to layer median
    d_tot = np.zeros(len(weights))
    for layer in zip(*weights):
        med   = np.median(np.stack(layer, axis=0), axis=0)
        d_tot += np.linalg.norm(
            (np.stack(layer, axis=0) - med).reshape(len(weights), -1), axis=1
        )
    d_tot /= len(weights)

    worst_idx   = np.argsort(d_tot)[-k:]    # indices of clients to ignore
    ignored_hex = [cids[i] for i in worst_idx]  # hex IDs of ignored clients

    # Plot per round for discarded clients
    if plot:
        labels = [cid_map.get(h, h[:6]) if cid_map else h[:6] for h in cids]
        cut    = d_tot[worst_idx[0]]
        plt.figure(figsize=(7,3))
        plt.scatter(range(len(d_tot)), d_tot, c="tab:blue")
        plt.scatter(worst_idx, d_tot[worst_idx], c="tab:red", label="trimmed")
        plt.axhline(cut, ls="--", c="k", lw=0.8, label=f"cutoff ≈{cut:.3f}")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
        plt.ylabel("avg L2 distance to layer-median")
        plt.title(f"Round {rnd}: client distances")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.grid()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"dists_round_{rnd:02d}.png")
        plt.close()

    # aggregate remaining clients (layer median)
    keep = np.ones(len(weights), bool); keep[worst_idx] = False
    new_layers = [
        np.median(np.stack(layer, axis=0)[keep], axis=0) for layer in zip(*weights)
    ]
    return fl.common.ndarrays_to_parameters(new_layers), ignored_hex

def weighted_median_aggregate(
        results, rnd,
        *, gamma=8.0, low_frac=0.8,
        eps=1e-12, return_debug=False):

    # ---------------------------------------------------- unpack parameters
    w_nd   = [fl.common.parameters_to_ndarrays(r.parameters) for _, r in results]
    hexs   = [p.cid for p, _ in results]         # client hex‑IDs in weight order
    n      = len(hexs)

    client_w   = np.zeros(n)
    out_layers = []

    # ---------------------------------------------------- layer‑wise pass
    for layer in zip(*w_nd):
        stacked = np.stack(layer, axis=0)
        med     = np.median(stacked, axis=0)

        dists  = np.linalg.norm((stacked - med).reshape(n, -1), axis=1)
        d_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-12)

        raw_w  = np.exp(-gamma * d_norm)
        raw_w  = raw_w if raw_w.sum() > eps else np.ones_like(raw_w)
        raw_w /= raw_w.sum()

        client_w += raw_w
        out_layers.append(np.average(stacked, axis=0, weights=raw_w))

    # ---------------------------------------------------- final weights
    client_w /= client_w.sum()

    fair      = 1.0 / n
    threshold = low_frac * fair
    ignored   = [h for h, w in zip(hexs, client_w) if w < threshold]

    # --------------- DEBUG: check the numbers in the console --------------
    print(f"[DBG] threshold = {threshold:.4f}")
    print(f"[DBG] weights   = " +
          ", ".join(f"{w:.3f}" for w in client_w))          
    print(f"[DBG] ignored   = {len(ignored)} clients")

    print(f"[DBG rnd {rnd}] " +
          ", ".join(f"{h[:6]}:{w:.3f}" for h, w in zip(hexs, client_w)))
    # ---------------------------------------------------------------------

    if return_debug:
        return (fl.common.ndarrays_to_parameters(out_layers),
                ignored, client_w, threshold, hexs)
    return fl.common.ndarrays_to_parameters(out_layers), ignored


# -----------------------------------------------------------------------------#
#                                Strategies                                    #
# -----------------------------------------------------------------------------#
class MedianAggregationStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit          = SAMPLE_SIZE / TOTAL_CLIENTS,
            min_fit_clients       = SAMPLE_SIZE,
            min_available_clients = TOTAL_CLIENTS,
            fraction_evaluate     = SAMPLE_SIZE / TOTAL_CLIENTS,
            min_evaluate_clients      = SAMPLE_SIZE,
        )
        self.cid_map : dict[str,str] = {}

    def aggregate_fit(self, rnd, results, failures):                        
        if not results: return None, {}
        for p,r in results:
            if (cid:=r.metrics.get("client_id")) is not None:
                self.cid_map[p.cid] = f"Client {cid}"

        new_params, ignored_hex = median_trim_aggregate(
            results, rnd, trim_frac=0.30,       # fraction of clients to ignore
            cid_map=self.cid_map, plot=True
        )

        ignored_readable = [self.cid_map.get(h,h[:6]) for h in ignored_hex]
        print(f"[Median] ignored: {ignored_readable}")
        ignored_client_log.append(ignored_readable)
        ignored_client_hex_log.append(ignored_hex)
        return new_params, {}

    def aggregate_evaluate(self, rnd, results, failures):               
        if not results: return None, {}
        loss = np.mean([r.loss                       for _,r in results])
        acc  = np.mean([r.metrics["accuracy"]        for _,r in results])
        round_metrics.append((rnd,loss,acc))
        return float(loss), {"accuracy": float(acc)}


class WeightedMedianStrategy(fl.server.strategy.FedAvg):            
    def __init__(self):
        super().__init__(
            fraction_fit          = SAMPLE_SIZE / TOTAL_CLIENTS,                    
            min_fit_clients       = SAMPLE_SIZE,                        
            min_available_clients = TOTAL_CLIENTS,
            fraction_evaluate     = SAMPLE_SIZE / TOTAL_CLIENTS,
            min_evaluate_clients      = SAMPLE_SIZE,
        )
        self.cid_map : dict[str,str] = {}

    def aggregate_fit(self, rnd, results, failures):                
        if not results: return None, {}

        for p,r in results:
            if (cid:=r.metrics.get("client_id")) is not None:
                self.cid_map[p.cid] = f"Client {cid}"

        #new_params, ignored_hex = weighted_median_aggregate(results)
        new_params, ignored_hex, w, thr, hexs = weighted_median_aggregate(
                results, rnd, gamma=8.0, low_frac=0.8, return_debug=True)

        #plot_distance_profile(rnd, w, thr, ignored_hex, self.cid_map, block=True)
        plot_weight_profile(rnd, w, thr, ignored_hex, self.cid_map, hexs=hexs, block=True)

        ignored_readable = [self.cid_map.get(h,h[:6]) for h in ignored_hex]
        print(f"[WeightedMedian] very-low-weight: {ignored_readable}")
        ignored_client_log.append(ignored_readable)
        ignored_client_hex_log.append(ignored_hex)
        return new_params, {}


    def aggregate_evaluate(self, rnd, results, failures):
        if not results: return None, {}
        loss = np.mean([r.loss                       for _,r in results])
        acc  = np.mean([r.metrics["accuracy"]        for _,r in results])
        round_metrics.append((rnd,loss,acc))
        return float(loss), {"accuracy": float(acc)}


class FedAvgWithEval(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit          = SAMPLE_SIZE / TOTAL_CLIENTS,
            min_fit_clients       = SAMPLE_SIZE,
            min_available_clients = TOTAL_CLIENTS,
            fraction_evaluate     = SAMPLE_SIZE / TOTAL_CLIENTS,
            min_evaluate_clients      = SAMPLE_SIZE,
        )
        self.cid_map : dict[str,str] = {}

    def aggregate_evaluate(self, rnd, results, failures):
        if not results: return None, {}
        loss = np.mean([r.loss                       for _,r in results])
        acc  = np.mean([r.metrics["accuracy"]        for _,r in results])
        round_metrics.append((rnd,loss,acc))
        return float(loss), {"accuracy": float(acc)}


# -----------------------------------------------------------------------------#
#                                   Main                                       #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["fedavg","median","weighted"],
                        required=True)
    args = parser.parse_args()

    if args.strategy == "median":
        print("[Server] Using Median Aggregation Strategy")
        strategy = MedianAggregationStrategy()
    elif args.strategy == "weighted":
        print("[Server] Using Weighted Median Aggregation Strategy")
        strategy = WeightedMedianStrategy()
    else:
        print("[Server] Using FedAvg (no defence)")
        strategy = FedAvgWithEval()

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(NUM_ROUNDS),
    )

    # ---------- CSV report ----------------------------------
    attacker_ids = set()
    try:
        with open("attacker_ids.txt") as f:
            attacker_ids = set(f.read().strip().split(","))
    except FileNotFoundError:
        pass

    ts   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag  = {"median":"Median_flipping",
            "weighted":"Weighted_flipping",
            "fedavg":"FedAvg_flipping"}[args.strategy]
    path = f"{tag}_{ts}.csv"

    hex_to_num = {h:int(lbl.split()[-1]) for h,lbl in strategy.cid_map.items()}
    rows = []

    for i,(rnd,loss,acc) in enumerate(round_metrics):
        if args.strategy == "fedavg":
            rows.append([rnd,loss,acc,0,0,0])
        else:
            ignored_hex = ignored_client_hex_log[i]
            att = [h for h in ignored_hex
                     if str(hex_to_num.get(h,"-1")) in attacker_ids]
            rows.append([rnd,loss,acc,
                         len(ignored_hex), len(att), len(ignored_hex)-len(att)])

    df = pd.DataFrame(rows, columns=[
        "round","loss","accuracy",
        "ignored_total","ignored_attackers","ignored_benign"
    ])
    df.to_csv(path,index=False)
    print(f" CSV report saved as {path}")
