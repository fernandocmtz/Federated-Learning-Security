import pandas as pd
import matplotlib.pyplot as plt

# Load CSV with ignored client stats
df = pd.read_csv("ignored_summary.csv")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df["Round"], df["Ignored_Total"], marker="o", label="Total Ignored")
plt.plot(df["Round"], df["Ignored_Attackers"], marker="x", linestyle="--", label="Attackers Ignored")
plt.plot(df["Round"], df["Ignored_Benigns"], marker="s", linestyle=":", label="Benigns Ignored")

plt.title("Ignored Clients per Round (Median Aggregation)")
plt.xlabel("Round")
plt.ylabel("Number of Clients")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ignored_clients_plot.png")
plt.show()
