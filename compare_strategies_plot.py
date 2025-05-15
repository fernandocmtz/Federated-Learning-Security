import pandas as pd
import matplotlib.pyplot as plt

strategies = {
    "FedAvg": "ignored_summary_fedavg.csv",
    "Median": "ignored_summary_median.csv",
    "Weighted": "ignored_summary_weighted.csv",
}

# Helper function to plot any column
def plot_column(column_name, title, ylabel, output_file):
    plt.figure(figsize=(10, 6))
    for name, filename in strategies.items():
        try:
            df = pd.read_csv(filename)
            plt.plot(df["Round"], df[column_name], marker="o", label=name)
        except FileNotFoundError:
            print(f" File not found: {filename}")
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Plot Attackers Ignored
plot_column(
    column_name="Ignored_Attackers",
    title="Attackers Ignored per Round - Strategy Comparison",
    ylabel="Attackers Ignored",
    output_file="plot_ignored_attackers.png"
)

# Plot Benigns Ignored
plot_column(
    column_name="Ignored_Benigns",
    title="Benign Clients Ignored per Round - Strategy Comparison",
    ylabel="Benign Clients Ignored",
    output_file="plot_ignored_benigns.png"
)

# Plot Total Ignored
plot_column(
    column_name="Ignored_Total",
    title="Total Clients Ignored per Round - Strategy Comparison",
    ylabel="Total Clients Ignored",
    output_file="plot_ignored_total.png"
)
