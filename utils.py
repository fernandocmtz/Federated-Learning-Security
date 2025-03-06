import matplotlib.pyplot as plt

def plot_results(accuracies, title="Accuracy Over Rounds"):
    plt.figure()
    plt.plot(accuracies, marker="o", linestyle="dashed", label="Accuracy")
    plt.xlabel("Training Round")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f"results/{title.replace(' ', '_')}.png")
    plt.show()
