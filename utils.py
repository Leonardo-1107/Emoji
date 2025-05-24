import matplotlib.pyplot as plt

def plot_train(loss_record, epochs, save_name):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_record, marker='*', linestyle='-', color='b', label='Training Loss')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_name}.png')