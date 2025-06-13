import matplotlib.pyplot as plt

# Load the loss history
with open("loss.txt", "r") as f:
    losses = [float(line.strip()) for line in f.readlines()]

# Plot the loss
plt.figure(figsize=(10, 4))
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
