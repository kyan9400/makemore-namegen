import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Check if the loss curve image exists
loss_file = "loss_curve.png"

if os.path.exists(loss_file):
    print(f"📈 Displaying {loss_file}...")
    img = mpimg.imread(loss_file)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Training & Validation Loss Curve")
    plt.show()
else:
    print("❌ No loss_curve.png found. Please run train.py first.")
