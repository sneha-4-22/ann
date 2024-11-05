import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

epochs = range(1, 51)
training_losses = [86.8894, 11.6427, 8.2430, 7.3443, 6.9879, 5.0984, 3.6775, 3.0974, 2.8555, 3.1799,
                  6.7805, 5.8103, 4.1192, 4.4003, 4.9199, 5.8706, 5.5937, 5.5388, 5.5046, 5.4764,
                  5.4778, 5.4379, 5.4776, 5.4759, 5.4980, 5.4307, 5.4014, 5.4677, 5.4507, 5.4815,
                  5.4317, 5.4631, 5.4523, 5.4816, 5.4442, 5.4806, 5.5076, 5.3970, 5.4228, 5.4161,
                  5.4791, 5.4898, 5.4989, 5.5064, 5.4636, 5.4861, 5.4506, 5.4376, 5.4413, 5.4105]

validation_losses = [15.0307, 8.4935, 8.0484, 8.9842, 5.9528, 3.1775, 3.3234, 2.6571, 2.6307, 5.6442,
                    5.8101, 4.2874, 3.9225, 4.8535, 5.4923, 5.9268, 5.4530, 5.5516, 5.4988, 5.5163,
                    5.4391, 5.4554, 5.4669, 5.4157, 5.4537, 5.4200, 5.4603, 5.4594, 5.4616, 5.4828,
                    5.4302, 5.4410, 5.4151, 5.4506, 5.4179, 5.4601, 5.4818, 5.4682, 5.4481, 5.4697,
                    5.4603, 5.4464, 5.4758, 5.4001, 5.4867, 5.4526, 5.4237, 5.4421, 5.4351, 5.4398]

sns.set()

sns.set_palette("husl")

plt.figure(figsize=(20, 15))

plt.subplot(2, 2, 1)
plt.plot(epochs, training_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs, validation_losses, 'r-', label='Validation Loss', linewidth=2)
plt.title('Training and Validation Loss Over Time', pad=20, size=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
loss_diff = np.array(training_losses) - np.array(validation_losses)
plt.plot(epochs, loss_diff, 'g-', linewidth=2)
plt.title('Training-Validation Loss Difference', pad=20, size=14)
plt.xlabel('Epoch')
plt.ylabel('Loss Difference')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')

plt.subplot(2, 2, 3)
plt.plot(epochs[:15], training_losses[:15], 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs[:15], validation_losses[:15], 'r-', label='Validation Loss', linewidth=2)
plt.title('Early Training Phase (First 15 Epochs)', pad=20, size=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.hist(training_losses, bins=30, alpha=0.5, label='Training Loss', color='blue')
plt.hist(validation_losses, bins=30, alpha=0.5, label='Validation Loss', color='red')
plt.title('Loss Distribution', pad=20, size=14)
plt.xlabel('Loss Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

plt.tight_layout(pad=3.0)
plt.show()

print("\n=== Model Training Analysis Report ===")
print("\nBest Performance Metrics:")
best_train_loss = min(training_losses)
best_val_loss = min(validation_losses)
best_train_epoch = training_losses.index(best_train_loss) + 1
best_val_epoch = validation_losses.index(best_val_loss) + 1

print(f"Best Training Loss: {best_train_loss:.4f} (Epoch {best_train_epoch})")
print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_val_epoch})")

print("\nTraining Stability Analysis:")
final_avg_train = np.mean(training_losses[-10:])
final_avg_val = np.mean(validation_losses[-10:])
final_std_train = np.std(training_losses[-10:])
final_std_val = np.std(validation_losses[-10:])

print(f"Final 10 Epochs - Average Training Loss: {final_avg_train:.4f} (±{final_std_train:.4f})")
print(f"Final 10 Epochs - Average Validation Loss: {final_avg_val:.4f} (±{final_std_val:.4f})")

avg_diff = np.mean(np.abs(loss_diff[-10:]))
print(f"Average Train-Val Difference (last 10 epochs): {avg_diff:.4f}")

print("\nTraining Convergence Analysis:")
if avg_diff < 0.5:
    print("Model shows good convergence with minimal overfitting")
elif avg_diff < 1.0:
    print("Model shows moderate convergence with some overfitting")
else:
    print("Model shows signs of significant overfitting")

print("\nRecommendations:")
if avg_diff > 0.5:
    print("- Consider implementing early stopping")
    print("- Try increasing dropout or regularization")
if final_std_train > 0.5:
    print("- Learning rate might be too high, consider reducing it")
if best_val_epoch < 10:
    print("- Model might benefit from slower learning rate decay")