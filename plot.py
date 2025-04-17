import matplotlib.pyplot as plt
import numpy as np

#RNN
train_losses = [
    6.2729, 5.9329, 5.7666, 5.6608, 5.5782, 5.5121, 5.4591, 5.4131, 5.3720, 5.3367,
    5.3025, 5.2728, 5.2448, 5.2180, 5.1938, 5.1695, 5.1481, 5.1272, 5.1074, 5.0869,
    5.0683, 5.0495, 5.0326
]

val_losses = [
    6.0515, 5.8161, 5.7080, 5.6231, 5.5704, 5.5228, 5.4836, 5.4540, 5.4326, 5.4106,
    5.3914, 5.3732, 5.3631, 5.3528, 5.3386, 5.3251, 5.3159, 5.3076, 5.3003, 5.2885,
    5.2868, 5.2813, 5.2719
]








#Transformer
# train_losses = [
#     6.3182, 5.9781, 5.8546, 5.7414, 5.6679, 5.6123, 5.5664, 5.5252, 5.4852, 5.4476,
#     5.4129, 5.3777, 5.3438, 5.3125, 5.2799, 5.2476, 5.2186, 5.1865, 5.1587, 5.1248,
#     5.0935, 5.0638, 5.0325, 5.0016, 4.9733, 4.9399, 4.9122, 4.8829, 4.8556, 4.7820
# ]

# val_losses = [
#     6.0464, 5.9273, 5.8055, 5.7367, 5.6837, 5.6500, 5.6263, 5.5913, 5.5701, 5.5518,
#     5.5245, 5.5080, 5.4905, 5.4730, 5.4600, 5.4491, 5.4378, 5.4300, 5.4228, 5.4181,
#     5.4098, 5.4111, 5.4036, 5.4038, 5.4050, 5.4011, 5.4058, 5.4140, 5.4119, 5.4043
# ]

epochs = np.arange(1, len(val_losses) + 1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Training Loss", marker='o')
plt.plot(epochs, val_losses, label="Validation Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prog2_out/loss_RNN.png")
plt.show()
