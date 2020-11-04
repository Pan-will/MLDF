import numpy as np

data = np.load("image_data.npy")
print(data)
print(len(data[0]))

print("$"*44)

label = np.load("image_label.npy")
print(label)
print(len(label[0]))


