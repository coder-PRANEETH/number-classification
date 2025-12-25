import numpy as np


def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4),"big")

        assert magic == 2051, "Invalid IDX3 image file"

        data = f.read(num_images * rows * cols)

        images = np.frombuffer(data, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

        return images
def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        assert magic == 2049, "Invalid IDX1 label file"

        labels = np.frombuffer(f.read(num_labels), dtype=np.uint8)

        return labels
X_train = load_mnist_images("train-images.idx3-ubyte")
y_train = load_mnist_labels("train-labels.idx1-ubyte")

X_test  = load_mnist_images("t10k-images.idx3-ubyte")
y_test  = load_mnist_labels("t10k-labels.idx1-ubyte")




X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0


image = X_test[61,...]

label = y_test[61,...]

data = np.load("weights.npz")
W1 = data["W1"]


W2 = data["W2"]
B1 = data["B1"]
B2 = data["B2"]
filter1 = data["filter1"]
filter2 = data["filter2"]

out1 = np.zeros((26,26))
out2 = np.zeros((24,24))
pooled = np.zeros((12,12))

for i in range(26):
            for j in range(26):
                out1[i,j] = np.sum(image[i:i+3, j:j+3] * filter1)

A1 = np.maximum(0, out1)


for i in range(24):
    for j in range(24):
        out2[i,j] = np.sum(A1[i:i+3, j:j+3] * filter2)


for i in range(0,24,2):
    for j in range(0,24,2):
        pooled[i//2, j//2] = np.max(out2[i:i+2, j:j+2])

neu_input = pooled.flatten()
neu_input = neu_input.reshape(1,144)




Z1 = neu_input@W1+B1

AN1 = np.maximum(0,Z1)

Z2 = AN1@W2+B2


Z2_stable = Z2 - np.max(Z2, axis=1, keepdims=True)
exp_Z = np.exp(Z2_stable)
y_pred = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)



print((y_pred[0]))



print("tre :",label)


