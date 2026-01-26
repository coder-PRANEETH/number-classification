import numpy as np
import matplotlib.pyplot as plt

                                                              
def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), "big")

        assert magic == 2051

        data = f.read(num_images * rows * cols)
        images = np.frombuffer(data, dtype=np.uint8)
        return images.reshape(num_images, rows, cols)


def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        assert magic == 2049
        return np.frombuffer(f.read(num_labels), dtype=np.uint8)


X_test = load_mnist_images("t10k-images.idx3-ubyte").astype("float32") / 255.0
y_test = load_mnist_labels("t10k-labels.idx1-ubyte")


                                                              
data = np.load("weights.npz")
W1, B1 = data["W1"], data["B1"]
W2, B2 = data["W2"], data["B2"]
filter1, filter2 = data["filter1"], data["filter2"]

print(filter1,filter2)

                                                              
num_samples = 500
num_classes = 10

y_true = []
y_pred = []

for h in range(num_samples):

    image = X_test[h]
    label = y_test[h]

                                                                  
    out1 = np.zeros((26, 26))
    for i in range(26):
        for j in range(26):
            out1[i, j] = np.sum(image[i:i+3, j:j+3] * filter1)

    A1 = np.maximum(0, out1)

                                                                  
    out2 = np.zeros((24, 24))
    for i in range(24):
        for j in range(24):
            out2[i, j] = np.sum(A1[i:i+3, j:j+3] * filter2)

                                                                  
    pooled = np.zeros((12, 12))
    for i in range(0, 24, 2):
        for j in range(0, 24, 2):
            pooled[i//2, j//2] = np.max(out2[i:i+2, j:j+2])

                                                                  
    x = pooled.flatten().reshape(1, 144)

    Z1 = x @ W1 + B1
    A2 = np.maximum(0, Z1)

    Z2 = A2 @ W2 + B2
    Z2 -= np.max(Z2, axis=1, keepdims=True)

    probs = np.exp(Z2)
    probs /= np.sum(probs, axis=1, keepdims=True)

    pred = np.argmax(probs)

    y_true.append(label)
    y_pred.append(pred)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


                                                              
accuracy = np.mean(y_true == y_pred)


                                                              
precision_per_class = []

for c in range(num_classes):
    tp = np.sum((y_pred == c) & (y_true == c))
    fp = np.sum((y_pred == c) & (y_true != c))

    if tp + fp == 0:
        precision_per_class.append(0.0)
    else:
        precision_per_class.append(tp / (tp + fp))

precision = np.mean(precision_per_class)

plt.bar(range(num_classes), precision_per_class)
plt.xlabel("Class")
plt.ylabel("Precision")
plt.title("Precision per Class")
plt.show()

print(f"Accuracy on {num_samples} samples : {accuracy:.4f}")
print(f"Precision (macro)               : {precision:.4f}")
