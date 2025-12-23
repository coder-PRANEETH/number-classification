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

X_train = X_train[..., None]
X_test  = X_test[..., None]



lr =0.01
epochs =  5

eps =1e-12

labels = np.zeros(shape=(1,10))


filter1 = np.random.uniform(0,1,(3,3))
filter2 = np.random.uniform(0,1,(3,3))
W1 = np.random.randn(144,64) * 0.01
W2 = np.random.randn(64,10) * 0.01

B1 = np.zeros(shape=(1,64) )
B2 = np.zeros(shape=(1,10) )
out1 = np.zeros((26,26))
out2 = np.zeros((24,24))
pooled = np.zeros((12,12))

for e in range(epochs):
     totalloss=0
     for idx in range(len(X_train)):

        image = X_train[idx, :, :, 0]
        label = y_train[idx]
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




        label = y_train[0]
        labels[0][label]=1
        y_true= labels
        loss = -np.log(np.clip(y_pred[0, label], eps, 1 - eps))
        totalloss+=loss
        dZ2 = y_pred-y_true

        dW2 = AN1.T @ dZ2
        dB2 = dZ2

        dAN1 = dZ2 @ W2.T 
        dZ1 = dAN1*(Z1>0)
        dW1 = neu_input.T @ dZ1
        dB1 = dZ1

        dback = dZ1 @ W1.T 
        back=dback.reshape(12,12)

        W2 -= lr * dW2
        B2 -= lr * dB2
        W1 -= lr * dW1
        B1 -= lr * dB1

        dout2 = np.zeros(shape=(24,24))

        dout2 = np.zeros_like(out2)

        for i in range(12):
            for j in range(12):
                                                        
                window = out2[i*2:i*2+2, j*2:j*2+2]

                                                        
                max_idx = np.unravel_index(np.argmax(window), window.shape)

                                                        
                dout2[i*2 + max_idx[0], j*2 + max_idx[1]] = back[i, j]

                
        dfilter2 = np.zeros_like(filter2)    
        for u in range(3):
            for v in range(3):
                for i in range(24):
                    for j in range(24):
                        dfilter2[u, v] += dout2[i, j] * A1[i+u, j+v]


        dA1 = np.zeros_like(A1)                                                   

        for i in range(24):                                                       
            for j in range(24):                                                   
                for u in range(3):                                                    
                    for v in range(3):                                                
                        dA1[i+u, j+v] += dout2[i, j] * filter2[u, v]


        dout1 = dA1 * (out1 > 0)  


        dfilter1 = np.zeros_like(filter1)    

        for u in range(3):    
            for v in range(3):
                for i in range(26):                                                       
                    for j in range(26): 

                        dfilter1[u,v] += dout1[i,j] * image[i+u,j+v]

        filter1 -= lr * dfilter1
        filter2 -= lr * dfilter2
     print(totalloss)

print(filter2)