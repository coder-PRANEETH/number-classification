import cv2
import numpy as np

                                           
                                  
                                           
CANVAS_SIZE = 280
FINAL_SIZE = 28
BRUSH_RADIUS = 8

canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
drawing = False
arr = None                                    

                                           
                                  
                                           
def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), BRUSH_RADIUS, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Draw Here")
cv2.setMouseCallback("Draw Here", draw)

print("Press 's' to predict | 'c' to clear | 'q' to quit")

                                           
                                  
                                           
data = np.load("weights.npz")
W1, B1 = data["W1"], data["B1"]
W2, B2 = data["W2"], data["B2"]
filter1, filter2 = data["filter1"], data["filter2"]

                                           
                                  
                                           
while True:
    cv2.imshow("Draw Here", canvas)
    key = cv2.waitKey(1) & 0xFF

                                               
    if key == ord('s'):
        resized = cv2.resize(canvas, (FINAL_SIZE, FINAL_SIZE),
                              interpolation=cv2.INTER_AREA)

        arr = resized / 255.0
        image = arr

                                                   
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
        Z2 -= np.max(Z2)

        probs = np.exp(Z2)
        probs /= np.sum(probs)

        pred = np.argmax(probs)

        print(f"\nPredicted Digit: {pred}")
        
        cv2.imshow("28x28 Input", resized)

                                               
    elif key == ord('c'):
        canvas[:] = 0

                                               
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
