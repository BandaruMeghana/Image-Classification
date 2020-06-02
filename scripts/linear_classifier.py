import numpy as np
import cv2
from config import linear_classifier


labels = ["dog","cat","panda"]
# np.random.seed()

"""
Initialize the weight and bias matrices
- W dimension will be K*N where
    - K is the number of classes, 3 in this case
    - N is the 1D size of the image. Here the images are of size (32*32*3) = 3072
- b dimension will be K
"""
W = np.random.rand(linear_classifier["K"],linear_classifier["N"])
b = np.random.rand(linear_classifier["K"])

#These W and b needs to be updated via optimization algorithms. For now, lets keep them constant.
original_image = cv2.imread(linear_classifier['data_path'])
#Reshape the image
image = cv2.resize(original_image, (32,32)).flatten()

# get the score
scores = W.dot(image) + b
# print(scores)
for(label, score) in zip(labels,scores):
    print("[INFO]{}: {:.2f}".format(label,score))


cv2.putText(original_image, "Label: {}".format(labels[np.argmax(scores)]), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Image", original_image)
cv2.waitKey(0)