from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from config import linear_classifier_with_GD
import glob
from modules.pre_processor import Processor
from modules.dataset_loader import DataSetLoader
from sklearn.preprocessing import LabelEncoder

def sigmoid_activation(x):
    """
    :param x: The biased data point. i.e., Wx + b
    :return: sigmoid value of x
    """
    return 1.0/(1+np.exp(-x))

def predict(x, W):
    """
    :param x: data points with the bias included
    :param W: weight matrix
    :return: predictions
    """

    preds = sigmoid_activation(x.dot(W))
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    return preds

# Read the images
image_paths = []
for image_path in glob.glob(linear_classifier_with_GD['2_animal_dataset']):
    image_paths.append(image_path)

print("[INFO] Loading the images...")
processor = Processor(32,32)
data_loader = DataSetLoader([processor])
(data,labels) = data_loader.load_data(image_paths,verbose=500)
# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = labels.reshape((labels.shape[0],1))
# print(labels.shape)
data = data.reshape(data.shape[0],3072) # Flatten
# Concatenate the bias column to X(data)
data = np.c_[data, np.ones((data.shape[0]))]
# print(data.shape)

# train-test split the data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#Initialise the weights
W = np.random.rand(data.shape[1],1)
losses = []

print("[INFO] Training...")
for epoch in np.arange(0,linear_classifier_with_GD['epochs']):
    preds = sigmoid_activation(trainX.dot(W))
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)
    # print("trainX: ", trainX.shape)
    # print("error: ", error.shape)
    gradient = trainX.T.dot(error)
    # print("gradient: ", gradient)
    W += -linear_classifier_with_GD['alpha'] * gradient

    # Display the weight updates
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoc={}, loss={:.7f}".format(int(epoch + 1), loss))

print("[INFO] Testing...")
preds = predict(testX, W)
print(classification_report(testY, preds))
# Visualize the loss
plt.style.use('ggplot')
plt.figure()
plt.title('Training Loss')
plt.plot(np.arange(0, linear_classifier_with_GD["epochs"]), losses)
plt.xlabel(' Epoch #')
plt.ylabel('Loss')
plt.show()

