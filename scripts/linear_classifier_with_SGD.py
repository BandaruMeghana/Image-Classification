from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from config import linear_classifier_with_SGD

def sigmoid_activation(x):
    return 1.0/(1 + np.exp(-x))


def predict(x,W):
    preds = sigmoid_activation(x.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds

def next_batch(X, y, bacth_size):
    """
    :param X: training dataset of feature vectors/raw image pixel intensities.
    :param y: The class labels associated with each of the training data points.
    :param bacth_size: The size of each mini-batch that will be returned.
    :return:
    """

    for i in np.arange(0,X.shape[0], bacth_size):
        yield (X[i:1+bacth_size], y[i:1+bacth_size])


'''
Get the dataset with 2 features(2D) and 1000 data points. Lables are either 0 or 1
i.e., cols = x1, x2, y
'''
(X,y) = make_blobs(n_samples=1000, n_features=2,centers=2,cluster_std=1.5, random_state=1)
print(y.shape)
y = y.reshape((y.shape[0],1))
print(y.shape)

# Incorporate the bias trick into X, to avoid tracking of b separately.
print(X.shape)
X = np.c_[X, np.ones((X.shape[0]))]
print(X.shape)

# train-test split
(trainX, testX, trainY, testY) = train_test_split(X,y, test_size=0.5, random_state=42)
print("[INFO] training")
#Initialize random weights
W = np.random.rand(X.shape[1],1)
losses = []

for epoch in np.arange(0, linear_classifier_with_SGD['epochs']):
    epochLoss = []
    for (batchX, batchY) in next_batch(X,y,linear_classifier_with_SGD["batch_size"]):
        preds = sigmoid_activation(batchX.dot(W))
        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))

        gradient = batchX.T.dot(error)
        W += -linear_classifier_with_SGD['alpha'] * gradient

    loss = np.average(epochLoss)
    losses.append(loss)

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
plt.plot(np.arange(0, linear_classifier_with_SGD["epochs"]), losses)
plt.xlabel(' Epoch #')
plt.ylabel('Loss')
plt.show()




