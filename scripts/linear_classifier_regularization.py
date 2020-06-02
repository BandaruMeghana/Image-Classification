from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from config import regularization
import glob
from modules.pre_processor import Processor
from modules.dataset_loader import DataSetLoader
from sklearn.preprocessing import LabelEncoder

print("[INFO] Loading images")
image_paths = []
for path in glob.glob(regularization['dataset']):
    image_paths.append(path)

processor = Processor(32,32)
data_loader = DataSetLoader([processor])
(data,labels) = data_loader.load_data(image_paths,verbose=500)
data =  data.reshape((data.shape[0], 3072))
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

for reg_method in (None, 'l1', 'l2'):
    # Train a SGD classifier using a softmax loss function the specified regularization function
    print('[INFO]: Training model with {} penalty'.format(str(reg_method).upper()))
    model = SGDClassifier(loss='log', penalty=reg_method, max_iter=10,
                          learning_rate="constant", eta0=regularization['alpha'], random_state=42)
    model.fit(trainX, trainY)

    # Evaluate the classifier
    accuracy = model.score(testX, testY)
    print('[INFO]: {} penalty accuracy: {:.2f}%'.format(str(reg_method).upper(), accuracy*100))