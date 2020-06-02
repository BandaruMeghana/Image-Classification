from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from modules.image_to_array_preprocessor import ImageToArrayPreProcessor
from modules.pre_processor import Processor
from modules.dataset_loader import DataSetLoader
from networks.CONV.shallow_net import ShallowNet
from tensorflow.keras.optimizers import SGD
import glob
import numpy as np
from config import shallownet_animals
import matplotlib.pyplot as plt


image_paths = list(path for path in glob.glob(shallownet_animals['dataset_path']))
processor = Processor(32,32)
print("processor: ", vars(processor))
img2array_processor = ImageToArrayPreProcessor()
print("img2array: ", vars(img2array_processor))
print("[INFO] Loading the images...")
"""
Applying 2 preprocessing steps
1. Resize the images to 32*32
2. Handle the channel ordering
"""
data_loader = DataSetLoader(preprocessors=[processor, img2array_processor])
(data, labels) = data_loader.load_data(image_paths, verbose=500)
# scale the images to the range [0,1]
data = data.astype("float")/255.0

# train-test split the data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

# convert the labels from ints to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print("[INFO] Compiling the model...")
opt = SGD(shallownet_animals["alpha"])
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss=shallownet_animals["loss"], optimizer=opt, metrics=["accuracy"])

print("[INFO] Training the network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=shallownet_animals["batch_size"], epochs=shallownet_animals["epochs"], verbose=1)

print("[INFO] Evaluating the network...")
preds = model.predict(testX, batch_size=shallownet_animals["batch_size"])
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=["cat", "dog", "panda"]))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="training loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0,100), H.history["accuracy"], label="training accuracy")
plt.plot(np.arange(0,100), H.history["val_accuracy"], label="validation accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()



