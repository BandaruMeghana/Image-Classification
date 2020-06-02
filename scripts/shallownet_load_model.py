from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from modules.dataset_loader import DataSetLoader
from modules.pre_processor import Processor
from modules.image_to_array_preprocessor import ImageToArrayPreProcessor
import numpy as np
import glob
from config import shallownet_animals
import cv2


class_labels = ["cat", "dog", "panda"]
print("[INFO] Sampling the test images...")
image_paths = np.array(list(path for path in glob.glob(shallownet_animals['dataset_path'])))
# randomly select 10 images for testing purpose
indexes = np.random.randint(0, len(image_paths), size=(10,))
print(indexes)
print(indexes.dtype)
image_paths = image_paths[indexes]

print("[INFO] Loading the test images...")
processor = Processor(32,32)
img2Array = ImageToArrayPreProcessor()
data_loader = DataSetLoader(preprocessors=[processor, img2Array])
(data, labels) = data_loader.load_data(image_paths, verbose=500)
data = data.astype("float")/255.0

print("[INFO] Loading the saved model")
model = load_model(shallownet_animals["save_model_path"])

print("[INFO] Making predictions...")
preds = model.predict(data, batch_size= shallownet_animals["batch_size"]).argmax(axis=1)

# display the predictions
for (i, image_path) in enumerate(image_paths):
    image = cv2.imread(image_path)
    cv2.putText(image, "Label: {}".format(class_labels[preds[i]]), (10,30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)