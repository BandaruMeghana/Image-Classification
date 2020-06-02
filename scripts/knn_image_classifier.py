from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from modules.dataset_loader import DataSetLoader
from modules.pre_processor import Processor
import glob
from config import knn_classifier

image_paths = []
for image_path in glob.glob(knn_classifier['dataset']):
    image_paths.append(image_path)

print("[INFO] Loading images...")
processor = Processor(32,32)    #Resize the images to 32*32
data_loader = DataSetLoader([processor])
(data,labels) = data_loader.load_data(image_paths,verbose=500)
print(data.shape)
data = data.reshape(data.shape[0],3072)     #To flatten the numpy array from (3000*32*32*3) to (3000*3072)
print(data.shape)
print("[INFO] feature matrix: {:.1f}MB".format(data.nbytes/(1024*1000.0)))
print(labels)
le = LabelEncoder()
labels = le.fit_transform(labels)
# print(labels)

#train-test split
(trainX, testX, trainY, testY) = train_test_split(data,labels,test_size=0.25, random_state=42)

print("[INFO] Training K-NN classifier")
model = KNeighborsClassifier(n_neighbors=knn_classifier['k'], n_jobs=knn_classifier['jobs'])
model.fit(trainX,trainY)
print("[INFO] Evaluating K-NN classifier")
print(classification_report(testY, model.predict(testX),target_names=le.classes_))