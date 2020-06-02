import os
import numpy as np
import cv2

class DataSetLoader:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load_data(self, image_paths, verbose =-1):
        '''4
        :param image_paths: in the format /path_to_dataset/class/{image}.jpg
        :param verbose: for degubbing purpose
        :return: an tuple of ([images],[labels)
        '''
        data = []
        labels = []

        for (index,image_path) in enumerate(image_paths):
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            #pre-process the images
            if self.preprocessors is not None:
                for process in self.preprocessors:
                    image = process.preprocess(image)
            data.append(image)
            labels.append(label)

            #print the INFO after loading 500 images
            if verbose>0 and index>0 and (index+1)%verbose == 0:
                print("[INFO] processed {}/{}".format(index+1,len(image_paths)))
        return (np.array(data), np.array(labels))