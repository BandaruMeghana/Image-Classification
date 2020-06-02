from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArrayPreProcessor:
    def __init__(self, data_format=None):
        """
        :param data_format: None indicates to use the default value from the keras.json config file.
                            Explicitly we can pass other values like, "channels_first", "channels_last"
        :return:
        """
        self.data_format = data_format

    def preprocess(self, image):
        """
        :purpose: Orders the image in the specified data format.
        :return:  numpy with channels ordered properly.
        """
        print("self", vars(self))
        return img_to_array(image, data_format=self.data_format)
