from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K


class ShallowNet:
    """
    architecture: INPUT => CONV => RELU => FC
    """
    @staticmethod
    def build(width, height, depth, classes):

        model = Sequential()
        input_shape = (width, height, depth)

        # update the input_shape to (depth, width, height) if data_format='channel_first'
        if K.image_data_format == 'channel_first':
            input_shape = (depth, width, height)

        model.add(Conv2D(32, (3,3), padding='same')) # CONV layer will have 32 filters each of size 3*3
        model.add(Activation("relu"))
        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
