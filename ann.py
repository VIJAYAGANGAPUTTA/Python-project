from keras import Sequential

from keras.src.layers import Flatten, Dense, Conv2D, MaxPooling2D, Reshape, SimpleRNN, LSTM, BatchNormalization

class DeepANN():
    def simple_model(self):
        # add sequential,flatten,dense methods and activation functions also96+
        model=Sequential()
        #flatten is used to convert multi dimensional to single dimensional
        model.add(Flatten())
        model.add(Dense(128,activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        #before fitting the model we will configure with parameters
        # cstochastic gradient descent and accuracy
        model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
        return model