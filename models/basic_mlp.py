from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.metrics import Precision, Recall, AUC


class BasicMLP:

    def __init__(self, num_classes, vector_size, name="basic"):
        self.name = name
        self.num_classes = num_classes
        self.vector_size = vector_size
        self.input_type = 'vectors'

    def build(self):

        model = models.Sequential()
        model.add(layers.Dense(500, input_shape=(self.vector_size,), activation='relu'))
        model.add(layers.Dense(700, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1000, activation='relu'))
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(250, activation='relu'))
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['acc', Precision(name="prec"), Recall(name="rec"), AUC(name='auc')])

        return model