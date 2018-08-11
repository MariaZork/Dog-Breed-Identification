"""
    Created on 22.05.2018  
    
    author: Maria Zorkaltseva
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Flatten, Activation, Dropout


class CNN:
    def __init__(self, num_classes, width=32, height=32, depth=3, batch_size=32,
                 steps_per_epoch=1, num_epochs=12, activation='relu',
                 optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']):

        self.model = Sequential()
        self.num_classes = num_classes
        self.image_shape = (width, height, depth)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def build_my_cnn(self):
        # 1st Convolutional Layer
        self.model.add(Conv2D(16, (3, 3), data_format="channels_last", input_shape=self.image_shape))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(0.5))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        # 2nd Convolutional Layer
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(0.5))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        # 3rd Convolutional Layer
        self.model.add(Conv2D(48, (3, 3)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(0.7))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        # 4th Convolutional Layer
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(0.7))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        # flatten
        self.model.add(Flatten())

        # 1st dense layer
        self.model.add(Dense(1024, activation=self.activation))

        # 2nd dense layer
        self.model.add(Dense(512, activation=self.activation))

        # 3rd dense layer
        self.model.add(Dense(256, activation=self.activation))

        # softmax output layer
        self.model.add(Dense(120, activation='softmax'))

        self.model.summary()

    def build_lenet(self):
        # 1st Convolutional Layer
        self.model.add(Conv2D(6, (5, 5), strides=1, data_format="channels_last",
                              activation=self.activation, input_shape=self.image_shape))

        self.model.add(AveragePooling2D(pool_size=2, strides=2, data_format="channels_last"))

        # 2nd Convolutional Layer
        self.model.add(Conv2D(16, (5, 5), strides=1, data_format="channels_last",
                              activation=self.activation))

        self.model.add(AveragePooling2D(pool_size=2, strides=2, data_format="channels_last"))

        # dense layers
        self.model.add(Flatten())
        # 1st dense layer
        self.model.add(Dense(120, activation=self.activation))
        # 2nd dense layer
        self.model.add(Dense(84, activation=self.activation))
        # softmax output layer
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.summary()

    def build_alexnet(self):
        # 1st Convolutional Layer
        self.model.add(Conv2D(96, (11, 11), strides=(4, 4), data_format="channels_last",
                              activation=self.activation, input_shape=self.image_shape))

        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_last"))

        self.model.add(BatchNormalization())

        # 2nd Convolutional Layer
        self.model.add(Conv2D(256, (5, 5), padding='same', data_format="channels_last", activation=self.activation))

        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_last"))

        self.model.add(BatchNormalization())

        # 3rd Convolutional Layer
        self.model.add(Conv2D(384, (3, 3), padding='same', data_format="channels_last", activation=self.activation))

        self.model.add(BatchNormalization())

        # 4th Convolutional Layer
        self.model.add(Conv2D(384, (3, 3), padding='same', data_format="channels_last", activation=self.activation))

        self.model.add(BatchNormalization())

        # 5th Convolutional Layer
        self.model.add(Conv2D(256, (3, 3), padding='same', data_format="channels_last", activation=self.activation))

        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(BatchNormalization())

        # dense layers
        self.model.add(Flatten())
        # 1st dense layer
        self.model.add(Dense(9216, activation=self.activation))
        self.model.add(Dropout(0.4))
        self.model.add(BatchNormalization())

        # 2nd dense layer
        self.model.add(Dense(4096, activation=self.activation))
        self.model.add(Dropout(0.4))
        self.model.add(BatchNormalization())

        # 3rd dense layer
        self.model.add(Dense(4096, activation=self.activation))
        self.model.add(Dropout(0.4))
        self.model.add(BatchNormalization())

        # softmax output layer
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.summary()

    # def build_alexnet(self):
    #     self.model.add(Conv2D(96, (11, 11), strides=(4, 4), data_format="channels_last",
    #                           activation=self.activation, input_shape=self.image_shape))
    #
    #     self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_last"))
    #
    #     self.model.add(Conv2D(256, (5, 5), strides=(2, 2), data_format="channels_last", activation=self.activation))
    #
    #     self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_last"))
    #
    #     self.model.add(Conv2D(384, (3, 3), strides=(2, 2), data_format="channels_last", activation=self.activation))
    #
    #     self.model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_last"))
    #
    #     self.model.add(Conv2D(256, (3, 3), data_format="channels_last", activation=self.activation))
    #
    #     self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_last"))
    #
    #     self.model.add(Flatten())
    #     self.model.add(Dense(4096, activation=self.activation))
    #     self.model.add(Dense(4096, activation=self.activation))
    #     self.model.add(Dense(self.num_classes, activation='softmax'))

    def compile_net(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def fit_net(self, x, y):
        self.model.fit(x=x, y=y, steps_per_epoch=self.steps_per_epoch, epochs=self.num_epochs)

    def fit_net_generator(self, data_generator):
        self.model.fit_generator(generator=data_generator,
                                 steps_per_epoch=self.steps_per_epoch, epochs=self.num_epochs)

    def save_net_weights(self, output_dir):
        self.model.save_weights(filepath=output_dir)

    def save_model(self, output_dir):
        print("writing model to json")
        model_json = self.model.to_json()
        with open(output_dir, 'w') as json_file:
            json_file.write(model_json)
