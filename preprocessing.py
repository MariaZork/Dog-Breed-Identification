"""
    Created on 21.05.2018  
    
    author: Maria Zorkaltseva
"""

from keras.preprocessing import image
import numpy as np

class ImagesPreprocessing:

    def __init__(self, width=32, height=32, batch_size=32):
        self.image_shape = (width, height)
        self.batch_size = batch_size

    def extract_img_paths(self, data_dir, data):
        """

        :param data_dir: directory name where dataset is located
        :type data_dir: str
        :param data: dataset
        :type data: Dataframe
        :return: list of paths of every image in dataset
        :rtype: list
        """
        from os.path import join, exists

        if exists(data_dir):
            try:
                image_paths = []
                for i in range(len(data)):
                    image_path = join(data_dir, data["id"].iloc[i] + ".jpg")
                    image_paths.append(image_path)
                return image_paths
            except NameError:
                print("There is no column with such name")
        else:
            print("There is no such directory")

    @staticmethod
    def image_load(image_path, image_shape):
        """
        For AlexNet image_shape = (227, 227)
        For LeNet image_shape = (32, 32)

        :param image_path: directory of the image
        :type image_path: str
        :param image_shape: input image shape (height, width)
        :type image_shape: tuple
        :return: image in numpy array
        :rtype: numpy array
        """

        try:
            img = image.load_img(image_path, target_size=image_shape)
            img = image.img_to_array(img)
            print('Loaded image %s' % image_path)
            return img
        except:
            print('Failed to open or resizing image %s' % image_path)

    def images_to_matrix(self, image_paths):
        """
        :param image_paths: list of the image paths
        :type image_paths: list
        :return: images into a numpy array of size
                 matrix_of_images = (num_samples, width, height, channels)
        """

        from os.path import isfile

        matrix_of_images = []

        for image_path in image_paths:
            if isfile(image_path):
                temp = self.image_load(image_path, self.image_shape)
                matrix_of_images.append(temp)
        return np.array(matrix_of_images)

    @staticmethod
    def to_flatten(x):
        """

        :param x: numpy array of size x = (num_samples, width, height, channels)
        :return: numpy array of size x = (num_samples, width*height*channels)
        """
        x_flatten = x.reshape(x.shape[0], -1)
        return x_flatten

    @staticmethod
    def rescale(x):
        """

        :param x: numpy array of size x = (num_samples, width, height, channels)
        :return: numpy array of size x_scaled = (num_samples, width, height, channels) divided by 225
        """
        x_scaled = x / 225
        return x_scaled

    @staticmethod
    def encode_labels(labels):
        """

        :param labels: numpy array
        :return: encoded and categorized (0, 0, 1, ..., 0, 0) numpy array
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        label_en = LabelEncoder()
        labels = label_en.fit_transform(labels)
        labels = labels.reshape(-1, 1)
        one_hot_en = OneHotEncoder()
        onehot_encoded_arr = one_hot_en.fit_transform(labels).toarray()
        return onehot_encoded_arr

    def image_data_generator(self, x, y):
        """

        :param x: numpy array of size (num_samples, width, height, channels)
        :param y: numpy array of size (num_samples, num_classes)
        :return: batches of the images through NumpyArrayIterator
        """
        if (len(x) and len(y)) != 0:
            data_gen = image.ImageDataGenerator(data_format="channels_last")
            data_generator = data_gen.flow(x, y, batch_size=self.batch_size)
        return data_generator
