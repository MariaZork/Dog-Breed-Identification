"""
    Created on 21.05.2018

    author: Maria Zorkaltseva
"""

import pandas as pd
import numpy as np
from cnn_nets import CNN
from preprocessing import ImagesPreprocessing

if __name__ == '__main__':

    data_dir = "dataset\\"
    train_data_dir = data_dir + "train\\"
    test_data_dir = data_dir + "test\\"
    for_train = pd.read_csv(data_dir + "labels.csv")
    for_test = pd.read_csv(data_dir + "sample_submission.csv")

    img_rows, img_cols, depth = 150, 150, 3
    BS = 32

    IMG = ImagesPreprocessing(width=img_rows, height=img_cols, batch_size=32)

    # loading train images into numpy array
    train_image_paths = IMG.extract_img_paths(train_data_dir, for_train)
    image_labels = for_train["breed"]
    X_train = IMG.images_to_matrix(train_image_paths)

    # loading test images into numpy array
    test_image_paths = IMG.extract_img_paths(test_data_dir, for_test)
    X_test = IMG.images_to_matrix(test_image_paths)

    # divide pixels of the images by 255
    X_train = IMG.rescale(X_train)
    X_test = IMG.rescale(X_test)

    num_classes = len(np.unique(image_labels))
    labels = IMG.encode_labels(image_labels)

    train_generator = IMG.image_data_generator(X_train, labels)

    CNNModel = CNN(num_classes, width=img_rows, height=img_cols, depth=depth,
                   steps_per_epoch=len(X_train) / BS, num_epochs=20)
    CNNModel.build_alexnet()
    CNNModel.compile_net()
    CNNModel.save_model("results\\MyAlexNet_model_%d_%d" % (img_rows, img_cols))
    CNNModel.fit_net_generator(train_generator)

    print("saving weights to hdf5")
    CNNModel.save_net_weights("results\\weights_AlexNet_%d_%d.h5" % (img_rows, img_cols))

    predictions = CNNModel.model.predict(X_test, batch_size=BS, verbose=1)
    submission_res = pd.DataFrame(data=predictions, index=for_test["id"], columns=list(for_test.columns[1:]))
    submission_res.index.name = 'id'
    submission_res.to_csv('results\\submission.csv', encoding='utf-8', index=True)

