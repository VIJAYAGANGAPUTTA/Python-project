import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ann as an
import preprocess as dp

if __name__ == "__main__":
    images_folder_path = "C:\\Users\\P.VIJAYA GANGA\\Downloads\\dl dataset\\train"
    imdata = dp.PreProcess_Data()
    imdata.visualization_images(images_folder_path, 2)
    train, label, image_df = imdata.preprocess(images_folder_path)
    print(label)
    train_generator, test_generator, validate_generator = imdata.generate_train_test_images(train, label)
    AnnModel = an.DeepANN()
    Model1 = AnnModel.simple_model()
    print("train generator", train_generator)
    ANN_history = Model1.fit(train_generator, epochs=2, validation_data=validate_generator)
    test_loss, test_acc = Model1.evaluate(test_generator)

    Model1.save("my_model3.keras")
    print(f'Test accuracy:{test_acc}')
    print("ann architecture")
    print(Model1.summary())
    print("plot graph")
    imdata.plot_history(ANN_history)
