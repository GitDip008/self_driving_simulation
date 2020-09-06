import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam



def get_name(filepath):
    return filepath.split("\\")[-1]

def import_data_info(path):
    columns = ["Center", "Left", "Right", "Steering", "Throttle", "Break", "Speed"]
    data = pd.read_csv(os.path.join(path, "driving_log.csv"), names=columns)
    # print(data.head())
    data["Center"] = data["Center"].apply(get_name)
    # print(data.head())
    print(f"Total Images in this Column: {data.shape[0]}")
    return data

def balance_data(data, display = True):
    number_of_bins = 31
    samples_per_bin = 1000
    hist, bins = np.histogram(data["Steering"], number_of_bins)
    # print(bins)    # but there is no 0 value here
    if display:
        center = (bins[1:] + bins[:-1]) * 0.5
        # print(center)
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1, 1), (samples_per_bin, samples_per_bin))
        plt.show()

    # To remove the upper extra values of 0
    remove_index_list = []
    for j in range(number_of_bins):
        bin_data_list = []
        for i in range (len(data["Steering"])):
            if data["Steering"][i] >= bins[j] and data["Steering"][i] <= bins[j+1]:
                bin_data_list.append(i)
        bin_data_list = shuffle((bin_data_list))
        bin_data_list = bin_data_list[samples_per_bin:]

        remove_index_list.extend(bin_data_list)

    print("Removed Images: ", len(remove_index_list))
    data.drop(data.index[remove_index_list], inplace=True)
    print("Remaining Images: ", len(data))

    if display:
        hist, _ = np.histogram(data["Steering"], number_of_bins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samples_per_bin, samples_per_bin))
        plt.show()

    return data

def load_data(path, data):
    images_path = []
    steering_values = []

    for i in range(len(data)):
        indexed_data = data.iloc[i]
        # print(indexed_data)

        images_path.append(os.path.join(path, 'IMG', indexed_data[0]))
        steering_values.append((float(indexed_data[3])))

    images_path = np.asarray(images_path)
    steering_values = np.asarray(steering_values)

    return images_path, steering_values

def aug_img(image, steering_angle):
    img = mpimg.imread(image)

    # PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x': (-0.10, 0.10), 'y': (-0.10, .10)})
        img = pan.augment_image(img)

    # ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    # BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.7))
        img = brightness.augment_image(img)

    # FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering_angle = -(steering_angle)

    return img, steering_angle

# img_result, steering_angle = aug_img('test.jpg', 0)
# plt.imshow(img_result)
# plt.show()

def preprocessing(img):
    img = img[60:135, :, :]                        # Cropping the height only
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)     # Changing the ColorSpace
    img = cv2.GaussianBlur(img, (3, 3), 0)         # Adding blur
    img = cv2.resize(img, (200, 66))               # Resize the image
    img = img/255    # doing Normalization[keeping the values in (0-1) instead of (0-255)]

    return img


# img_result = preprocessing(mpimg.imread('test.jpg'))
# plt.imshow(img_result)
# plt.show()

def batch_generator(images_path, steering_values, batch_size, trainFlag):
    while True:
        img_batch = []
        steering_batch = []

        for i in range(batch_size):
            index = random.randint(0, len(images_path)-1)
            if trainFlag:
                # We want to augment the image only for training, not for validation
                img, steering = aug_img(images_path[index], steering_values[index])
            else:
                img = mpimg.imread(images_path[index])
                steering = steering_values[index]

            img = preprocessing(img)
            img_batch.append(img)
            steering_batch.append(steering)
        yield np.asarray(img_batch), np.asarray(steering_batch)


def create_model():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2),  activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2),  activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')
    return model


