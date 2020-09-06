from utilities import *
from sklearn.model_selection import train_test_split

#### 1. Importing the data information
path = 'data'
data = import_data_info(path)


# Next, We have to balance the data so that we have equal number of information from
# the center, left and right camera.


#### 2. Visualization and Distribution of data
data = balance_data(data, display=False)

#### 3. Preparing the data for processing

# We have to put all the images in one list and the steering values in another list.
# Right now the data is in pandas format.We'll put it in a list and convert it into a numpy array

images_path, steering_values = load_data(path, data)
print((images_path[0], steering_values[0]))

#### 4. Splitting the data into train and validation set
x_train, x_val, y_train, y_val = train_test_split(images_path, steering_values,
                                                   test_size=0.20, random_state=42)
print("Images for Training: ", len(x_train))
print("Images for Validation: ", len(x_val))


#### 5. Augment the Data to add more variety and variance

# This will help us to train the model more efficiently.
# By the augmentation of our data, we can increase the number of images we have.


#### 6. Pre-processing the Data

# We want to crop the car in the image.
# We also want to remove the background(mountain, tress) from our image.


#### 7. Generating Batches

# We don't send all the images together to the training model.
# We send the images in bathes. It helps in generalization.
# But before sending to the model, we must augment and pre-process the images


#### 8. Creating the model

# We are going to use the model propsed by NVIDIA for end-to-end self driving
# We will use Kera from Tensorflow to build the model

model = create_model()
model.summary()

#### 9. Training the Model

# We will input the images and the steering. But we will not directly input them.
# We will create batches of them and then send them in the model to fit

outputs = model.fit(batch_generator(x_train, y_train, 100, 1), steps_per_epoch=300, epochs=10,
           validation_data=batch_generator(x_val, y_val, 100, 0), validation_steps=200)

#### 10. Saving the Model and Plot the Losses
model.save('model2.h5')
print("Your model has been saved.")

# Plotting the losses
plt.plot(outputs.history['loss'])
plt.plot(outputs.history['val_loss'])
plt.legend(["Training Data", "Validation Data"])
plt.ylim([0, 1])
plt.title("Losses in the model")
plt.xlabel("Epoch")
plt.show()

