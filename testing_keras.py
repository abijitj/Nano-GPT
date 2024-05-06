# Importing required libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# Sample data
# Let's assume we have 100 samples with 3 features each
X = np.random.random((100, 3))
# And we want to predict a target variable for each sample
y = np.random.randint(2, size=(100, 1))

# Creating the model
model = Sequential()

# Adding a fully connected layer with 4 neurons and specifying the input shape
# Since this is the first layer in the model, we need to specify the input shape
model.add(Dense(4, input_shape=(3,), activation='relu'))

# Adding the output layer with 1 neuron since we're predicting a single value
# Using sigmoid activation function for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X, y, epochs=10, batch_size=10)

# Now, you can use this trained model for predictions

