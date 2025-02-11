import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load the dataset
evaluation_fused_binary_df = pd.read_pickle('ECFP4_4096_BINARY.pkl')
X = np.array(evaluation_fused_binary_df['ECFP4_fingerprint'].tolist())  
Y = np.array(evaluation_fused_binary_df['binary_vector'].tolist())   

# Define the DNN model as described in the paper
def create_dnn_model(input_dim, output_dim):
    model = models.Sequential()
    
    # Input layer and the first hidden layer (1000 neurons)
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.2))  # 20% dropout to prevent overfitting
    
    # Second hidden layer (500 neurons)
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.2))  # 20% dropout to prevent overfitting
    
    # Output layer (100 neurons - corresponding to 100 targets)
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    
    # Compile the model with Adam optimizer and binary crossentropy loss
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Set input and output dimensions
input_dim = X.shape[1] 
output_dim = Y.shape[1] 

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Create the model
model = create_dnn_model(input_dim, output_dim)

# Train the model on the training data with early stopping
model.fit(X_train, Y_train, epochs=200, batch_size=500, validation_data=(X_test, Y_test), callbacks=[early_stopping])

# Save the trained model
model.save('ecfp4_4096_dnn_model.h5')
print("Model saved as 'ecfp4_4096_dnn_model.h5'")

# Make predictions for the test set
Y_pred = model.predict(X_test)
print("Predictions completed.")

# Store predictions along with the actual test data
predictions_data = {
    "X_test": X_test,
    "Y_test": Y_test,
    "Y_pred": Y_pred,
}

# Save predictions and test set as a pickle file
with open('predictions.pkl', 'wb') as f:
    pickle.dump(predictions_data, f)
print("Predictions and test set saved as 'predictions.pkl'")

