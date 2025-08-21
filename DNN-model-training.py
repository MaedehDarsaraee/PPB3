# DNN model training using the ECFP4 fingerprint as input
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import pickle

# Load the dataset
evaluation_ecfp4_binary_df = pd.read_pickle('evaluation_ecfp4_binary_df.pkl')
X = np.array(evaluation_ecfp4_binary_df['ECFP4_fingerprint'].tolist())  
Y = np.array(evaluation_ecfp4_binary_df['binary_vector_evaluation'].tolist())   

# Define the DNN model as described in the paper
def create_dnn_model(input_dim, output_dim):
    model = models.Sequential()
    
    # Input layer and the first hidden layer (1000 neurons)
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.2))  
    
    # Second hidden layer (500 neurons)
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.2)) 
    
    # Output layer (7546 neurons  corresponding to 7546 targets)
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    
    # Compile the model with Adam optimizer 
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Set input and output dimensions
input_dim = X.shape[1] 
output_dim = Y.shape[1] 

# Set up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Track the fold number
fold_number = 1

# Iterate over each fold
for train_index, test_index in kf.split(X):
    print(f"Processing fold {fold_number}...")

    # Split the data into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    # Create a new instance of the model for each fold
    model = create_dnn_model(input_dim, output_dim)
    
    # Train the model on the training data with early stopping
    model.fit(X_train, Y_train, epochs=200, batch_size=500, validation_data=(X_test, Y_test)])
    
    # Save the model for each fold
    model.save(f'dnn_model_fold_{fold_number}.h5')
    print(f"Model for fold {fold_number} saved as 'dnn_model_fold_{fold_number}.h5'")
    
    # Make predictions for the test set
    Y_pred = model.predict(X_test)
    print(f"Predictions for fold {fold_number} completed.")
    
    # Store predictions along with the actual test data for each fold
    predictions_data = {
        "fold_number": fold_number,
        "test_index": test_index,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_pred": Y_pred,
    }

    # Save predictions and test set as a pickle file
    with open(f'predictions_fold_{fold_number}.pkl', 'wb') as f:
        pickle.dump(predictions_data, f)
    print(f"Predictions and test set for fold {fold_number} saved as 'predictions_fold_{fold_number}.pkl'")
    fold_number += 1
