# DNN model training using the ECFP4 fingerprint as input
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import pickle

# load the dataset
evaluation_ecfp4_binary_df = pd.read_pickle('evaluation_ecfp4_binary_df.pkl')
X = np.array(evaluation_ecfp4_binary_df['ECFP4_fingerprint'].tolist())  
Y = np.array(evaluation_ecfp4_binary_df['binary_vector_evaluation'].tolist())   

# define the DNN model as described in the paper
def create_dnn_model(input_dim, output_dim):
    model = models.Sequential()
    
    # input layer and the first hidden layer (1000 neurons)
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.2))  
    
    # second hidden layer (500 neurons)
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.2)) 
    
    # output layer (7546 neurons  corresponding to 7546 targets)
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    
    # compile the model with Adam optimizer 
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# input and output dimensions
input_dim = X.shape[1] 
output_dim = Y.shape[1] 

# set up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_number = 1

# iterate over each fold
for train_index, test_index in kf.split(X):
    print(f"Processing fold {fold_number}...")

    # split the data into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    model = create_dnn_model(input_dim, output_dim)
    
    # train the model 
    model.fit(X_train, Y_train, epochs=200, batch_size=500, validation_data=(X_test, Y_test))
    
    # save the model for each fold
    model.save(f'dnn_model_fold_{fold_number}.h5')
    print(f"Model for fold {fold_number} saved as 'dnn_model_fold_{fold_number}.h5'")
    
    # make predictions for the test set
    Y_pred = model.predict(X_test)
    print(f"Predictions for fold {fold_number} completed.")
    
    # store predictions
    predictions_data = {
        "fold_number": fold_number,
        "test_index": test_index,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_pred": Y_pred,
    }
    with open(f'predictions_fold_{fold_number}.pkl', 'wb') as f:
        pickle.dump(predictions_data, f)
    print(f"Predictions and test set for fold {fold_number} saved as 'predictions_fold_{fold_number}.pkl'")
    fold_number += 1
