import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load dataset
df = pd.read_pickle('Fingerprint_data.pkl')
X = df['fp'].values.tolist()  
Y = df['binary_vector'].values.tolist()

# Convert lists to numpy arrays
X, Y = np.array(X), np.array(Y)

# Define DNN model
def create_dnn_model(input_dim, output_dim):
    model = models.Sequential([
        layers.Dense(1000, activation='relu', input_shape=(input_dim,)),  # first hidden layer
        layers.Dropout(0.2),
        layers.Dense(500, activation='relu'), # second hidden layer
        layers.Dropout(0.2),
        layers.Dense(output_dim, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train model
model = create_dnn_model(X.shape[1], Y.shape[1])
model.fit(X_train, Y_train, epochs=200, batch_size=500, validation_data=(X_test, Y_test), callbacks=[early_stopping])

# Save model
model.save('dnn_model.h5')
