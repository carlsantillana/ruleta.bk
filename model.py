import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

MODEL_PATH = 'models/number_model.keras'

@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def train_number_model(numbers, n_steps=10):
    X, y = [], []
    for i in range(len(numbers) - n_steps):
        X.append(numbers[i:i + n_steps])
        y.append(numbers[i + n_steps])
    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
        tf.keras.layers.LSTM(25, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=mse)
    model.fit(X, y, epochs=200, batch_size=16, verbose=1, validation_split=0.4)

    model.save(MODEL_PATH, save_format='keras')

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={'mse': mse})

def evaluate_model(model, X_test, y_test):
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_pred = model.predict(X_test, verbose=0)
    y_pred = np.round(y_pred).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def predict_next_numbers(model, numbers, n_steps, num_predictions=4):
    if model is None:
        return ["Model not trained yet"]
    input_seq = numbers[-n_steps:]
    input_seq = np.array(input_seq).reshape((1, n_steps, 1))
    predictions = []

    for _ in range(num_predictions):
        pred = model.predict(input_seq, verbose=0)
        pred = int(pred[0][0])
        if pred < 0:
            pred = 0
        elif pred > 36:
            pred = 36
        if pred not in predictions:
            predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
    
    return sorted(predictions)

def reset_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
