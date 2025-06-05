# ann_surrogate_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_ann_model(input_dim, output_dim, hidden_layers=, activation_func='relu', dropout_rate=0.1):
    """
    Membuat model JST (MLP).
    :param input_dim: Dimensi input (jumlah parameter TMD, misal 3: md, kd, cd)
    :param output_dim: Dimensi output (jumlah metrik performa, misal 1: max displacement)
    :param hidden_layers: List jumlah neuron di setiap hidden layer
    :param activation_func: Fungsi aktivasi untuk hidden layers
    :param dropout_rate: Rate untuk dropout layer (0 untuk tidak ada dropout)
    :return: Model Keras yang belum dilatih
    """
    model = Sequential()
    model.add(Dense(hidden_layers, input_dim=input_dim, activation=activation_func))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    for neurons in hidden_layers[1:]:
        model.add(Dense(neurons, activation=activation_func))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
            
    model.add(Dense(output_dim, activation='linear')) # Output biasanya linear untuk regresi

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

def train_ann_model(model, X_data, y_data, epochs=100, batch_size=32, validation_split=0.2):
    """
    Melatih model JST.
    Data akan di-scale (standardized) sebelum pelatihan.
    :return: Tuple (trained_model, history, scaler_X, scaler_y)
    """
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=validation_split, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1) if len(y_train.shape)==1 else y_train)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1,1) if len(y_val.shape)==1 else y_val)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train_scaled, y_train_scaled,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val_scaled, y_val_scaled),
                        callbacks=[early_stopping],
                        verbose=1) # Set verbose=0 untuk silent training
    
    return model, history, scaler_X, scaler_y

def predict_performance_ann(ann_model, tmd_params_array, scaler_X, scaler_y):
    """
    Memprediksi performa menggunakan model JST yang sudah dilatih.
    :param ann_model: Model Keras yang sudah dilatih.
    :param tmd_params_array: NumPy array dari parameter TMD (n_samples x n_features).
    :param scaler_X: Scaler yang digunakan untuk input saat training.
    :param scaler_y: Scaler yang digunakan untuk output saat training.
    :return: Prediksi performa (sudah di-inverse transform).
    """
    tmd_params_scaled = scaler_X.transform(tmd_params_array)
    predictions_scaled = ann_model.predict(tmd_params_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    return predictions

if __name__ == '__main__':
    # Contoh Penggunaan (dummy data)
    num_samples = 500
    # Input: [m_d, k_d, c_d]
    X_dummy = np.random.rand(num_samples, 3) * np.array([100, 1e5, 1e3]) 
    # Output: [max_displacement] (contoh sederhana, fungsi non-linear dummy)
    y_dummy = 0.5 * X_dummy[:,0]/10 - 0.2 * X_dummy[:,1]/1e4 + 0.1 * X_dummy[:,2]/1e2 + np.random.randn(num_samples) * 0.05
    y_dummy = y_dummy.reshape(-1,1)

    input_dim = X_dummy.shape
    output_dim = y_dummy.shape

    ann = create_ann_model(input_dim, output_dim)
    ann_trained, history, sc_X, sc_y = train_ann_model(ann, X_dummy, y_dummy, epochs=50) # epochs sedikit untuk tes cepat

    # Test prediction
    test_params = np.array([[50, 5e4, 500]]) # Satu sampel
    predicted_val = predict_performance_ann(ann_trained, test_params, sc_X, sc_y)
    print(f"Parameter TMD tes: {test_params}")
    print(f"Prediksi performa (misal, max_disp): {predicted_val}")

    # Plot learning curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Kurva Pembelajaran Model JST')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()