import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout, BatchNormalization, Masking

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))  # Add a Masking layer
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
 
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def quantize_model(model):
    # Define a quantization aware model
    q_aware_model = tf.keras.models.clone_model(model)
    q_aware_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Quantize the model
    quantize_model = tfmot.quantization.keras.quantize_model(q_aware_model)

    return quantize_model
