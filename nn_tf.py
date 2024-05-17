import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Distintos valores a probar
activations = ['tanh', 'sigmoid']
neurons = [16, 24, 32, 48, 64]
epochs = range(10, 101, 10)

# Cargar datos de entrenamiento y prueba desde archivos CSV
train_data = pd.read_csv('components_train.csv')
test_data = pd.read_csv('components_test.csv')

# Separar características (X) y etiquetas (y)
X_train = train_data.drop('Name', axis=1).values#.reshape(-1,1)
y_train = train_data['Name'].values

X_test = test_data.drop('Name', axis=1).values#.reshape(-1,1)
y_test = test_data['Name'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir etiquetas a valores numéricos si es necesario
onehot_encoder = OneHotEncoder()
y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = onehot_encoder.transform(y_test.reshape(-1, 1)).toarray()

# Función que varía la activación utilizada y la cantidad de neuronas
def create_model(activation, neurons):
    # Inicializar el constructor
    model = Sequential()

    for i in range(2):
        model.add(Dense(neurons, activation=activation))

    model.add(Dense(19, activation = 'softmax'))

    sgd = keras.optimizers.SGD(learning_rate=1.0)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )
    return model

# Implementación de grid search
for activation in activations:
    for neuron in neurons:
        for epoch in epochs:
            mod = create_model(activation, neuron)
            mod.fit(X_train, y_train, epochs=epoch, verbose=0)
            print([activation, neuron, epoch])
            print(mod.evaluate(X_test, y_test, verbose=0))
