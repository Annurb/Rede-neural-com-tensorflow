from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

#carrega o dataset do mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/ 255.0, x_test/255.0

#modelo de rede neural
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#compila o modelo
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#treina o modelo
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

#faz as previsoes do x_test
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

#mostra as primeiras 10 previsoes
print('Primeiras 10 previsôes: ')
for i in range(10):
    print(f'Imagem {i+1}: Previsto = {predicted_classes[i]}, Real = {y_test[i]}')

#salva o modelo em um arquivo .keras
model.save('mnist_model.keras')