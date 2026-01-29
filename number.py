import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test/ 255.0

print('\nIniciando modo de visualização de números...\n')

loop = True
error = False

while loop:
    if error:
        print('Voce digitou uma resposta invalida.')
        error = False

    response = input('Escolha um index de exibição de um número, (0 a 59999)\nOu digite "sair".\nResposta: ')

    if response == 'sair':
        loop = False
    else:
        try:
            index = int(response)
            print('Número: ', y_train[index])
            plt.imshow(x_train[index], cmap='binary')
            plt.axis('off')
            plt.show()
        except:
            error = True
        print()