import tensorflow as tf
from PIL import Image
import numpy as np

#carregar o modelo salvo
loaded_model = tf.keras.models.load_model('mnist_model.keras')

#abrindo imagem
img = Image.open('./numero2.png')
x = img.size[0]
y = img.size[1]
pixel = img.load()

#criando variavel com os valores dos pizxels em 0 
img_data = np.zeros((y, x))

#loop de escrita dos valores dos pixels da imagem

for yy in range (y):
    for xx in range(x):
        if pixel[xx, yy][0] <255:
            img_data[yy, xx] = (255 -pixel[xx, yy][0])/255

#ajuste dos dados da imagem
data = []
data.append(img_data)
data = np.array(data)

prediction = loaded_model.predict(data)
print('Predição: ', np.argmax(prediction, axis=1))
