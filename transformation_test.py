import improc, numpy
import tensorflow as tf

from PIL import Image
import elasticdeform.tf as etf
import elasticdeform
from random import random
import imageio

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


X = numpy.zeros((200, 300, 3))
X[::10, ::10,0] = 50
X[::10, ::10,1] = 80
X[::10, ::10,2] = 155
#X = numpy.zeros((200, 300))
#X[::10, ::10] = 50


# apply deformation with a random 3 x 3 grid
displacement = numpy.random.randn(3,1, 3, 3) * 25

X_deformed = elasticdeform.deform_grid(X, displacement, axis=(0, 1, 2))
#X_deformed = elasticdeform.deform_random_grid(X, sigma=10, points=3)

imageio.imsave('test_X.png', X)
imageio.imsave('test_X_deformed.png', X_deformed)

#displacement_val = numpy.random.randn(3,2,2,1)/5



rotation = random()*360
print(rotation)
x_rotated = improc.rotate(x_train[0:10],rotation)


Image.fromarray(x_train[1]).save("/Volumes/Transcend/TU Delft/Master/CS4240 | Deep Learning/Reprocubility Project/pyoneer-main/cifar_sample.png")
Image.fromarray(x_rotated[1]).save("/Volumes/Transcend/TU Delft/Master/CS4240 | Deep Learning/Reprocubility Project/pyoneer-main/cifar_sample_deformed.png")

