# 학습한 cyclegan 모델을 통해 원하는 image를 translation
# 필요한 library import
from numpy import load
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow
from matplotlib import pyplot
 
# image load
def load_image(filename, size=(256,256)):
	# image load 및 256*256으로 resizing
	pixels = tensorflow.keras.preprocessing.image.load_img(filename, target_size=size)
	# numpy array로 변환
	pixels = tensorflow.keras.preprocessing.image.img_to_array(pixels)
	# sample에서 변환
	pixels = expand_dims(pixels, 0)
	# [0,255]에서 [-1,1]로 scaling
	pixels = (pixels - 127.5) / 127.5
	return pixels
 
# image load
image_src = load_image('real2sketch/test_sample.jpeg')
# model load
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_009000.h5', cust)
# image translate
image_tar = model_AtoB.predict(image_src)
# [-1,1]에서 [0,1]로 scaling
image_tar = (image_tar + 1) / 2.0
# translated image plot
pyplot.imshow(image_tar[0])
pyplot.show()