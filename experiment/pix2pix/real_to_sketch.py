# 학습한 pix2pix 모델을 통해 원하는 image를 translation
# 필요한 library import
from keras.models import load_model
import tensorflow
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
 
# image load
def load_image(filename, size=(256,256)):
	# image load 및 256*256으로 resizing
	pixels = tensorflow.keras.preprocessing.image.load_img(filename, target_size=size)
	# numpy array로 변환
	pixels = tensorflow.keras.preprocessing.image.img_to_array(pixels)
	# [0,255]에서 [-1,1]로 scaling
	pixels = (pixels - 127.5) / 127.5
	# 1 sample로 reshape
	pixels = expand_dims(pixels, 0)
	return pixels
 
# source image load
src_image = load_image('experiment/data/real2sketch/test_sample.jpeg')
print('Loaded', src_image.shape)
# model load
model = load_model('experiment/pix2pix/model/model_045500.h5')
# source로부터 image 생성
gen_image = model.predict(src_image)
# [-1,1]에서 [0,1]로 scaling
gen_image = (gen_image + 1) / 2.0
# image plot
pyplot.imshow(gen_image[0])
pyplot.axis('off')
pyplot.show()