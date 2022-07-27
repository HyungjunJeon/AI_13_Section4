# 학습한 model을 이용해 dataset 중 random한 image로 translation 결과 확인
# 필요한 library import
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
 
# training image들 load 및 준비
def load_real_samples(filename):
	# 압축된 arrays load
	data = load(filename)
	# arrays unpack
	X1, X2 = data['arr_0'], data['arr_1']
	# [0,255]에서  [-1,1]로 scaling
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 
# source, generated and target image들 plot
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# [-1,1]에서 [0,1]로 scaling
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# image들 row by row로 plot
	for i in range(len(images)):
		# subplot 정의
		pyplot.subplot(1, 3, 1 + i)
		# axis turn off
		pyplot.axis('off')
		# raw pixel data plot
		pyplot.imshow(images[i])
		# title 보여주기
		pyplot.title(titles[i])
	pyplot.show()
 
# dataset load
[X1, X2] = load_real_samples('experiment/data/real2sketch.npz')
print('Loaded', X1.shape, X2.shape)
# model load
model = load_model('experiment/pix2pix/model/model_045500.h5')
# random example 선택
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# source로부터 image 생성
gen_image = model.predict(src_image)
# source, generated, target image들 plot
plot_images(src_image, gen_image, tar_image)
