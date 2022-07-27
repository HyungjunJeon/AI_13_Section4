# 학습한 model을 이용해 dataset 중 random한 image로 translation 결과 확인
# 필요한 library import
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
 
# training image들 load해서 준비
def load_real_samples(filename):
	# dataset load
	data = load(filename)
	# arrays unpack
	X1, X2 = data['arr_0'], data['arr_1']
	# [0,255]에서 [-1,1]로 scaling
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 
# dataset로부터 random하게 image sample 선택
def select_sample(dataset, n_samples):
	# random instances 선택
	ix = randint(0, dataset.shape[0], n_samples)
	# 선택된 image들
	X = dataset[ix]
	return X
 
# image, translation, reconstruction plot
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
	# [-1,1]에서 [0,1]로 scaling
	images = (images + 1) / 2.0
	# row by row로 image들 plot
	for i in range(len(images)):
		# subplot 정의
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# raw pixel data plot
		pyplot.imshow(images[i])
		# title 보여주기
		pyplot.title(titles[i])
	pyplot.show()
 
# dataset load
A_data, B_data = load_real_samples('real2sketch.npz')
print('Loaded', A_data.shape, B_data.shape)
# model들 load
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_009000.h5', cust)
model_BtoA = load_model('g_model_BtoA_009000.h5', cust)
# A->B->A plot
A_real = select_sample(A_data, 1)
B_generated  = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)
# B->A->B plot
B_real = select_sample(B_data, 1)
A_generated  = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)