# cyclegan model 정의 및 학습
# 필요한 library import
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
 
# 판별자 model 정의
def define_discriminator(image_shape):
	# 가중치 초기화
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# model 정의
	model = Model(in_image, patch_out)
	# compile model
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model
 
# resnet block 생성자
def resnet_block(n_filters, input_layer):
	# 가중치 초기화
	init = RandomNormal(stddev=0.02)
	# 첫번째 convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# 두번째 convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# 채널에 따라 imput layer concatenate
	g = Concatenate()([g, input_layer])
	return g
 
# 독립형 model 정의
def define_generator(image_shape, n_resnet=9):
	# 가중치 초기화
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# model 정의
	model = Model(in_image, out_image)
	return model
 
# 적대적과 cycle loss에 의한 생성자 update를 위해 composite model 정의
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# 생성자 model을 trainable하게 설정
	g_model_1.trainable = True
	# 판별자 model trainable 하지 않게 설정
	d_model.trainable = False
	# 다른 생성자 model을 trainable 하지 않게 설정
	g_model_2.trainable = False
	# 판별자 element
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity element
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# forward cycle
	output_f = g_model_2(gen1_out)
	# backward cycle
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
	# model graph 정의
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	# optimization algorithm configuration 정의
	opt = Adam(lr=0.0002, beta_1=0.5)
	# 최소 squares loss and L1 loss의 가중치로 model compile 
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model
 
# training image들 load 및 준비
def load_real_samples(filename):
	# dataset load
	data = load(filename)
	# arrays unpack
	X1, X2 = data['arr_0'], data['arr_1']
	# [0,255]에서 [-1,1]로 scaling
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 
# random sample들의 batch 선택 및 image들과 target return
def generate_real_samples(dataset, n_samples, patch_shape):
	# random instances 선택
	ix = randint(0, dataset.shape[0], n_samples)
	# 선택된 image들
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y
 
# image들의 batch 생성 및  image들과 target들 return
def generate_fake_samples(g_model, dataset, patch_shape):
	# fake instance 생성
	X = g_model.predict(dataset)
	# 'fake' class labels (0) 생성
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y
 
# generator model들을 file로 저장
def save_models(step, g_model_AtoB, g_model_BtoA):
	# 첫번째 생성자 model 저장
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# 두번째 생성자 model 저장
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
 
# sample들 생성하고 plot으로 저장 및 model 저장
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# input image들 sample 선택
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# translated image들 생성
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# 모든 pixel들을 [-1,1]에서 [0,1]로 scaling
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# real image들 plot
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	# translated image plot
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	# plot을 file로 저장
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	pyplot.savefig(filename1)
	pyplot.close()
 
# fake image들을 위한 image pool update
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# pool 저장
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# image를 사용하지만 pool에 저장하지는 않음
			selected.append(image)
		else:
			# 존재하는 image를 대체하고 대체된 image를 사용
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)
 
# cyclegan model들 학습
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
	# 학습 property들 정의
	n_epochs, n_batch, = 100, 1
	# 판별자 출력 square shape 정의
	n_patch = d_model_A.output_shape[1]
	# dataset unpack
	trainA, trainB = dataset
	# fake들을 위한 image pool 준비
	poolA, poolB = list(), list()
	# 학습 epoch마다의 batch 수 계산
	bat_per_epo = int(len(trainA) / n_batch)
	# 학습 반복 횟수 계산
	n_steps = bat_per_epo * n_epochs
	# 수동으로 epoch 반복
	for i in range(n_steps):
		# real sample들의 batch 선택
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		# fake sample들의 batch 생성
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
		# pool로부터 fakes update
		X_fakeA = update_image_pool(poolA, X_fakeA)
		X_fakeB = update_image_pool(poolB, X_fakeB)
		# adversarial와 cycle loss를 통해 generator B->A update
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		# A -> [real/fake]를 위한 판별자 update
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		# adversarial and cycle loss를 통해 generator A->B update 
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		# B -> [real/fake]를 위한 판별자 update
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		# performance 요약
		print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
		# model performance 평가
		if (i+1) % (bat_per_epo * 1) == 0:
			# A->B translation
			summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
			# B->A translation
			summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
		if (i+1) % (bat_per_epo * 5) == 0:
			# model들 저장
			save_models(i, g_model_AtoB, g_model_BtoA)
 
# image data load
dataset = load_real_samples('real2sketch.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# load된 dataset에 기반해 image shape 정의
image_shape = dataset[0].shape[1:]
# 생성자 A -> B
g_model_AtoB = define_generator(image_shape)
# 생성자 B -> A
g_model_BtoA = define_generator(image_shape)
# 판별자 A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# 판별자 B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# model들 학습
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)