# pix2pix model 정의 및 학습
# 필요한 library import
from numpy import load
from numpy import zeros
from numpy import ones
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
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
 
# 판별자 model 정의
def define_discriminator(image_shape):
	# 가중치 초기화
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# 채널에 따라  images concatenate
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# model 정의
	model = Model([in_src_image, in_target_image], patch_out)
	# model compile
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model
 
# encoder block 정의
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# 가중치 초기화
	init = RandomNormal(stddev=0.02)
	# downsampling layer 추가
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# batch 정규화 추가
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu 활성화함수
	g = LeakyReLU(alpha=0.2)(g)
	return g
 
# decoder block 정의
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# 가중치 초기화
	init = RandomNormal(stddev=0.02)
	# upsampling layer 추가
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# batch 정규화 추가
	g = BatchNormalization()(g, training=True)
	# dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu 활성화함수
	g = Activation('relu')(g)
	return g
 
# 독립형 generator model 정의
def define_generator(image_shape=(256,256,3)):
	# 가중치 초기화
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# model 정의
	model = Model(in_image, out_image)
	return model
 
# generator update를 위해 generator와 discriminator model 결합해 정의
def define_gan(g_model, d_model, image_shape):
	# discriminator의 가중치를 학습되지 않게 만들기
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# source image 정의
	in_src = Input(shape=image_shape)
	# source image를 generator input에 연결
	gen_out = g_model(in_src)
	# source input과 generator output을 discriminator input에 연결
	dis_out = d_model([in_src, gen_out])
	# src image를 input, generated image와 classification output
	model = Model(in_src, [dis_out, gen_out])
	# model compile
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model
 
# training images load해서 준비
def load_real_samples(filename):
	# 압축된 arrays load
	data = load(filename)
	# arrays unpack
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 
# random한 sample batch 선택, image들과 target return
def generate_real_samples(dataset, n_samples, patch_shape):
	# dataset unpack
	trainA, trainB = dataset
	# random instance 선택
	ix = randint(0, trainA.shape[0], n_samples)
	# 선택된 image들
	X1, X2 = trainA[ix], trainB[ix]
	# 'real' class labels (1) 생성
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y
 
# batch of image들의 batch 생성, image들과  target들 return
def generate_fake_samples(g_model, samples, patch_shape):
	# fake instance 생성
	X = g_model.predict(samples)
	# 'fake' class labels (0) 생성
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y
 
# sample들을 생성하고 plot으로 저장 및 model 저장
def summarize_performance(step, g_model, dataset, n_samples=3):
	# input image들의 sample 선택
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# fake sample들의 batch 생성
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# 모든 pixel들을 [-1,1]에서 [0,1]로 scaling
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# real source image들 plot
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# 생성된 target image plot
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# real target image plot
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# plot을 file로 저장
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig('experiment/pix2pix/plot/' + filename1)
	pyplot.close()
	# generator model 저장
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save('experiment/pix2pix/model/' + filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
 
# pix2pix model들 학습
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
	# 판별자의 output square shape 정의
	n_patch = d_model.output_shape[1]
	# dataset unpack
	trainA, trainB = dataset
	# training epoch마다의 batch들의 수 계산
	bat_per_epo = int(len(trainA) / n_batch)
	# training iteration 수 계산
	n_steps = bat_per_epo * n_epochs
	# 수동으로 epochs 반복
	for i in range(n_steps):
		# real sample들의 batch 선택
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# fake sample들의 batch 생성
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# real sample들에 대해 판별자 update
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# 생성된 sample들에 대해 판별자 update
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# 생성자 update
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# model performance 요약
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)
 
# image data load
dataset = load_real_samples('experiment/data/real2sketch.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# load된 dataset에 기반해 input shape 정의 
image_shape = dataset[0].shape[1:]
# model들 정의
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# composite model 정의
gan_model = define_gan(g_model, d_model, image_shape)
# model 학습
train(d_model, g_model, gan_model, dataset)