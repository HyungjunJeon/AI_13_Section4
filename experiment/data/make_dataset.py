# pix2pix model 학습을 위한 dataset 만들기
# 필요한 library import
from os import listdir
import tensorflow
from numpy import asarray
from numpy import vstack
from numpy import savez_compressed
 
# directory 내의 image 파일들 불러오기
# 256*256 size로 resizing
def load_images(path, size=(256,256)):
	data_list = list()
	# directory 내의 모든 image 중 1%만 불러오기 (시간 및 memory 문제 해결)
	for i, filename in enumerate(listdir(path)):
		if i > int(len(listdir(path)) / 100):
			break
		# image load 및 resizing
		pixels = tensorflow.keras.preprocessing.image.load_img(path + filename, target_size=size)
		# numpy array로 변환
		pixels = tensorflow.keras.preprocessing.image.img_to_array(pixels)
		# 저장
		data_list.append(pixels)
	return asarray(data_list)
 
# dataset 경로
path = 'common/data/real2sketch/'
# dataset A load
dataA1 = load_images(path + 'trainA/')
dataAB = load_images(path + 'testA/')
dataA = vstack((dataA1, dataAB))
print('Loaded dataA: ', dataA.shape)
# dataset B load
dataB1 = load_images(path + 'trainB/')
dataB2 = load_images(path + 'testB/')
dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
# 압축된 numpy array로 저장
filename = 'experiment/data/real2sketch.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)