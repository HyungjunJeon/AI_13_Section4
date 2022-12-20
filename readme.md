# Pix2pix 활용 스케치 프로젝트

코드스테이츠 AI 부트캠프 Section4 프로젝트입니다

## 프로젝트 소개

가상의 한국인 사진 이미지와 각각에 대응하는 스케치 이미지를 Pix2pix에 학습시켜 사진을 스케치로 변환하도록 하는 프로젝트 입니다

## comon 디렉토리

원본 image data 디렉토리입니다
사진, 스케치 image train 각 8000여장, test 각 1000여장 포함
용량 문제로 github에 업로드 되지는 않았습니다

## Pix2pix 디렉토리

Pix2pix 활용 프로젝트 관련 디렉토리 입니다

- data 디렉토리 : dataset 및 관련 코드 포함
- model 디렉토리 : 총 100번의 epoch 중 10번마다의 model 포함
- plot 디렉토리 : 총 100번의 epoch 중 10번마다의 source, generated, target image들 포함
- pix2pix.py : pix2pix 모델 구성 및 학습과 관련된 코드
- image_translation.py : 학습한 model을 이용해 dataset 중 random한 image로 translation 결과를 확인하는 코드
- real_to_sketch.py : 학습한 pix2pix 모델을 통해 원하는 image를 translation

## experiment 디렉토리

paired dataset을 cyclegan에 학습시켜 pix2pix와 비교하면 어떤 결과가 나올지 확인해본 실험 관련 디렉토리 입니다
각 모델은 cyclegan과 pix2pix 디렉토리 내에 있으며 구성은 상단과 유사합니다

## result 디렉토리
발표 영상과 발표 자료가 담긴 디렉토리입니다
pptx 파일과 pdf 파일의 내용은 동일합니다
