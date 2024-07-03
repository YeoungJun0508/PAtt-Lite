## PAtt-Lite: Lightweight Patch and Attention MobileNet for Challenging Facial Expression Recognition


 MobileNetV1을 기반으로 한 경량 패치 및 주의 네트워크인 PAtt-Lite.


### 용어
- 패치: 이미지 전체에서 잘라낸 작은 부분 또는 조각
- 특징 맵: 합성곱 신경망의 합성곱 층을 통과한 후 생성되는 다차원 배열 ex) 2차원 배열(예: 높이 x 너비) 또는 3차원 배열(예: 높이 x 너비 x 채널) 형태


 ### 모델구조

 - 백본 모델 : MobileNetV1 , 기본적인 특징 추출기로 경량화된 신경망으로서, 깊이별 분리 합성곱(depthwise separable convolutions)을 사용하여 파라미터 수와 연산량을 줄인다.

-  패치 추출 블록
  1. 첫 번째 합성곱 층, 작은 커널 크기(예: 3x3)를 사용하는 합성곱 층으로 초기 특징 맵을 생성.
     - ex) 눈, 코, 입과 같은 얼굴의 주요 구성 요소를 탐지
  2. 두 번째 합성곱 층, 더 큰 커널 크기, 초기 특징 맵을 더 깊이 있는 특징으로 변환하여 다양한 얼굴 구성 요소의 세부 사항을 더 잘 포착
     - ex) 눈썹의 모양, 입술의 곡선 등
  3. 세 번째 합성곱 층, 다양한 크기의 커널을 사용하여 서로 다른 스케일의 특징을 통합
     - ex) 전체 얼굴 윤곽과 같은 큰 스케일의 특징과 눈, 입술 등의 작은 스케일의 특징을 모두 포착

- 주의 분류기(Attention Classifier): 특징맵 정제.
  1. Global Average Pooling (GAP)
     - 목적: 공간적 차원을 축소하여 벡터화.
     - 구성: 특징 맵의 각 채널에 대해 평균을 계산하여 벡터로 변환.
    
  2. Self-Attention Layer
     - 목적: 각 위치의 중요도를 학습하여 특징 맵을 재정렬.
     - 구성: 특징 벡터를 입력으로 받아 자기 자신에 대한 가중치를 계산하여 중요한 특징을 강조.
     - 출력: 강화된 특징 벡터.
    
  3. Fully Connected Layer
     - 목적: 최종 분류.
     - 구성: 주의 분류기 출력 벡터를 입력으로 받아 소프트맥스 활성화 함수를 사용하여 각 클래스에 대한 확률을 출력.
     - 출력: 클래스 확률 분포.

 ### 모델 구조 순서

 - MobileNetV1 백본: 입력 이미지를 받아 초기 특징 맵 생성.
- 패치 추출 블록: MobileNetV1의 출력을 받아 로컬 패치 특징 맵 생성.
- Global Average Pooling: 패치 추출 블록의 출력을 받아 공간적 차원을 축소.
- Self-Attention Layer: GAP 출력을 받아 중요한 특징을 강조.
- Fully Connected Layer: 주의 분류기 출력을 받아 최종 분류 결과 출력.








### Architecture of the proposed PAtt-Lite.

![image](https://github.com/YeoungJun0508/PAtt-Lite/assets/145903037/9ae3e045-dcf6-48f7-809b-ae44be30bd6a)






# 모델 아키텍처

모델은 감정 인식을 위해 설계되었으며, MobileNet을 백본으로 사용. 모델의 각 레이어의 입력 및 출력 형태는 아래와 같음.

## 모델 요약

```plaintext
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 universal_input (InputLaye  [(None, 48, 48, 3)]          0         []                            
 r)                                                                                               
                                                                                                  
 resize (Resizing)           (None, 224, 224, 3)          0         ['universal_input[0][0]']     
                                                                                                  
 augmentation (Sequential)   (None, 224, 224, 3)          0         ['resize[0][0]']              
                                                                                                  
 tf.math.truediv_11 (TFOpLa  (None, 224, 224, 3)          0         ['augmentation[0][0]']        
 mbda)                                                                                            
                                                                                                  
 tf.math.subtract_11 (TFOpL  (None, 224, 224, 3)          0         ['tf.math.truediv_11[0][0]']  
 ambda)                                                                                           
                                                                                                  
 base_model (Functional)     (None, 14, 14, 512)          821952    ['tf.math.subtract_11[0][0]'] 
                                                                                                  
 patch_extraction (Sequenti  (None, 2, 2, 256)            272128    ['base_model[0][0]']          
 al)                                                                                              
                                                                                                  
 spatial_dropout2d_1 (Spati  (None, 2, 2, 256)            0         ['patch_extraction[0][0]']    
 alDropout2D)                                                                                     
                                                                                                  
 gap (GlobalAveragePooling2  (None, 256)                  0         ['spatial_dropout2d_1[0][0]'] 
 D)                                                                                               
                                                                                                  
 dropout_12 (Dropout)        (None, 256)                  0         ['gap[0][0]']                 
                                                                                                  
 pre_classification (Sequen  (None, 32)                   8352      ['dropout_12[0][0]']          
 tial)                                                                                            
                                                                                                  
 attention (Attention)       (None, 32)                   1         ['pre_classification[0][0]',  
                                                                     'pre_classification[0][0]']  
                                                                                                  
 dropout_13 (Dropout)        (None, 32)                   0         ['attention[0][0]']           
                                                                                                  
 classification_head (Dense  (None, 7)                    231       ['dropout_13[0][0]']          
 )                                                                                                
                                                                                                  
==================================================================================================
Total params: 1102664 (4.21 MB)
Trainable params: 280648 (1.07 MB)
Non-trainable params: 822016 (3.14 MB)
__________________________________________________________________________________________________


모델의 pre_classification 레이어는 256차원의 벡터를 32차원의 임베딩 벡터로 변환

모델의 최종 레이어인 classification_head는 주어진 입력 이미지에 대해 감정을 예측. 7개의 감정 클래스 중 하나로 분류.



