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








