# <느린심 장박동> 모델소개 및 실행가이드
개요) <p>
심전도 리드별 데이터를 따로 훈련시켜 스태킹 모델 구축 
  
## 선택 주제
  주제2.심전도 데이터셋을 활용한 부정맥 진단 AI 모델 공모(심전도 데이터셋을 활용한 부정맥 진단 AI 모델 개발)

  
## 데이터 전처리
  - Decode
    - 64진법으로 encode된 raw data의 waveformdata를 다시 숫자로 decode
      참고링크: [ecgdata_decoder](https://github.com/hewittwill/ECGXMLReader)
    - Decode할 때 리드별로 정리하여 각 normal/arrhythmia마다 정리
    - 즉, lead1별로 normal, arrhythmia가 train, valid별로 정리됨 
      - (lead1,normal,train), ..., (lead12,normal,train) / (lead1,arrhythmia,train), ..., (lead12,arrhythmia,train) 
      - (lead1,normal,valid), ..., (lead12,normal,valid) / (lead1,arrhythmia,train), ..., (lead12,arrhythmia,valid) 
      - 총 48개 csv 파일 형성
      - 이름형식: (lead1, normal, train) = train_normal_lead_df0.csv
  - 리드 조정
    - 실제 데이터에는 리드가 8개인 것과 12개인 것이 혼재되어 있었음
    - 8개인 데이터의 경우 결여된 4개의 리드에 대해서는, 위의 decoder에서 참고한 식 활용
     ``` python
      leads['III'] = np.subtract(leads['II'], leads['I'])
      leads['aVR'] = np.add(leads['I'], leads['II'])*(-0.5)
      leads['aVL'] = np.subtract(leads['I'], 0.5*leads['II'])
      leads['aVF'] = np.subtract(leads['II'], 0.5*leads['I']
    ``` 
  - 리드별로 데이터를 나눈 이유
    - 환자별 각 리드에 해당하는 wavedata가 약 600~5000개이기 때문에 한 환자에서 나온 총 데이터는 (리드수)x(리드별데이터수) = 12x5000 = 60000개가 된다. 
    - 이를 모두 사용하기 보다는 심전도의 리드별로 특징이 다른 것에 착안하여, 리드별로 데이터를 구분, 그에따라서 모델도 다르게 구성하기로 하였다. 
  - 다만 모델을 리드별로 12개를 만든 후, 이 12개에 따라 나온 결과값을 다시 train_set으로 받아 최종모델에 넣는 스태킹방식을 취하기로 하였다. 
 - 이상치 처리
    - 원소가 모두 0인 데이터는 제외하여 따로 저장

## 모델 
  ![image](https://user-images.githubusercontent.com/68943859/145714130-3a6e8ffa-f4de-4240-bda2-cffe9aacfbd0.png)
  - 이미 '전처리가 완료된' train set, validation set 모두 활용한다 
    - 각 리드별로 train set(normal, arrhythmia)을 적재한 후, label을 normal=0, arrhythmia=1로 둔 후 합쳐서 train_set을 만든다
    - 이후 train_set을 shuffle하여 무작위로 섞어 label(y_train)와 특성값(x_train)을 분리한다
      
  - 리드별 모델
    - 모델로는 CNN모델을 사용한다.
    - 이런 식으로 리드12개 각각에 대한 모델을 총 12개 만든다. 
    ``` python
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(4,3,activation='relu',input_shape=(x_train.shape[1],1)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10,activation='relu'),
            tf.keras.layers.Dense(10,activation='relu'),
          tf.keras.layers.Dense(1,activation='sigmoid')])
      ``` 
    **단, 이때, 모델별로 x_test_t(validation set)의 예측치를 기록해두어 최종모델의 validation 데이터로 쓰도록 한다.**
    ``` python
      new_train_set[leadid] = model.predict(x_train_t).flatten() #최종모델을 위한 train
      new_valid_set[leadid] = model.predict(x_test_t).flatten() #최종모델에 쓰일 validation
    ``` 
  - 최종모델
    - 각 리드별 모델에서 나온 결과값(sigmoid함수의 결과값, 확률)을 모아 또 하나의 new_train_set을 형성한다. 
    - new_train_set은 즉, 예를 들어 환자ID 1번의 리드 12개를 각 12개모델에 돌려서 나온 결과값 12개를 row별로 쌓은 형식이며, 그에 따른 label을 원래 y_train으로 둔다. 
    - new_train_set은 LGBM분류기를 사용하여 최종 train 시킨다. 
  
  - 성능
    - new_valid_set을 활용하여 최종모델의 성능을 파악한다. 
  
    ![image](https://user-images.githubusercontent.com/68943859/145713645-f21d6053-94b0-49e0-a636-de9cc61e6186.png)

   
## 참고자료
- Xiuzhu Yang, Xinyue Zhang, Mengyao Yang, Lin Zhang,
12-Lead ECG arrhythmia classification using cascaded convolutional neural network and expert feature, Journal of Electrocardiology, Volume 67, 2021, Pages 56-62,
[link](https://doi.org/10.1016/j.jelectrocard.2021.04.016)
- Junsang Park, Junho An, Jinkook Kim, Sunghoon Jung, Yeongjoon Gil, Yoojin Jang, Kwanglo Lee, Il-young Oh, Study on the use of standard 12-lead ECG data for rhythm-type ECG classification problems, Computer Methods and Programs in Biomedicine, 2021
[link](https://doi.org/10.1016/j.cmpb.2021.106521)
