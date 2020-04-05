# M5 competition
- accuracy: https://www.kaggle.com/c/m5-forecasting-accuracy
- uncertainty: https://www.kaggle.com/c/m5-forecasting-uncertainty/overview



## Source Code

아래와 같은 과정을 통한 파이프라인을 구축할 예정입니다.

- install

  ```
  pip install -r requirements.txt
  pre-commit install
  ```

  

- train

- inference


## Branch rule

**master branch에 push는 원칙적으로 금합니다.**



1. issue 작성

   예시) [dataloader] augmentation 기능 추가

2. 해당 issue 번호로 branch 생성 후 해당 내용 작업

   예시) feature/#1

3. 해당 branch로 push 후 master branch에 PR날리기

4. code review 및 merge



## 위키 작성

위키 작성은 대회를 하면서 아래와 같은 정보를 효과적으로 관리하기 위함입니다.



- **EDA**

  - 해당 데이터 중 유의미한 feature는 무엇인가?
  - 서로 상관관계가 있는 feature는 무엇인가?
  - etc

- **실험설계**

  - dataframe을 어떻게 구성할 것인지?
  - dataframe을 어떻게 관리할 것인지?
  - 어떤 하이퍼파라미터를 조절 할 것인지?
  - 어떤 모델을 실험할 것인지?

- **실험결과**

  - 위의 실험설계에 대한 결과를 정리합니다.

- **submission 결과**

  - kaggle competition에서 채점에 사용되는 데이터와의 분포가 다를 수 있으므로, 수시로 check해줍니다.

    





## 칸반보드

github project는 칸반보드를 제공합니다.  칸반보드는 각 구성원이 맡은 task를 효과적으로 관리하기 위함입니다.





**각 구성원의 책임감있는 모습을 기대하며, 모두 화이팅입니다. :)**

