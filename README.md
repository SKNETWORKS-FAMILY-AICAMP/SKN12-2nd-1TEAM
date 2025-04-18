# 넷플릭스 고객 이탈 예측 시스템
## 팀 명 : Team Brainrot
## 👥 팀원소개

<p align="center">
<table width="100%">
  <tr align="center">
    <td width="25%">
      <a href="https://github.com/ypck">
        <img src="IMAGES/보네카 암발라부.webp" width="150" height="150"><br>
        <b>고남혁</b>
      </a>
    </td>
    <td width="25%">
      <a href="https://github.com/seunghak-kim">
        <img src="IMAGES/브루발로니 룰릴롤리.webp" width="150" height="150"><br>
        <b>김승학</b>
      </a>
    </td>
    <td width="25%">
      <a href="https://github.com/juyeong608">
        <img src="IMAGES/블루베리니 옥토푸시니.webp" width="150" height="150"><br>
        <b>이주영</b>
      </a>
    </td>
    <td width="25%">
      <a href="https://github.com/1203choi">
        <img src="IMAGES/퉁퉁퉁퉁퉁퉁퉁퉁퉁 사후르.webp" width="150" height="150"><br>
        <b>최요섭</b>
      </a>
    </td>
  </tr>
</table>
</p>
---

## 🖥️ 프로젝트

### 📅 개발 기간
- **2025.04.17 ~ 2025.04.18 (총 2일)**

### 🎬 프로젝트 주제
- 머신러닝을 활용한 넷플릭스 사용자 이탈 예측 시스템


## 📌 프로젝트 개요

### 📝 프로젝트 소개
넷플릭스 사용자 데이터를 기반으로 고객의 이탈 여부를 예측하는 머신러닝 모델을 구축하고, 이를 분석 및 시각화하여 웹 애플리케이션 형태로 구현한 프로젝트입니다.

### 🔍 프로젝트 필요성
- OTT 서비스 시장이 성장함에 따라 사용자 이탈 관리가 점점 더 중요해지고 있음.
- 고객 이탈은 수익 감소로 직결되기 때문에 사전 예측과 빠른 대응이 필수적임.
- 정확한 이탈 예측 모델은 효율적인 마케팅 전략 수립에 도움을 줌.
- 데이터 기반 분석은 기업의 의사결정을 체계적이고 객관적으로 만들어 줌.

### 🎯 프로젝트 목표
- 고객 이탈 문제를 데이터 기반으로 해결할 수 있는 실질적인 분석 역량 확보
- 다양한 모델을 적용하고 비교하며 최적의 이탈 예측 시스템 구현
- 분석 결과를 직관적으로 전달할 수 있는 웹 서비스 형태로 제공
- 실제 비즈니스 상황에 적용 가능한 인사이트 도출과 문제 해결 능력 강화

### 프로젝트 기대효과
- 넷플릭스 사용자의 이탈 예측 정확도를 높여 더 효율적인 사용자 유지 전략 개발
- 예측 모델을 통해 마케팅 및 고객 서비스 팀의 의사결정 지원
- 비즈니스 측면에서 이탈 위험이 높은 사용자를 선별하여 맞춤형 대응 가능
- 데이터 기반의 사용자 행동 패턴 분석으로 콘텐츠 전략 및 추천 시스템 개선
- Streamlit을 활용한 예측 결과 시각화로 실시간 모니터링 및 빠른 의사결정 지원

## 데이터 소개

본 프로젝트에서 사용한 데이터는 다음과 같습니다:

### Netflix 사용자 데이터
- **데이터셋:** netflix_users_final.csv
- **레코드 수:** 25,000명의 사용자 정보
- **포함 정보:**
  - 사용자 ID (User_ID) - 분석 과정에서 제외
  - 이름 (Name) - 분석 과정에서 제외
  - 나이 (Age)
  - 국가 (Country)
  - 구독 유형 (Subscription_Type)
  - 시청 시간 (Watch_Time_Hours)
  - 선호 장르 (Favorite_Genre)
  - 마지막 로그인 (Last_Login) - 일수 기준으로 변환하여 사용
  - 만족도 점수 (satisfaction_score)
  - 일일 시청 시간 (daily_watch_hours)
  - 주 사용 기기 (primary_device)
  - 월 소득 (monthly_income)
  - 프로모션 이용 횟수 (promo_offers_used)
  - 프로필 수 (profile_count)
  - 이탈 상태 (churn_status) - 목표 변수
  - 선호 시청 시간대 (preferred_watching_time)

---

## 기술 스택
- **언어**
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)

- **데이터 분석**
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)

- **머신러닝**
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-EC0000?logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-9ACD32?logo=lightgbm)

- **데이터 시각화**
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-00CED1?logo=seaborn)

- **데이터 균형화**
![SMOTE](https://img.shields.io/badge/SMOTE-Data%20Balancing-FF69B4)

- **모델 해석**
![SHAP](https://img.shields.io/badge/SHAP-Model%20Explainability-FF4500)

- **개발 환경**
![Google Colab](https://img.shields.io/badge/Google%20Colab-Cloud-F9AB00?logo=googlecolab)
![VS Code](https://img.shields.io/badge/VS%20Code-IDE-007ACC?logo=visualstudiocode)

- **비전 관리**
![Git](https://img.shields.io/badge/Git-Version--Control-F05032?logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)

---

## 분석 방법론
1. **데이터 전처리**
   - 불필요한 컬럼(User_ID, Name) 제거
   - 음수 월소득 데이터 제거
   - 마지막 로그인(Last_Login) 날짜를 현재 기준 경과 일수로 변환
   - 30일 이상 로그인하지 않은 사용자 이탈 식별(month_churn)
   - 범주형 변수 Label Encoding

2. **데이터 불균형 처리**
   - SMOTE를 활용한 소수 클래스 오버샘플링
   - 학습 데이터와 테스트 데이터 분리(80:20)

3. **특성 표준화**
   - StandardScaler를 통한 수치형 데이터 정규화

4. **모델링 및 하이퍼파라미터 튜닝**
   - 로지스틱 회귀(LogisticRegression)
     - C, penalty, solver 파라미터 튜닝
   - 랜덤 포레스트(RandomForestClassifier)
     - n_estimators, max_depth, min_samples_split, max_features 파라미터 튜닝
   - XGBoost(XGBClassifier)
     - n_estimators, learning_rate, max_depth, subsample, colsample_bytree 파라미터 튜닝
   - LightGBM(LGBMClassifier)
     - num_leaves, learning_rate, n_estimators, feature_fraction 파라미터 튜닝
   - GridSearchCV를 통한 최적 파라미터 탐색

5. **모델 평가**
   - 정확도(Accuracy)
   - F1 점수(F1 Score)
   - 분류 보고서(Classification Report)
   - 혼동 행렬(Confusion Matrix)
   - ROC 곡선 및 AUC
   
6. **모델 해석**
   - 특성 중요도(Feature Importance) 분석
   - SHAP 값을 통한 모델 해석
   - 시각화를 통한 결과 비교 분석

---

## 주요 발견 사항
- 사용자의 마지막 로그인 후 경과 일수(Last_Login_days)가 이탈 예측에 가장 중요한 지표로 확인되었습니다.
- RandomForest 모델이 F1 점수 0.82로 가장 높은 성능을 보였으며, 다른 모델들도 안정적인 성능을 보였습니다.
- 범주형 변수들(국가, 선호 장르, 선호 시청 시간대 등)을 Label Encoding하여 모델 성능을 향상시켰습니다.
- SMOTE를 통한 데이터 불균형 해소가 모델의 예측 성능 향상에 기여했습니다.
- 트리 기반 모델(RandomForest, XGBoost, LightGBM)이 로지스틱 회귀보다 약 5-8% 더 좋은 성능을 보였습니다.
- 30일 이상 로그인하지 않은 사용자를 이탈 고객으로 정의한 month_churn 특성이 모델 학습에 유용한 정보를 제공했습니다.

---

## 프로젝트 구조
```
netflix-churn-prediction/
│
├── data/
│   └── netflix_users.csv    # 최종 분석에 사용된 데이터셋
│
├── notebooks/
│   └── netflix_churn_prediction.ipynb    # 전체 분석 과정이 담긴 노트북
│
├── models/
│   ├── LogisticRegression_best_model.pkl  # 로지스틱 회귀 모델 (F1: 0.77)
│   ├── RandomForest_best_model.pkl        # 랜덤 포레스트 모델 (F1: 0.82)
│   ├── XGBoost_best_model.pkl             # XGBoost 모델 (F1: 0.80)
│   └── LightGBM_best_model.pkl            # LightGBM 모델 (F1: 0.80)
│
├── results/
│   └── model_results.csv                  # 모델 성능 비교 결과
│
├── visualizations/
│   ├── confusion_matrices/                # 각 모델의 혼동 행렬
│   ├── roc_curves/                        # 각 모델의 ROC 곡선
│   ├── feature_importance/                # 각 모델의 특성 중요도
│   └── shap_values/                       # SHAP 값 시각화
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 결과 및 성능

모델 성능 비교 결과는 다음과 같습니다(model_results.csv 기준):

| 모델 | 정확도(Accuracy) | F1 점수(F1 Score) | 최적 파라미터 |
|------|-----------------|-----------------|--------------|
| RandomForest | 0.8201 | 0.8218 | max_depth=None, max_features='sqrt', min_samples_split=2, n_estimators=100 |
| LightGBM | 0.8018 | 0.8036 | feature_fraction=0.8, learning_rate=0.1, n_estimators=100, num_leaves=31 |
| XGBoost | 0.7950 | 0.7986 | colsample_bytree=0.8, learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8 |
| LogisticRegression | 0.7670 | 0.7700 | C=1, penalty='l2', solver='liblinear' |

※ 정확한 수치는 model_results.csv 파일에서 확인 가능합니다.

**성능 결과 분석:**
- RandomForest 모델이 정확도 82.01%, F1 점수 82.18%로 가장 높은 성능을 보였습니다.
- 트리 기반 모델(RandomForest, LightGBM, XGBoost)이 로지스틱 회귀보다 우수한 성능을 보였습니다.
- GridSearchCV를 통한 하이퍼파라미터 최적화가 모델 성능 향상에 크게 기여했습니다.
- 모든 모델이 77% 이상의 정확도를 보이며 안정적인 성능을 나타냈습니다.

**중요 특성 (랭킹 기준):**
1. Last_Login_days (마지막 로그인 이후 경과 일수)
2. Watch_Time_Hours (시청 시간)
3. satisfaction_score (만족도 점수)
4. daily_watch_hours (일일 시청 시간)
5. Age (나이)

**각 모델별 특징:**
- **RandomForest**: 앙상블 학습을 통한 안정적인 성능과 높은 해석 가능성을 제공하여 본 프로젝트에서 최고 성능을 기록
- **LightGBM**: 리프 중심 트리 분할 방식으로 빠른 학습 속도와 메모리 효율성 제공
- **XGBoost**: 경사 부스팅 기반으로 높은 예측 정확도를 보이며, 과적합에 강한 특성을 보임
- **LogisticRegression**: 모델 구조가 단순하여 해석이 용이하나, 복잡한 패턴 학습에는 한계 존재

---


---

## 향후 개선 방향
- **특성 공학 강화**
  - 시청 패턴에 따른 추가 파생 변수 생성
  - 시계열 분석을 통한 이탈 징후 조기 발견 기능 구현
  - 범주형 변수의 One-Hot Encoding 적용 및 효과 측정

- **모델 고도화**
  - 신경망 기반 모델(Deep Learning) 적용 및 성능 비교
  - 더 많은 하이퍼파라미터 조합 테스트를 통한 모델 성능 향상
  - 모델 앙상블 기법을 통한 예측 정확도 개선

- **서비스 확장**
  - 고객 세그먼트별 맞춤형 이탈 예측 모델 개발
  - 이탈 방지를 위한 추천 시스템과의 연동
  - 지역별, 연령별 이탈 패턴 분석 및 타겟 마케팅 전략 제안

- **실시간 예측 시스템 구축**
  - 실시간 사용자 행동 데이터 수집 및 분석 파이프라인 구축
  - 사용자 행동 변화에 따른 이탈 확률 모니터링 시스템 개발

---



---





## 💭 한줄 회고 
- 고남혁 :
- 김승학 :
- 이주영 : 
- 최요섭 : 



