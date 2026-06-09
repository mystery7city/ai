# 🌍 World Happiness Report Analysis

세계 행복 지수 데이터를 분석하고 머신러닝으로 행복 점수를 예측하는 프로젝트입니다.

## 📋 프로젝트 구조

```
world-happiness-analysis/
├── world_happiness_analysis.ipynb   # 메인 분석 노트북
├── README.md                        # 프로젝트 설명
└── requirements.txt                 # 의존 패키지 목록
```

## 🔍 분석 내용

1. **데이터 전처리** — 결측값 처리, 기초 통계량 확인
2. **탐색적 데이터 분석 (EDA)** — 분포, 지역별 비교, 상위/하위 국가 시각화
3. **상관관계 분석** — 히트맵, 산점도 및 추세선
4. **머신러닝 모델링** — 5가지 회귀 모델 비교 및 Feature Importance 분석

## 🛠 사용 기술

- **분석**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **모델링**: scikit-learn (Linear, Ridge, Lasso, RandomForest, GradientBoosting)

## 🚀 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 노트북 실행
jupyter notebook world_happiness_analysis.ipynb
```

## 📊 데이터 출처

[World Happiness Report — Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness)

> 노트북 내에서는 재현 가능한 샘플 데이터를 자동 생성합니다.  
> Kaggle에서 실제 데이터를 다운로드하여 교체하면 더욱 정확한 분석이 가능합니다.

## 📌 주요 발견

| 요인 | 행복과의 상관관계 |
|------|-----------------|
| GDP per Capita | ⬆️ 강한 양(+) |
| Social Support | ⬆️ 강한 양(+) |
| Healthy Life Expectancy | ⬆️ 강한 양(+) |
| Freedom | ↗️ 중간 양(+) |
| Corruption Perception | ⬇️ 음(-) |
| Generosity | ↔️ 약한 상관 |
