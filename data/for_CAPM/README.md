# 데이터 설명



## raw data

all_period : 10년간 서울시 아파트 거래 데이터 (raw data)





## for multivariate CAPM



CAPM은 서울시 내 113개 동을 대상으로 진행할 예정

- rawdata_113dong: 113개 동의 10년간 서울시 아파트 거래 데이터
- avgprice_month_113dong: 113개 동의 월별 동별 평균가격

![multivariate_CAPM](https://user-images.githubusercontent.com/109649288/204982507-a300c0d9-0920-4f70-9924-95f724600529.PNG)

- R_i : 서울시 113개 동의 월별 동별 전월대비 평균가격 변화율

- R_m : 서울시 전체 거래에 대한 전월대비 평균가격변화율

- R_f: 무위험 만기 이자수익률(국고채 3년)

- SMB: 큰 규모의 아파트의 평균가격변화율 - 중간 규모 아파트의 평균가격 변화율
