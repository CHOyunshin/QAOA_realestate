# import 
import os
from kqc_custom import generate_dependent_sample
from SA_ver2 import SimulatedAnnealing as SA
import matplotlib.pyplot as plt
from SA_basefunctions import get_QB as qb
import numpy as np
import pandas as pd

import plotly.express as px
import chart_studio.plotly as py

# csv data loader
'''
    data
    data_price
    data_price_change

    data_seoul
    data_risk
'''
data =  pd.read_csv(('00_application/source/csv/전체데이터(113개동).csv'),index_col=0)
data_price = pd.read_csv(('00_application/source/csv/월별동별평균가격(113개동).csv'),index_col=0)
data_price_change = pd.read_csv(('00_application/source/csv/월별동별평균가격변화율(113개동).csv'),index_col=0)
data_seoul = pd.read_csv(('00_application/source/csv/서울시전월대비가격변화율.csv'),index_col=0)
data_risk = pd.read_csv(('00_application/source/csv/무위험만기이자수익률(국고채 3년).csv'),index_col=0)

# ?
data = data_price_change
# test train set 
'''
    일단은 7 : 3 을 default로 
'''
training_data = data_price_change[:84]
test_data = data_price_change[84:]

# CAPM #1
'''
    regression 으로 Beta 들을 구해준다. 
    이때 전체 베타 train beta test beta 를 구해준다. 

    beta, beta_training, beta_test
''' 
beta = []
for i in range(113) :
    beta.append( np.polyfit( data_seoul["거래금액(만원)"]-data_risk["국고채(3년,평균)"],data[data.columns[i]]-data_risk["국고채(3년,평균)"].values  ,1 )[0] )

# training 베타
beta_training = []
for i in range(113) :
    beta_training.append( np.polyfit( data_seoul["거래금액(만원)"][:84]-data_risk["국고채(3년,평균)"][:84],data[data.columns[i]][:84]-data_risk["국고채(3년,평균)"][:84].values  ,1 )[0] )

# test 베타
beta_test = []
for i in range(113) :
    beta_test.append( np.polyfit( data_seoul["거래금액(만원)"][84:]-data_risk["국고채(3년,평균)"][84:],data[data.columns[i]][84:]-data_risk["국고채(3년,평균)"][84:].values  ,1 )[0] )    

# ERS
'''
    risk free rate 를 test 기간동안 설정해주어 
    각 동의 Market_risk_premium의 list 에 저장해 준다.  -> 113 by 1

    이를 이용해서 Expected Return on a security 
'''
# risk_free_rate = np.mean( data_risk[84:] )
Market_risk_premium = []
for i in range(113):
    Market_risk_premium.append ( np.mean(training_data)[i] - np.mean( data_risk[84:] ) )
# Expected Return on a Security
ERS = np.array(beta_training)*np.array( Market_risk_premium ).T + np.array( Market_risk_premium ).T
ERS = ERS.T                 ## for plot 

# Plot the ERS 
'''
    113개 동에 대한 이상치를 확인하기 위해서 다음과 같이
'''
fig = px.line(ERS)
py.iplot(fig)


## out put 정리 
'''
    SA -> Beta_training 이용 (array로 변환해주어야 됨)
    X  -> training_data 
    Q  -> 
'''
#--------------------------------------------------------------------------------------------------------#
#                                               SA
#--------------------------------------------------------------------------------------------------------#
# SA 를 위한 setting 
beta_training = np.array(beta_training)
Q=training_data.corr()

lst = [60, 60, 60, 60, 60, 120, 120, 120, 120, 120, 220, 220, 220, 220, 220]

#lst = [30, 50, 100]
k=2
alpha=0.9
tau=1
ld = 0.5

X  = training_data
X.columns = range(0,113)
y=data_seoul[:84]       

## SA 실험 진행
simulatedannealing = SA(X, y, lst, k, alpha, tau, Q = Q, Beta = beta_training, Lambda = ld, constraint = 10,condition = [0,1,2,3,4])

#simulatedannealing = SA(X, data_seoul[:84], lst, k, alpha, tau, Q = Q, Beta = beta_training, Lambda = ld, constraint = 10,condition = [0, 1])
sa_result = simulatedannealing.equal()

selected=test_data[data_price_change.columns[sa_result.columns]]

# BOX plot 은 SA 를 10 회 반복해서 선택된 부분에 대한 부분 -> 