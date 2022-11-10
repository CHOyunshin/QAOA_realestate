import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

def get_aic(x, y):
    '''
        aic를 구하는 함수
        input: data x, data y
        output: aic 값
    '''
    reg = LinearRegression().fit(x, y)
    prediction = reg.predict(x)
    residual = y - prediction

    N = len(x)
    s = len(x.columns)
    AIC = N*np.log(sum(residual**2)/N) + 2*(s + 2)

    return AIC

def get_bic(x, y):
    '''
        bic를 구하는 함수
        input: data x, data y
        output: bic 값
    '''
    reg = LinearRegression().fit(x, y)
    prediction = reg.predict(x)
    residual = y - prediction

    N = len(x)
    s = len(x.columns)
    BIC = N*np.log(sum(residual**2)/N) + np.log(N)*(s + 2)

    return BIC

def get_mspe(x, y, test_x, test_y):
    '''
        mspe를 구하는 함수
        input: data x, data y
        output: mspe 값
    '''
    reg = LinearRegression().fit(x, y)
    prediction = reg.predict(test_x)
    MSPE = sum((test_y - prediction)**2) / len(test_y)

    return MSPE

def get_bin(x, p):
    '''
        선택된 변수의 정수 index를 [01100110..] 방식으로 변환해주는 함수
        input: 정수 index array, 총 변수 개수 p
        output: binary 방식 변수 선택 결과
    '''
    zeros = np.zeros(p, dtype=int)
    zeros[x] = 1

    return zeros

def flip(k, x, p):
    '''
        기존 선택된 변수들에서 k개만큼 flip해주는 함수
        input: flip할 횟수 k, 정수 index array, 총 변수 개수 p
        output: 새롭게 선택된 변수 결과
    '''
    zeros = np.zeros(p, dtype=int)
    idx = np.random.choice(p, size = k, replace = False)
    zeros[idx] = 1

    old = get_bin(x, p)
    new = abs(old - zeros)

    return new

def get_cn(x):
    cn = np.sqrt(max(np.linalg.svd(x)[1])/min(np.linalg.svd(x)[1]))

    return cn

def get_vif(x):
  vif_df = pd.DataFrame()
  vif_df['variables'] = x.columns

  vif_df['vif'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

  return vif_df


def get_QB(x, p, Q, beta, lmbd):
  x_ = np.zeros(p, dtype = int)
  x_[x.columns] = 1

  value = lmbd * x_.T @ Q @ x_ + (1 - lmbd) * beta.T @ x_

  return -1*value





