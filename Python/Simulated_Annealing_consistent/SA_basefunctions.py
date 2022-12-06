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

def flip2(k, x, ind, cond, p):
    '''
    k: 몇 개 뒤집을 것인지, 2의 배수여야 함
    x: 기존 1이었던 정수 index
    ind: 뒤집어도 되는(반드시 뽑아야 하는 것들을 제외한) 정수 index
    cond: 반드시 뽑아햐 하는 정수 index
    p: 총 변수 개수
    '''
    # 뒤집을 수 있는 것 중 기존에 1이었던 것들
    one = np.where(get_bin(x, p) * get_bin(ind, p))[0].tolist()
    # 뒤집을 수 있는 것 중 기존에 0이었던 것들
    zero = np.where((get_bin(ind, p) - get_bin(x, p))==1)[0].tolist()
    idx_onetozero = np.random.choice(one, size = int(k/2), replace = False)
    idx_zerotoone = np.random.choice(zero, size = int(k/2), replace = False)

    new = get_bin(x, p)
    new[idx_onetozero] = 0
    new[idx_zerotoone] = 1

    return new

def flip3(k, x, ind, cond, p):
    '''
    x: 기존 1이었던 정수 index
    ind: 뒤집어도 되는(반드시 뽑아야 하는 것들을 제외한) 정수 index
    cond: 반드시 뽑아햐 하는 정수 index
    p: 총 변수 개수
    '''
    zeros = np.zeros(p, dtype=int)
    idx = np.random.choice(ind, size = k, replace = False)
    zeros[idx] = 1

    old = get_bin(x, p)
    new = abs(old - zeros)

    return new

def flip4(k, x, ind, cond, p, const):
    '''
    x: 기존 1이었던 정수 index
    ind: 뒤집어도 되는(반드시 뽑아야 하는 것들을 제외한) 정수 index
    cond: 반드시 뽑아야 하는 정수 index
    p: 총 변수 개수
    '''
    # 뒤집을 수 있는 것 중 기존에 1이었던 것들
    one = np.where(get_bin(x, p) * get_bin(ind, p))[0].tolist()
    # 뒤집을 수 있는 것 중 기존에 0이었던 것들
    zero = np.where((get_bin(ind, p) - get_bin(x, p))==1)[0].tolist()
    
    condition = True

    while condition:
      num_onetozero = min(len(one), np.random.randint(k+1))
      num_zerotoone = min(len(zero), np.random.randint(k-num_onetozero+1))
      if (num_onetozero + num_zerotoone) == 0:
        condition = True
      else:
         condition = (len(x)-num_onetozero+num_zerotoone) > const

    idx_onetozero = np.random.choice(one, size = num_onetozero, replace = False)
    idx_zerotoone = np.random.choice(zero, size = num_zerotoone, replace = False)
    
    new = get_bin(x, p)
    if idx_onetozero.size == 0:
      pass
    else:
      new[idx_onetozero] = 0
    if idx_zerotoone.size == 0:
      pass
    else:
      new[idx_zerotoone] = 1

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
  '''
  x: input data
  p: 변수의 개수
  Q: Q(보통 data의 correlation)
  beta: mu/beta, CAPM 결과
  lmbd: lambda, Q와 beta의 가중치 조절
  '''
  x_ = np.zeros(p, dtype = int)
  x_[x.columns] = 1

  value = lmbd * x_.T @ Q @ x_ - (1 - lmbd) * beta.T @ x_

  return value





