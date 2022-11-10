import pandas as pd
import numpy as np
import SA_basefunctions as bf
from sklearn.linear_model import LinearRegression

'''
  simulated annealing 구현을 위한 class입니다.
  구체적인 작동 알고리즘별로 geometric/lundymee 방식으로 구분되어 있습니다.
  a = SimulatedAnnealing(x, y): data x, y를 입력합니다.

  a.geometric(schedule_list, k_flip, alpha, tau, objective)
  a.lundymee(schedule_list, k_flip, alpha, tau, objective)

  schedule_list: stage별 반복 횟수가 적혀진 list ex: [10, 10, 20, 20]
  k_flip: 변수 선택 결과를 몇 개씩 뒤집어가며 찾을 것인지 ex: 2
  alpha: 일반적으로 0과 1 사이의 float ex: 0.83
  tau: 초기 온도 ex: 1

  objective: 'aic'/'bic'/'mspe' 세 가지 목적함수 중 하나를 선택하여 입력

  -> 변경점, 사용하는 instance 4개에서 objective function의 formulation에 필요한 
    Q, Beta, Lambda가 추가되었다. 
'''

class SimulatedAnnealing:
  def __init__(self, x, y,
                schedule_list,
                k_flip,
                alpha,
                tau,
                Q,
                Beta,
                Lambda
    ):
    self.x, self.y = x, y
    self.schedule_list, self.k_flip, self.alpha, self.tau = schedule_list, k_flip, alpha, tau
    self.Q = Q
    self.Beta = Beta
    self.Lambda = Lambda

  def geometric(self):
    schedule_list, k_flip, alpha, tau = self.schedule_list, self.k_flip, self.alpha, self.tau
    # 0.8 <= alpha <= 0.99
    schedule = schedule_list
    p = len(self.x.columns)
    theta_zero = np.random.randint(2, size = p)
    obj = bf.get_QB

    for j in schedule:
      for m in range(j):
        theta_star = bf.flip(k_flip, np.where(theta_zero)[0], p)
        if np.random.rand(1) <= min(1, np.exp((obj(self.x.iloc[:, np.where(theta_zero)[0]], p, Q = self.Q, beta = self.Beta, lmbd = self.Lambda)-obj(self.x.iloc[:, np.where(theta_star)[0]], p, Q = self.Q, beta = self.Beta, lmbd = self.Lambda))/tau)):
          theta_zero = theta_star 
      tau = alpha * tau

    result = self.x.iloc[:, np.where(theta_zero)[0]]

    return result
  
  def lundymee(self):
    schedule_list, k_flip, alpha, tau = self.schedule_list, self.k_flip, self.alpha, self.tau
    # alpha > 0
    schedule = schedule_list
    initial_t = tau
    p = len(self.x.columns)
    theta_zero = np.random.randint(2, size = p)
    obj = bf.get_QB

    for j in schedule:
      tau = initial_t / (1 + alpha * j)
      for m in range(j):
        theta_star = bf.flip(k_flip, np.where(theta_zero)[0], p)
        if np.random.rand(1) <= min(1, np.exp((obj(self.x.iloc[:, np.where(theta_zero)[0]], p, Q = self.Q, beta = self.Beta, lmbd = self.Lambda)-obj(self.x.iloc[:, np.where(theta_star)[0]], p, Q = self.Q, beta = self.Beta, lmbd = self.Lambda))/tau)):
          theta_zero = theta_star 

    result = self.x.iloc[:, np.where(theta_zero)[0]]

    return result







