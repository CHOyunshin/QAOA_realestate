import pandas as pd
import numpy as np
import SA_basefunctions as bf
from pandas.core.aggregation import relabel_result
from sklearn.linear_model import LinearRegression

class SimulatedAnnealing:
  def __init__(self, x, y,
                schedule_list,
                k_flip,
                alpha,
                tau,
                Q,
                Beta,
                Lambda,
                constraint,
                condition = []
    ):
    self.x, self.y = x, y
    self.schedule_list, self.k_flip, self.alpha, self.tau = schedule_list, k_flip, alpha, tau
    self.Q = Q
    self.Beta = Beta
    self.Lambda = Lambda
    self.constraint = constraint
    self.condition = condition

  def equal(self):
    schedule_list, k_flip, alpha, tau, constraint, condition = self.schedule_list, self.k_flip, self.alpha, self.tau, self.constraint, self.condition
    if k_flip % 2 != 0:
      raise Exception('k는 2의 배수여야 합니다.')
    if len(condition) > 0 and (k_flip/2 > (constraint - len(condition))): 
      raise Exception('k는 constraint 개수와 condition 개수 차의 2배 이하여야 합니다. k를 줄이거나 constraint를 늘려보세요')
    
    # 0.8 <= alpha <= 0.99
    schedule = schedule_list
    p = len(self.x.columns)
    theta_zero = bf.get_bin(condition, p)
    idx = np.argwhere(theta_zero==0).flatten().tolist()
    theta_zero[np.random.choice(idx, size = constraint-len(condition), replace = False)] = 1
    obj = bf.get_QB

    for j in schedule:
      for m in range(j):
        theta_star = bf.flip2(k_flip, np.where(theta_zero)[0].tolist(), idx, condition, p)
        if np.random.rand(1) <= min(1, np.exp((obj(self.x.iloc[:, np.where(theta_zero)[0]], p, Q = self.Q, beta = self.Beta, lmbd = self.Lambda)-obj(self.x.iloc[:, np.where(theta_star)[0]], p, Q = self.Q, beta = self.Beta, lmbd = self.Lambda))/tau)):
          theta_zero = theta_star
        tau = alpha * tau

    result = self.x.iloc[:, np.where(theta_zero)[0]]

    return result

  def unequal(self):
    schedule_list, k_flip, alpha, tau, constraint, condition = self.schedule_list, self.k_flip, self.alpha, self.tau, self.constraint, self.condition

    # 0.8 <= alpha <= 0.99
    schedule = schedule_list
    p = len(self.x.columns)
    theta_zero = bf.get_bin(condition, p)
    idx = np.argwhere(theta_zero==0).flatten().tolist()
    theta_zero[np.random.choice(idx, size = constraint-len(condition), replace = False)] = 1
    theta_first = theta_zero
    obj = bf.get_QB

    for j in schedule:
      for m in range(j):
        theta_star = bf.flip4(k_flip, np.where(theta_zero)[0].tolist(), idx, condition, p, constraint)
        if sum(theta_star) <= constraint:
          if np.random.rand(1) <= min(1, np.exp((obj(self.x.iloc[:, np.where(theta_zero)[0]], p, Q = self.Q, beta = self.Beta, lmbd = self.Lambda)-obj(self.x.iloc[:, np.where(theta_star)[0]], p, Q = self.Q, beta = self.Beta, lmbd = self.Lambda))/tau)):
            theta_zero = theta_star  
        tau = alpha * tau   

    result = self.x.iloc[:, np.where(theta_zero)[0]]

    return result

