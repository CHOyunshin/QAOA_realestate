import pandas as pd
import numpy as np
import SA_basefunctions as bf
from sklearn.linear_model import LinearRegression

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

    # 추가한 budget constraint in base function 
    # obj = bf.get_QB_constraint 

    for j in schedule:
      tau = initial_t / (1 + alpha * j)
      for m in range(j):
        theta_star = bf.flip(k_flip, np.where(theta_zero)[0], p)
        if np.random.rand(1) <= min(1, np.exp((obj(self.x.iloc[:, np.where(theta_zero)[0]], p, Q = self.Q, beta = self.Beta, lmbd = self.Lambda)-obj(self.x.iloc[:, np.where(theta_star)[0]], p, Q = self.Q, beta = self.Beta, lmbd = self.Lambda))/tau)):
          theta_zero = theta_star 

    result = self.x.iloc[:, np.where(theta_zero)[0]]

    return result