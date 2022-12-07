import sys

module_path = "/Users/minhyeong-gyu/Documents/GitHub/QAOA_realestate/Python"
if module_path not in sys.path:
    sys.path.append(module_path)

from dimod import ConstrainedQuadraticModel, Integer, QuadraticModel
import pandas as pd
import numpy as np
from optimizer import basefunctions as bf

class DWAVE_optimizer:
    def __init__(self,sampler):
        self.sampler = sampler

    def optimize(self,
                Q,
                beta,
                lamda=0.5,
                k = None):
        
        p = len(Q)
        integer_list = []
        for i in range(p) :
            integer_list += [Integer(str("x")+str(i).zfill(3), upper_bound=1,lower_bound=0)]
        linear_qubo = QuadraticModel()
        for i in range(p): 
            linear_qubo += (1-lamda)*beta[i]*integer_list[i]    
        quadratic_qubo = QuadraticModel()
        for j in range(p):
            for i in range(p):
                quadratic_qubo += lamda*Q[i][j]*integer_list[i]*integer_list[j]
        
        Qubo = linear_qubo + quadratic_qubo
        cqm = ConstrainedQuadraticModel()
        cqm.set_objective(Qubo)
        
        if type(k) == int:
            n_asset = QuadraticModel()
            for i in range(p):
                n_asset += integer_list[i]
            cqm.add_constraint(n_asset==k)
        
        sampleset = self.sampler.sample_cqm(cqm)
        sampleset = sampleset.to_pandas_dataframe()
        sampleset_true = sampleset[sampleset["is_feasible"]==True]
        self.result = sampleset_true.loc[sampleset_true["energy"] == np.min(sampleset_true["energy"]),:].iloc[0,:p].tolist()
        self.sampleset = sampleset
        return self.result

class SimulatedAnnealing:
  def __init__(self,
               schedule_list = [100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300, 300, 400, 400, 400, 400, 400, 400],
               k_flip=2,
               alpha=0.9,
               tau=1,
               ):
    self.schedule_list, self.k_flip, self.alpha, self.tau = schedule_list, k_flip, alpha, tau

  def optimize(self,
               Q,
               beta,
               lamda=0.5,
               k = None,
               ):
    schedule_list, k_flip, alpha, tau = self.schedule_list, self.k_flip, self.alpha, self.tau
    if k_flip % 2 != 0:
      raise Exception('k는 2의 배수여야 합니다.')
    
    theta_list = []
    p = len(Q)
    obj = bf.get_QB

    if k == None :
        theta_temp = np.random.randint(2,size=p)
        for j in schedule_list:
            for m in range(j):
                theta_star = bf.flip(k_flip, theta_temp, p)
                if np.random.rand(1) <= min(1, np.exp((obj(theta_temp, Q = Q, beta = beta, lmbd = lamda)-obj(theta_temp, Q = Q, beta = beta, lmbd = lamda))/tau)):
                    theta_temp = theta_star
                theta_list += [theta_temp]
                tau = alpha * tau


    elif type(k) == int :    
        theta_temp = bf.get_bin(np.random.choice(p,k,replace=False),p)
        for j in schedule_list:
            for m in range(j):
                theta_star = bf.flip2(k_flip, theta_temp, p)
                if np.random.rand(1) <= min(1, np.exp((obj(theta_temp, Q = Q, beta = beta, lmbd = lamda)-obj(theta_temp, Q = Q, beta = beta, lmbd = lamda))/tau)):
                    theta_temp = theta_star
                theta_list += [theta_temp]
                tau = alpha * tau

    else : Exception("Wrong value at constraint. The constraint must be integer or None")
    
    result = theta_temp
    self.result =result
    self.theta_list = theta_list
    return result

