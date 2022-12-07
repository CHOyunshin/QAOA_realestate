from dimod import ConstrainedQuadraticModel, Integer
import numpy as np
import pandas as pd
from dwave.system import LeapHybridCQMSampler
from sklearn.linear_model import LinearRegression


def dwave(x,y):
    data_x = pd.DataFrame(x)
    data_y = pd.DataFrame(y)
    p = data_x.shape[1]
    Q = np.abs(data_x.corr())
    beta = -Q.apply(sum)
    for i in range(p) : 
        Q.iloc[i,i] = 0 
        r_squared_list = []
        names = []
        for i in range( len(data_x.columns )) :
            names.append("x"+str(i))
        data_x.columns = names    
            
    beta2 = -Q.apply(sum)
    beta2_with_r2 = beta2 - partial_r1(x,y)
    B=beta2_with_r2
    integer_list = []
    for i in range(10):
        integer_list.append(Integer(str("x")+str("0")+str("0")+str(i), upper_bound=1,lower_bound=0))
    for i in range(10,100) :
        integer_list.append(Integer(str("x")+str("0")+str(i), upper_bound=1,lower_bound=0))
    for i in range(100,p) :
        integer_list.append(Integer(str("x")+str(i), upper_bound=1,lower_bound=0))
        
    linear_qubo = B[0]*integer_list[0]*0
    for i in range(p): 
        linear_qubo = linear_qubo + B[i]*integer_list[i]
    
    quadratic_qubo = Q[0][0]*integer_list[0]*integer_list[0]*0
    for j in range(p):
        for i in range(p):
            quadratic_qubo= quadratic_qubo+Q[i][j]*integer_list[i]*integer_list[j]
    
    Qubo = linear_qubo + quadratic_qubo
    cqm= ConstrainedQuadraticModel()
    cqm.set_objective(Qubo)
    sampleset = sampler.sample_cqm(cqm)
    
    result = sampleset.first[0]
    result_list=list(zip(result.keys(), result.values()))
    
    data_x_result=pd.DataFrame(result_list).T.loc[[1]]
    names = []
    for i in range( len(data_x_result.columns )) :
        names.append("x"+str(i))
    data_x_result.columns = names  
    data_y.columns = ["y"]
    
    data_x_result.index = ["Selection"]
    data_x_result=pd.to_numeric(data_x_result.T["Selection"])
    
    concat_x = pd.concat([pd.DataFrame(data_x_result).T, data_x], axis = 0)
    concat_x_one=concat_x.T[concat_x.T["Selection"]==1].T
    data_x_one =concat_x_one.loc[0:,:]
    
    
    mse_result=mse( np.array(data_x_one) ,np.array(data_y))
    aic_result = (len(data_x_one)* np.log(mse_result) - 10 *np.log(len(data_x_one)) + 2* len(data_x_one.columns))
    cn_result = cn(data_x_one)
    return result, mse_result, aic_result, cn_result


def dwave_QUBO(Q,beta):
    p = len(Q)

    integer_list = []
    for i in range(p) :
        integer_list += [Integer(str("x")+str(i).zfill(3), upper_bound=1,lower_bound=0)]
    
    linear_qubo = 0
    for i in range(p): 
        linear_qubo += [beta[i]*integer_list[i]]
    
    quadratic_qubo = 0
    for j in range(p):
        for i in range(p):
            quadratic_qubo += Q[i][j]*integer_list[i]*integer_list[j]
    
    Qubo = linear_qubo + quadratic_qubo
    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(Qubo)
    sampleset = sampler.sample_cqm(cqm)
    
    result = sampleset.first[0]
    result_list=list(zip(result.keys(), result.values()))
    
    return result_list


