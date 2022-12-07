from dimod import ConstrainedQuadraticModel, Integer

def dwave_QUBO(Q,beta,lamda,A,b,p,sampler):
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


