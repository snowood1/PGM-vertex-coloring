import numpy as np
from random import choices
from collections import Counter

class Vertex:
    def __init__(self, num, value=None):
        self.name = 'X'+str(num)
        self.value = value
        self.neighbor = []

def get_potential(par):
    if len(par)==1:
        return np.exp(par[0])
    else:
        return (1 if par[0] != par[1] else 0)


def printState(states):
    print('--- Gibbs Sampling ---')
    print([vex.name for vex in states])

    mat = list(map(list, zip(*states.values())))
    N= len(mat)
    if N>10:
        showrow=list(range(0,5))+list(range(N-5,N))
    else:
        showrow=range(N)
    for i in showrow:
        print('t=',i,':', mat[i])
        if i==4:
            print('......')

def gibbs(A,weights, init_states, burnin,its):
    N= len(A)
    K=len(weights)

    vertex_list = list()

    for i in range(N):
        vertex = Vertex(i)
        vertex_list.append(vertex)

    for i in range(N):
        for j in range(i+1,N):
            if A[i,j]==1:
                vertex_list[i].neighbor.append(vertex_list[j])
                vertex_list[j].neighbor.append(vertex_list[i])

    states = dict()
    for vex, init in zip(vertex_list,init_states):
        states[vex]= [init]
        vex.value = init

    for i in range(burnin+its) :
        # print('\n\n\niteration, ', i)

        for vex in vertex_list:
            # print('---variable', vex.name, '=', vex.value)
            # print('\t before sampling vex:', [vex.value for vex in vertex_list])
            var_prob = {}

            for k in range(K):
                var  =  weights[k]
                node_potential = get_potential([var])
                var_prob[k] = node_potential

                # print('\t1 joint_prob', var_prob)

                for neighbor in vex.neighbor:
                    # print('\t\tneighbor', neighbor.name, '=', neighbor.value)
                    edge_potential = get_potential([k,neighbor.value])

                    # print('\t\tk:', k, 'weights:',var,' neighbor:', neighbor.name, '=', neighbor.value,'f:',edge_potential)
                    var_prob[k] = var_prob[k] * edge_potential
                    # print('\t2 joint_prob', var_prob)

            # print('--- var_prob', var_prob)

            sampling = choices(range(K), var_prob.values())
            # print(Counter(sampling))
            # print('---variable', vex.name, '=', vex.value)
            vex_sample = sampling[0]
            vex.value = vex_sample
            # print('\t\tNow its sampling: ', vex_sample)
            # print('\t\t after sampling vex value:', [vex.value for vex in vertex_list])
            #

            if i< burnin:
                states[vex][0] = vex_sample
            else:
                states[vex].append(vex_sample)


    # print('\nFinal States:')

    # print(states)
    printState(states)

    # print('\n------------- calculate probabilities----------\n')

    prob= np.zeros([N, K])


    for i in range(N):
        vex=vertex_list[i]
        # print(vex.name)
        for k, w in enumerate(weights):
            # print('\tk,w',k,w)
            count = states[vex].count(k)
            prob[i,k] = count/len(states[vex])
            # print(vex.name,'=',k, 'w:',w ,'count:',count,prob[i,k])
    print(prob)
    return states


if __name__ == "__main__":

    # A = np.array(
    #     [[0, 1, 1, 1],
    #      [1, 0, 0, 1],
    #      [1, 0, 0, 1],
    #      [1, 1, 1, 0]]
    # )

    A = np.array(
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]])

    w = [1,2,3,4]
    init_states=[1,2,3,4]

    w =[0, 1, 2]
    init_states=[0, 1, 2]

    times = [2 ** 6, 2 ** 10, 2 ** 14, 2 ** 18]



    # print('\n\tInitial states:', init_states)
    gibbs(A,w, init_states, 2 ** 14,2 ** 14)
#
    # for burnin in times:
    #     for its in times:
    #
    #         print('\nburnin:', burnin, 'its:', its)
    #
    #         init_states=[1,2,3,4]
    #         print('\n\tInitial states:', init_states)
    #         gibbs(A,w, init_states, burnin,its)
    #
    #         init_states=[4,3,2,1]
    #         print('\n\tInitial states:', init_states)
    #         gibbs(A,w, init_states, burnin,its)
