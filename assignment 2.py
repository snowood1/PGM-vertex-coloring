import numpy as np
from itertools import product

class Vertex:
    def __init__(self, num, value=None):
        self.name = 'X'+str(num)
        self.neighbor = []

def get_potential(par):
    if len(par)==1:
        return np.exp(par[0])
    else:
        return (1 if par[0] != par[1] else 0)

def normalize(d):
    factor=1.0/sum(d.values())
    for k in d:
        d[k] = d[k]*factor
    return d

def belief(vex,x,message):
    b = get_potential([x])
    for nb in vex.neighbor:
        b *= message[(nb.name, vex.name)][x]
    return b

def calulate_prob(vex,weights,message):
    p = dict()
    for x in weights:
        p[x] = belief(vex,x,message)
    p=normalize(p)
    return p

def get_joint_prob(x):
    N=len(x)
    prob = 1
    for i in range(N):
        prob  = prob * get_potential([x[i]])
        for j in range(i+1,N):
            if A[i,j]==1:
                edge_potential = get_potential([x[i],x[j]])
            if edge_potential ==0:
                return 0
            prob = prob * edge_potential
    return prob

def send_msg(message,weights,i,j,option):

    for xj in weights:

        msg=dict()

        for xi in weights:
            node_potential = get_potential([xi])
            edge_potential = get_potential([xi,xj])

            msg_to_i = 1

            for nb in i.neighbor:
                if nb != j:
                    msg_to_i =  msg_to_i * message[(nb.name, i.name)][xi]

            msg[xi] = node_potential * edge_potential * msg_to_i

        if option=='sum_product':
            message[(i.name, j.name)][xj]= sum(msg.values())
        else:
            message[(i.name, j.name)][xj]= max(msg.values())

    # print('message[(i.name, j.name)]',message[(i.name, j.name)])
    message[(i.name, j.name)]= normalize(message[(i.name, j.name)])
    return message
    # print('message',i.name,'-->' ,j.name, message[(i.name, j.name)])


def BP(A,weights,its,option):
    N= len(A)

    # initiate  graph
    vertex_list = list()

    for i in range(N):
        vertex = Vertex(i)
        vertex_list.append(vertex)

    for i in range(N):
        for j in range(i+1,N):
            if A[i,j]==1:
                vertex_list[i].neighbor.append(vertex_list[j])
                vertex_list[j].neighbor.append(vertex_list[i])

    ##  initiate message

    message = dict()
    for vex in vertex_list:
        for nb in vex.neighbor:
            message[(vex.name, nb.name)]= {k: 1 for k in weights}


    ## belief propagation

    for t in range(its) :

        ###  i ---> j
        for m in range(N):
            i = vertex_list[m]
            for n in range(m+1,N):
                j=vertex_list[n]
                if j in i.neighbor:
                    message = send_msg(message,weights, i,j,option)

        ###  j ---> i

        for m in range(N-1, -1, -1):
            i=vertex_list[m]
            for n in range(m-1,-1,-1):
                j=vertex_list[n]
                if j in i.neighbor:
                    message = send_msg(message,weights,i,j,option)


    assignment= dict()

    # calculate belief
    for vex in vertex_list:
        p=calulate_prob(vex,weights,message)
        # print(vex.name, 'calulate_prob:', p)
        assignment[vex.name]= max(p, key=p.get)

    if option == 'sum_product':
        z= partion(weights,N)
        return z

    return assignment

def partion(weights,N):
    perm=[]
    for i in range(N):
        perm.append(weights)
    z=0
    for x in product(*perm):
        joint_prob=get_joint_prob(x)
        z = z + joint_prob
    return z


def sumprod(A, weights, its):
    return BP(A,weights,its,'sum_product')


def maxprod(A, weights, its):
    return BP(A,weights,its,'max_product')


if __name__ == "__main__":

    A = np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    )

    w = [1, 2, 3]

    its=100

    print('sum product')
    print(sumprod(A, w, its))

    print('\nmax product')
    print(maxprod(A, w, its))
