import numpy as np
from itertools import *

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
    s = sum(d.values())
    if s!=0:
        factor=1.0/s
        for k in d:
            d[k] = d[k]*factor
    return d

def belief(vex,i,x,message):
    # print('---belief')
    b = get_potential([x])
    # print('\tb',b,'x',x)
    # print('\tmessage',message)
    for nb in vex.neighbor:
        # print('\t\tnb',nb,message[(nb.name, vex.name)][i])
        b *= message[(nb.name, vex.name)][i]
    return b

def calulate_prob(vex,weights,message):
    K=len(weights)
    p = dict()
    for i in range(K):
        x = weights[i]
        p[i] = belief(vex,i,x, message)

    # print('p',p,'vex',vex.name,'weights',weights)
    p=normalize(p)
    return p

def get_joint_prob(weights,A,x):
    N=len(x)
    prob = 1
    # print('x',x)
    for i in range(N):
        idx = int(x[i])
        wi=weights[idx]
        # print('wi',wi)
        prob = prob * get_potential([wi])
        for j in range(i+1,N):
            if A[i,j]==1:
                edge_potential = get_potential([x[i],x[j]])
            if edge_potential ==0:
                return 0
            prob = prob * edge_potential
    return prob

def send_msg(message,weights,i,j,option):

    colors = list(range(len(weights)))
    # print('colors',colors)

    for xj in colors:

        msg=dict()

        for xi in colors:
            node_potential = get_potential([weights[xi]])
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
    K=len(weights)

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
            message[(vex.name, nb.name)]= {k: 1 for k in range(K)}
    # print('message',message)


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

    # print('\n--- message',message)
    # calculate belief
    p=dict()
    for vex in vertex_list:
        p[vex]=calulate_prob(vex,weights,message)
        # print(vex.name, 'calulate_prob:', p[vex])
        assignment[vex.name]= max(p[vex], key=p[vex].get)

    if option == 'sum_product':
        # print('A',A)
        z = partion(weights,A)
        return z, p

    return assignment

def partion(weights,A):
    N=len(A)

    colors = list(range(len(weights)))
    # print(colors)
    # for i in range(N):
    #     perm.append(colors)
    perm = [colors]*N
    # print(perm)
    z=0

    # lat_perm =[list(zip(lat_var, color)) for color in permutations(colors,N)]

    # perm = [color for color in permutations(colors,N)]
    # perm =[list(zip(range(N), color)) for color in permutations(colors,N)]

    joint_prob=dict()
    for x in product(*perm):
        # print('BP',x)
        jp = get_joint_prob(weights, A,x)
        # joint_prob[x] = jp
        # print(x,joint_prob)
        z = z + jp
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

    # w = [0.9,0.8,0.7,0.6]
    # w = [0.5,0.5,0.5,0.5]
    w=[0,1,2]
    its=10

    print('sum product')
    z,p  = sumprod(A, w, its)
    print(z,'\n',p)


    # print('\nmax product')
    # print(maxprod(A, w, its))

# exp(-inf)

# []
# sum_pc = np.array(list(p.values()))
# print('sum_pc',sum_pc)
# sum_pc = N * M * sum_pc
# #
#     a=[list(p[i].values()) for i in p]
#     [sum(x) for x in zip(*a)]

    # # for i in list(p.values()):
    #     print(i)
    # print(su)
