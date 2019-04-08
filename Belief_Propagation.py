import numpy as np
from itertools import product
from copy import deepcopy

def normalize(d):
    s = sum(d.values())
    if s!=0:
        factor=1.0/s
        for k in d:
            d[k] = d[k]*factor
    return d

class Vertex:
    def __init__(self, num, value=None):
        self.num = num
        self.name = 'X'+str(num)
        self.neighbor = []
        self.values = value

    def setValue(self,value):
        if type(value)!=list:
            self.values = [value]
        else:
            self.values = value

class BP:
    def __init__(self,A,w):
        self.vertexList = []
        self.edgeList = []
        self.A = A
        self.w = w
        self.message={}
        self.init_graph()
        self.pc={}
        self.pi={}
        self.z=0

    def init_graph(self):
        N= len(self.A)
        K=len(self.w)

        colors = list(range(K))

        for i in range(N):
            vertex = Vertex(i,colors)
            self.vertexList.append(vertex)

        for m in range(N):
            i = self.vertexList[m]
            for n in range(m+1,N):
                j=self.vertexList[n]
                if self.A[m,n]==1:
                    self.edgeList.append((i,j))
                    i.neighbor.append(j)
                    j.neighbor.append(i)

    def setValue(self,i,w):
        self.vertexList[i].setValue(w)

    def msg_product(self,i,xi,j=None):
        p = 1
        for nb in i.neighbor:
            if j!=None and nb==j:
                continue
            p *= self.message[(nb.name, i.name)][xi]
        return p

    def node_belief(self,vex):
        b={}
        for i in vex.values:
            b[i] = self.potential([i]) * self.msg_product(vex,i)
        return normalize(b)

    def edge_belief(self,i,j):
        b={}
        for xi,xj in product(i.values,j.values):
            b[xi,xj] = self.potential([xi]) * self.potential([xj])* self.potential([xi,xj])*\
            self.msg_product(i,xi,j) *  self.msg_product(j, xj,i)
        return normalize(b)

    def potential(self,par):
        if len(par)==1:
            return np.exp(self.w[par[0]])
        else:
            return (1 if par[0] != par[1] else 0)

    def send_msg(self,i,j,option):

        for xj in j.values:
            msg=dict()
            for xi in i.values:
                msg[xi] = self.potential([xi]) * self.potential([xi,xj]) * self.msg_product(i,xi,j)
            if option=='sum_product':
                self.message[(i.name, j.name)][xj]= sum(msg.values())
            else:
                self.message[(i.name, j.name)][xj]= max(msg.values())

        self.message[(i.name, j.name)]= normalize(self.message[(i.name, j.name)])
        return self.message

    def run(self,its,option):

        ##  initiate message

        for vex in self.vertexList:
            for nb in vex.neighbor:
                self.message[(vex.name, nb.name)]= {k: 1 for k in nb.values}

        ##  belief propagation

        for t in range(its) :
            old_msg =  deepcopy(self.message)

            ###  i ---> j
            for i,j in self.edgeList:
                self.send_msg(i,j,option)

            ###  j ---> i
            for i,j in reversed(self.edgeList):
                self.send_msg(j,i,option)

            if self.message == old_msg:
                break

        if option == 'sum_product':
            self.z, self.pi = self.partition()
            return  self.z, self.pi

    def partition(self):
        pi={}
        logZ=0

        for vex in self.vertexList:
            b = self.node_belief(vex)
            pi[vex]=b

            for x in vex.values:
                pb = self.potential([x])
                logZ += b[x]*np.log(pb)
                logZ -= np.log((b[x])**b[x])

        for i,j in self.edgeList:
            bi = pi[i]
            bj = pi[j]
            bij = self.edge_belief(i,j)
            pxij={}

            for xi,xj in product(i.values,j.values):

                if bij[xi,xj] * bi[xi] * bj[xj] !=0:
                    pxij[xi,xj] = bij[xi,xj] / (bi[xi] * bj[xj])
                    self.pc[i,j] = pxij
                    logZ -= bij[xi,xj]* np.log(pxij[xi,xj])
                    logZ += bij[xi,xj]* np.log(self.potential([xi,xj]))
                else:
                    pxij[xi,xj]=0
                    self.pc[i,j]=pxij
        return np.exp(logZ), pi

    def allJointProb(self):
        jp={}
        perm = [v.values for v in self.vertexList]

        for x in product(*perm) :
            jp[x] = self.oneJointProb(x)
        return jp

    def oneJointProb(self,x):
        jp=1
        for id, vex in enumerate(self.vertexList):
            xi=x[id]
            jp *= self.pi[vex][xi]
        for i,j in self.edgeList:
            xi = x[i.num]
            xj = x[j.num]
            if (xi,xj) in self.pc[i,j]:
                jp *= self.pc[i,j][xi,xj]
        return jp

    def sumprod(self, its):
        return self.run(its,'sum_product')

    def maxprod(self, its):
        return self.run(its,'max_product')


if __name__ == "__main__":

    # A = np.array(
    #     [[0, 1, 0],
    #      [1, 0, 1],
    #      [0, 1, 0]]
    # )
    #
    # # w = [0.9,0.8,0.7,0.6]
    # # w = [0.5,0.5,0.5,0.5]
    # w=[0,1,2]
    # its=10
    #
    # print('sum product')
    # g= BP(A,w)
    # z, p = g.sumprod(its)
    # print('z',z)
    # for i in p:
    #     print(i.name,p[i])

    A = np.array(
        [[0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 1, 0, 0],
         [1, 0, 0, 0]])

    w=[0,1,2]
    its=10
    g= BP(A,w)
    z, p = g.sumprod(its)
    print('sum product')
    print(z,'\n',p)
    # print(g.allJointProb())
    # print(g.oneJointProb([2,1,2,1]))


