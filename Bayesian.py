import numpy as np, networkx as nx, pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import product,combinations
from utility import *
import mst
np.seterr(divide='ignore')

class NB:
    def __int__(self):
        self.traindata=None
        self.alpha = 0
        self.N=0  #
        self.t=0
        self.py= None
        self.px_y =[]
        self.domain =[]

    def get_cnt(self,i):
        cnt = defaultdict(float)
        for row in self.traindata[:,i]:
            cnt[row] += 1
        return cnt

    def get_pair_cnt(self,i,j):
        cnt = {key: 0 for key in product(self.domain[i],self.domain[j])}
        for row in self.traindata[:,[i,j]]:
            cnt[tuple(row)] += 1
        return cnt

    def get_conditionProb(self,i,j):  # P(xi/y)
        cxy = self.get_pair_cnt(i,j)
        cy = self.get_cnt(j)
        px_y = defaultdict(dict)
        for x,y in cxy:
            px_y[y][x]= (cxy[x,y]+self.alpha)/(cy[y] + self.alpha * len(self.domain[i]))
        return px_y

    def importData(self,filename,j):
        df = pd.read_csv(filename, header=None)
        N, t = df.shape
        index =  [j] + [x for x in range(t) if x != j]  # Move the labels to the first column
        df = df[index]
        domain=[]
        for col in df:
            domain.append(set(df[col].unique()))
        return df.values, N, t, domain


    def fit(self,filename,j,alpha=0):
        self.alpha =alpha
        self.traindata, self.N, self.t, self.domain = self.importData(filename,j)

    def test(self,filename,j):

        testdata, N, t, domain = self.importData(filename,j)

        for a, b in zip(self.domain, domain):
            a = a.union(b)

        # marginal prob P(y)
        cnt = self.get_cnt(0)
        self.py = {key: cnt[key]/self.N for key in cnt}

        # conditional distributions P(xi|y) for naive bayes
        self.px_y = [self.get_conditionProb(i,0) for i in range(1, self.t) ]

        acc=0
        for row in testdata:
            X = tuple(row[range(1,t)])
            pred = self.predict(X)
            if row[0]== pred:
                acc += 1
        return acc/N
		
    def predict(self,X):
        logp =defaultdict()
        for y in self.py:
            logp[y] =  np.log(self.py[y])
            for px_y , xi in zip(self.px_y, X):
                logp[y] += np.log(px_y[y][xi])
        return max(logp, key=logp.get)


    def get_mutual_information(self,i,j):
        I=0
        cxi=self.get_cnt(i)
        cxj=self.get_cnt(j)
        cxij = self.get_pair_cnt(i,j)

        for xi,xj in cxij:
            if cxij[xi,xj]!=0:
                # I += (cxij[xi,xj]+self.alpha)* np.log ((cxij[xi,xj]+self.alpha)/((cxi[xi]+self.alpha)*(cxj[xj])+self.alpha))
                I += cxij[xi,xj]* np.log (cxij[xi,xj]/(cxi[xi]*cxj[xj]))
        return I


    def chow_liu_tree(self):
        G = mst.Graph(self.t)
        for i,j in combinations(range(self.t),2):
            G.addEdge(i, j, -self.get_mutual_information(i, j))
        T = G.KruskalMST()
        return G, T


    def tree_inference(self,tree,filename,j):

        testdata, N, t, domain = self.importData(filename,j)

        for a, b in zip(self.domain, domain):
            a = a.union(b)

        # marginal prob P(y)
        cnt = self.get_cnt(0)
        self.py = {key: cnt[key]/self.N for key in cnt}

        # conditional distributions P(xi|par) for Chou Liu Tree
        px_par = {}
        for par in tree:
            children =tree[par]
            for child in children:
                px_par[(par,child)] = self.get_conditionProb(child,par)

        acc=0
        for row in testdata:
            logp =defaultdict()
            for y in self.py:
                logp[y] =  np.log(self.py[y])

            for par,child in px_par:
                if par == 0:
                    for y in self.py:
                        p= px_par[(par,child)][ y ][ row[child] ]
                        logp[y] += np.log(p)
                else:
                    p = px_par[(par,child)][ row[par] ][ row[child] ]
                    for y in self.py:
                        logp[y] += np.log(p)

            pred = max(logp, key=logp.get)
            if row[0]== pred:
                acc += 1
        return acc/N

if __name__ == '__main__':

    nb=NB()
    nb.fit('mushroom_train.data',0)

    G, T= nb.chow_liu_tree()

    print('\nChou Liu Tree:',T)

    dt = G.BFS(T,0)

    print('\nSet the label as the root:',dt)

    print('\nTrain',nb.tree_inference(dt,'mushroom_train.data',0))
    print('Test',nb.tree_inference(dt,'mushroom_test.data',0))

    print('\nNaive Bayes with prior parameters alpha')
    for alpha in [0, 0.1,0.5,1,10,100,1000]:
        print('\nalpha', alpha)

        nb=NB()
        nb.fit('mushroom_train.data',0,alpha)

        print('Train',nb.test('mushroom_train.data',0))
        print('Test',nb.test('mushroom_test.data',0))
