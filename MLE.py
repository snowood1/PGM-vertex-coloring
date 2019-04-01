import numpy as np
from itertools import *
from Gibbs_Sampling import *
from Belief_Propagation import *


class features:
    def __init__(self,theta):
        self.theta = theta
        self.size = theta.size
        self.f = np.empty(theta.size)

    def get(self,x):
        vector = np.zeros(self.size)
        vector[x]=1
        return vector

    def calculate_fc(self,key,p=1):
        return np.array([self.get(x)*p for x in key]).sum(axis=0)


def colormle(A, samples, K, option,L=None):

    learning_rate = 10 ** -5
    iteration = 10000
    its = 10
    early_stop = 10 ** -9

    N=len(A)
    colors =[k for k in range(K)]
    theta = 1/K*np.ones(K)
    f=features(theta)

    #--------- anlaysis samples ------#

    sample= np.array(list(samples.values()))

    if option== 'EM':
        l=np.array(L)
        lat = np.where(l!=1)[0]
        lat_perm = [colors]*len(lat)

        obs =np.where(l==1)[0]
        obs_perm = [colors]*len(obs)

        sample=sample[obs].transpose()
    else:
        sample=sample.transpose()

    for t in range(iteration):

        # E step: get the expected distribution
        if option== 'EM':
            d={}
            for xobs in product(*obs_perm):
                temp={}
                px_sumy =0      # P(xi,Y|theta)
                for y in product(*lat_perm):

                    # print('\ty',y,end='')

                    x = np.zeros(N)
                    x[lat]=y
                    x[obs]=xobs

                    # print('\tX',x,end='\t')
                    pxy = get_joint_prob(theta, A,x)    # P(xi,y|theta)
                    temp[y] = pxy
                    px_sumy = px_sumy +pxy
                    # print('\tPxy:',pxy,end='|\t')
                # print('\ntemp', temp)
                # print('\n\tPxY:',px_sumy)

                if px_sumy !=0:   # only save the combination that will exist
                    d[xobs]=normalize(temp)

            fc={}
            for key in d:
                # print('key',key,d[key])
                fc_x =  f.calculate_fc(key)  # fc from observable variables
                # print('\tfc_x',fc_x)
                fc_y =0
                for y in d[key]:
                    p=d[key][y]
                    if p!=0:
                        fc_y = fc_y + f.calculate_fc(y,p)
                        # print('\tfc_y',fc_y, end='')
                # print()

                fc[key] = fc_x + fc_y
            # print('\nfc', fc)
            # fcM =f.calculate_fc(cnt,fc)


        # calculate fcM:

        if option== 'EM':
            cnt =Counter()
            for i in sample:
                i=tuple(i)
                cnt[i] += 1
            M = sum(cnt.values())
            fcM = np.array([cnt[c]*fc[c] for c in cnt]).sum(axis=0)

        else: # MLE
            cnt= Counter()
            for i in list(samples.values()):
                cnt = cnt+ Counter(i)
            M= [len(v) for v in samples.values()][0]
            fcM = np.array(sum([cnt[i]* f.get(i) for i in cnt]))


        # bp calculate p(xc,y|theta)

        z, p = sumprod(A, theta, its)
        a=np.array([list(p[i].values()) for i in p])
        sum_pc = np.array([sum(x) for x in zip(*a)])
        sum_pc_M = M * sum_pc

        # gradient
        delta = learning_rate *(fcM - sum_pc_M)
        if np.all(delta < early_stop):
            return theta

        theta = theta + delta

        if (t%100==0):
            print('--t=',t,'theta:',theta)#
    return theta


if __name__ == "__main__":

    # A = np.array(
    # [[0, 1, 0],
    #  [1, 0, 1],
    #  [0, 1, 0]])
    #
    # w = [1, 4, 8]
    # w = [i/sum(w) for i in w]  # My code will performance well when weights are normalized.
    # print('normalized weights:', w)
    #
    # init_states=[0, 1, 2]
    #
    #
    #
    A = np.array(
    [[0, 1, 0, 1],
     [1, 0, 1, 0],
     [0, 1, 0, 0],
     [1, 0, 0, 0]]
    )

    L=[1,1,0,0]

    w = [1, 2, 4, 8]
    w = [i/sum(w) for i in w]
    print('normalized weights:', w)
    init_states=[0, 1, 2, 3]

    # Let's prepare some samples from the gibbs sampler

    samples = gibbs(A, w, init_states, 2 ** 18, 2 ** 18)


    print('\n ----- MLE ------- \n')
    mle = colormle(A, samples, len(w),'MLE')

    print('\n ----- EM ------- \n')
    em = colormle(A, samples, len(w),'EM',L)

    print('\n----- Final Result -----\n')
    print('original weights',w)
    print('MLE:',mle)
    print('EM:',em)


