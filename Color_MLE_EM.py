from Belief_Propagation import *
from Gibbs_Sampling import *
import time
import matplotlib.pyplot as plt

def get_joint_prob(weights,A,x):   # This is a function to calculate joint probabilities given all variables without normalization
    N=len(x)
    prob = 1
    # print('x',x)
    for i in range(N):
        idx = int(x[i])
        wi=weights[idx]
        prob = prob * get_potential([wi])
        for j in range(i+1,N):
            if A[i,j]==1:
                edge_potential = get_potential([x[i],x[j]])
            if edge_potential ==0:
                return 0
            prob = prob * edge_potential
    return prob

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

class ColorMLE():
    def __init__(self,A, K):
        self.A= A
        self.K = K
        self.learning_rate = 10 ** -4
        self.iteration = 10000
        self.its=10
        self.patience = 5
        self.threshold = 10 ** -7
        self.show = 100

    def run(self,samples,option,L=None):
        run_start=time.time()

        colors =[k for k in range(self.K)]
        theta = []
        theta.append(np.zeros(self.K))
        times = []
        times.append(0)



        #0-1 vector:
        f=features(theta[0])

        #--------- anlaysis samples ------#
        sample= np.array(list(samples.values()))


        if option=='MLE':
            # sample=sample.transpose()
            cnt= Counter()
            for i in list(samples.values()):
                cnt = cnt+ Counter(i)
            M= [len(v) for v in samples.values()][0]

        else:
            l=np.array(L)

            # Positions of observable variables
            obs =np.where(l==1)[0]
            obs_perm = [colors]*len(obs)

            # Positions of latent varibales
            lat = np.where(l!=1)[0]
            lat_perm = [colors]*len(lat)


            sample=sample[obs].transpose()

            cnt =Counter()
            for i in sample:
                i=tuple(i)
                cnt[i] += 1
            M = sum(cnt.values())

        for t in range(self.iteration):
            iter_start = time.time()
            # print('\n----------- iter------',t,'\n')

            if option == 'MLE':
                fcM = np.array(sum([cnt[i]* f.get(i) for i in cnt]))
            else:
                # E step: get the expected distribution
                # We need to convert every combination of observable and latent variables into equivalent ones. \
                # We need to calculate P(y|xi,theta) and there are two solutons:

                # 1. Use exact inference
                # 2. Use BP but still we need to calculate it in every combination.

                fc={}
                for xobs in cnt:
                    if option == 'EM_BP':  # Construct new MRFs and use BP to calculate P(X(i),y|theta) in every case

                        g= BP(self.A,theta[-1])
                        for pos, val in zip(obs,xobs):
                            g.setValue(pos,val)
                        g.sumprod(self.its)
                        jp = g.allJointProb()  # pxy = product of beliefs

                    else:
                        #  Exact inference by computing the normalizing constant manually
                        jp={}
                        for y in product(*lat_perm):
                            x = np.zeros(len(self.A),dtype=int)
                            x[lat]=y
                            x[obs]=xobs
                            pxy = get_joint_prob(theta[-1], self.A,x)    # P(xi,y|theta)

                            # save to unnormalized joint probabilities
                            jp[tuple(x)] = pxy
                        jp =normalize(jp)

                    # # save fc in every combination of observable variables
                    fc[xobs]=0
                    for key in jp:
                        pxy = jp[key]
                        fc[xobs] += f.calculate_fc(key,pxy)

                # calculate fcM
                fcM = np.array([cnt[c]*fc[c] for c in cnt]).sum(axis=0)

            ## Use BP to calculate the marginal probabilities Pc(xc,y|theta)

            g= BP(self.A,theta[-1])
            _, p= g.sumprod(self.its)
            a=np.array([list(p[i].values()) for i in p])
            sum_pc = np.array([sum(x) for x in zip(*a)])
            sum_pc_M = M * sum_pc

            # gradient
            theta.append(theta[-1] + self.learning_rate *(fcM - sum_pc_M))
            times.append(time.time()-run_start)

            if all(np.all(theta[-i]-theta[-i-1] <= self.threshold) for i in range(1,min(self.patience,len(theta)))):
                print('--t=',t+1,'theta:',theta[-1],'s/iter:',time.time()-iter_start)
                print('\tTotal time:', times[-1])
                return theta, times

            if (t%self.show==0):
                print('--t=',t+1,'theta:',theta[-1],'s/iter:',time.time()-iter_start)
        return theta,times

    def colorMLE(self, samples):
        return self.run(samples,'MLE')

    def colorEM(self, samples,L):
        return self.run(samples,'EM',L)

    def colorEM_BP(self, samples,L):
        return self.run(samples,'EM_BP',L)

if __name__ == "__main__":

# CASE 1 ---------------------------
    A = np.array(
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]])
    L=[1,0,1]
    wo = [1, 2, 3]
    print('original weights:', wo)
    w = [i-sum(wo)/len(wo) for i in wo]
    print('shifted to mean 0 weights:', w)
    init_states=[0, 1, 2] # picking any valid initiate states will result in the same sample distributions

    samples = gibbs(A, w, init_states, 2 ** 14, 2 ** 14)
    g= ColorMLE(A,len(w))

# --------------------------------------------------

    print('\n ----- MLE ------- \n')
    mle = g.colorMLE(samples)

    print('\n ----- EM BP ------- \n')
    embp = g.colorEM_BP(samples,L)

    print('\n ----- EM ------- \n')
    em = g.colorEM(samples,L)

    print('\n----- Final Result -----\n')
    print('original weights:', wo)
    print('shifted to mean 0 weights:', w)
    print('MLE:',mle[0][-1],'time:',mle[1][-1])
    print('EM:',em[0][-1],'time:',embp[1][-1])
    print('EM by All BP:',embp[0][-1],'time:',embp[1][-1])


    for x,name in zip([mle,embp,em],['mle','em_bp','em']):
        theta = x[0]
        times = x[1]
        y= np.sqrt(np.sum(np.square(theta-np.array(w)), axis=1))

        plt.plot(times,y,label= name)
        plt.legend()

        plt.xlabel("time(s)")
        plt.ylabel('|True - predicted theta|')
    plt.show()

