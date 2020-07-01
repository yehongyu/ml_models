import numpy as np
#import matplotlib.pyplot as plt
 
def multiGaussian(x,mu,sigma):
    return 1/((2*np.pi)*pow(np.linalg.det(sigma),0.5))*np.exp(-0.5*(x-mu).dot(np.linalg.pinv(sigma)).dot((x-mu).T))
 
def computeGamma(X,mu,sigma,alpha,multiGaussian):
    n_samples=X.shape[0]
    n_clusters=len(alpha)
    gamma=np.zeros((n_samples,n_clusters))
    p=np.zeros(n_clusters)
    g=np.zeros(n_clusters)
    for i in range(n_samples):
        for j in range(n_clusters):
            p[j]=multiGaussian(X[i],mu[j],sigma[j])
            g[j]=alpha[j]*p[j]
        for k in range(n_clusters):
            gamma[i,k]=g[k]/np.sum(g)
    return gamma
 
class MyGMM():
    def __init__(self,n_clusters,ITER=50):
        self.n_clusters=n_clusters
        self.ITER=ITER
        self.mu=0
        self.sigma=0
        self.alpha=0

    def fit(self,data):
        n_samples=data.shape[0]
        n_features=data.shape[1]
        '''
        mu=data[np.random.choice(range(n_samples),self.n_clusters)]
        '''
        alpha=np.random.rand(self.n_clusters)
        alpha=alpha/np.sum(alpha)
        print('init alpha:', alpha)

        mu = []
        for i in range(self.n_clusters):
            mu.append(np.random.random(n_features))
        print('init mu:', mu)

        sigma=np.full((self.n_clusters,n_features,n_features),np.diag(np.full(n_features,0.1)))
        print('init sigma:', sigma)
        for i in range(self.ITER):
            gamma=computeGamma(data,mu,sigma,alpha,multiGaussian)
            alpha=np.sum(gamma,axis=0)/n_samples
            for i in range(self.n_clusters):
                #print(np.shape(np.sum(data*gamma[:,i].reshape((n_samples,1)),axis=0)/np.sum(gamma,axis=0)[i]))
                mu[i]=np.sum(data*gamma[:,i].reshape((n_samples,1)),axis=0)/np.sum(gamma,axis=0)[i]
                sigma[i]=0
                for j in range(n_samples):
                    sigma[i]+=(data[j].reshape((1,n_features))-mu[i]).T.dot((data[j]-mu[i]).reshape((1,n_features)))*gamma[j,i]
                sigma[i]=sigma[i]/np.sum(gamma,axis=0)[i]
        self.mu=mu
        self.sigma=sigma
        self.alpha=alpha

    def predict(self,data):
        pred=computeGamma(data,self.mu,self.sigma,self.alpha,multiGaussian)
        cluster_results=np.argmax(pred,axis=1)
        return cluster_results

n_cluster = 2
n_feature = 1
gmm = MyGMM(n_cluster)
a = np.random.normal(0, 1, (10, n_feature))
b = np.random.normal(1, 1, (12, n_feature))
c = np.random.normal(2, 1, (16, n_feature))
data = np.append(a, b, axis=0)
data = np.append(data, c, axis=0)
print(data.shape)
gmm.fit(data)
res = gmm.predict(data)
print(res)


