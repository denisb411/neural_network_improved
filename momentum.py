
# coding: utf-8

# In[2]:


import numpy as np

def forward(X, W1, b1, W2, b2):
    # sigmoid
    # Z = 1 / (1 + np.exp(-( X.dot(W1) + b1 )))

    # relu
    Z = X.dot(W1) + b1
    Z[Z < 0] = 0

    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

def derivative_w2(Z, T, Y):
    return Z.T.dot(Y - T)

def derivative_b2(T, Y):
    return (Y - T).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2, act_function='relu'):
    if act_function == 'sigmoid':
        return X.T.dot( ( ( Y-T ).dot(W2.T) * ( Z*(1 - Z) ) ) ) # for sigmoid
    return X.T.dot( ( ( Y-T ).dot(W2.T) * (Z > 0) ) ) # for relu

def derivative_b1(Z, T, Y, W2, act_function='relu'):
    if act_function == 'sigmoid':
        return (( Y-T ).dot(W2.T) * ( Z*(1 - Z) )).sum(axis=0) # for sigmoid
    return (( Y-T ).dot(W2.T) * (Z > 0)).sum(axis=0) # for relu


# In[6]:


from util import get_normalized_data, error_rate, cost, y2indicator
import matplotlib.pyplot as plt

max_iter = 20 # make it 30 for sigmoid
print_period = 100

X, Y = get_normalized_data()
lr = 0.00004
reg = 0.01

Xtrain = X[:-1000,]
Ytrain = Y[:-1000]
Xtest  = X[-1000:,]
Ytest  = Y[-1000:]
Ytrain_ind = y2indicator(Ytrain)
Ytest_ind = y2indicator(Ytest)

N, D = Xtrain.shape
batch_sz = 500
n_batches = N // batch_sz

M = 300
K = 10
W1 = np.random.randn(D, M) / np.sqrt(D)
b1 = np.zeros(M)
W2 = np.random.randn(M, K) / np.sqrt(M)
b2 = np.zeros(K)

# save initial weights
W1_0 = W1.copy()
b1_0 = b1.copy()
W2_0 = W2.copy()
b2_0 = b2.copy()


# In[7]:


# 1. batch
losses_batch = []
errors_batch = []
for i in range(max_iter):
    for j in range(n_batches):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
        Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
        pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
        # print "first batch cost:", cost(pYbatch, Ybatch)

        # updates
        W2 -= lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
        b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
        W1 -= lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
        b1 -= lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)
        
        pY, _ = forward(Xtest, W1, b1, W2, b2)
        l = cost(pY, Ytest_ind)
        losses_batch.append(l)

        if j % print_period == 0:
            print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

            e = error_rate(pY, Ytest)
            errors_batch.append(e)
            print("Error rate:", e)

pY, _ = forward(Xtest, W1, b1, W2, b2)
print("Final error rate:", error_rate(pY, Ytest))

plt.plot(losses_batch, label="batch")
plt.legend()
plt.show()


# In[8]:


# 2. batch with momentum
W1 = W1_0.copy()
b1 = b1_0.copy()
W2 = W2_0.copy()
b2 = b2_0.copy()
losses_momentum = []
errors_momentum = []
mu = 0.9
dW2 = 0
db2 = 0
dW1 = 0
db1 = 0
for i in range(max_iter):
    for j in range(n_batches):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
        Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
        pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

        # gradients
        gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
        gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
        gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
        gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1

        # update velocities
        dW2 = mu*dW2 - lr*gW2
        db2 = mu*db2 - lr*gb2
        dW1 = mu*dW1 - lr*gW1
        db1 = mu*db1 - lr*gb1

        # updates
        W2 += dW2
        b2 += db2
        W1 += dW1
        b1 += db1

        pY, _ = forward(Xtest, W1, b1, W2, b2)
        l = cost(pY, Ytest_ind)
        losses_momentum.append(l)
        if j % print_period == 0:
            
            print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

            e = error_rate(pY, Ytest)
            errors_momentum.append(e)
            print("Error rate:", e)
pY, _ = forward(Xtest, W1, b1, W2, b2)
print("Final error rate:", error_rate(pY, Ytest))

plt.plot(losses_momentum, label="momentum")
plt.legend()
plt.show()


# In[9]:


# 3. batch with Nesterov momentum
W1 = W1_0.copy()
b1 = b1_0.copy()
W2 = W2_0.copy()
b2 = b2_0.copy()

losses_nesterov = []
errors_nesterov = []

mu = 0.9
vW2 = 0
vb2 = 0
vW1 = 0
vb1 = 0
for i in range(max_iter):
    for j in range(n_batches):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
        Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
        pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

        # updates
        gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
        gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
        gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
        gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1

        # v update
        vW2 = mu*vW2 - lr*gW2
        vb2 = mu*vb2 - lr*gb2
        vW1 = mu*vW1 - lr*gW1
        vb1 = mu*vb1 - lr*gb1

        # param update
        W2 += mu*vW2 - lr*gW2
        b2 += mu*vb2 - lr*gb2
        W1 += mu*vW1 - lr*gW1
        b1 += mu*vb1 - lr*gb1
        
        pY, _ = forward(Xtest, W1, b1, W2, b2)
        l = cost(pY, Ytest_ind)
        losses_nesterov.append(l)

        if j % print_period == 0:
            
            print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

            e = error_rate(pY, Ytest)
            errors_nesterov.append(e)
            print("Error rate:", e)
pY, _ = forward(Xtest, W1, b1, W2, b2)
print("Final error rate:", error_rate(pY, Ytest))

plt.plot(losses_nesterov, label="nesterov")
plt.legend()
plt.show()


# In[10]:


plt.plot(losses_batch, label="batch")
plt.plot(losses_momentum, label="momentum")
plt.plot(losses_nesterov, label="nesterov")
plt.legend()
plt.show()

