#Vectorized implementation on Multiclass SVM Loss + L2 Regularization

import numpy as np


'''
delta= hyperparameter (generally taken as 1.0)
D= feature dimension
N= number of features
C= number of classes [0,1,2...C-1]

X= feature matrix with shape: (D,N)
Y= correct class matrix with shape: (C,N)
W= weight matrix with shape: (C,D)

therefore
predict class scores=  W.X
'''

def loss(X,Y,W, delta, lamda):
    N=np.shape(X)[1]
    scores=W.dot(X)
    correct_class_scores=scores[Y, np.arange(N)]
    loss_vec=np.maximum(0, scores-correct_class_scores+delta)
    loss_vec[Y, np.arange(N)]=0
    loss=np.sum(loss_vec)
    loss=loss/N
    l2reg=np.sum(W*W)
    total_loss=loss+lamda*l2reg
    return total_loss

X=np.random.randint(0,100, size=(5, 50))
Y=np.random.randint(0, 3, size=(4, 50))
W=np.random.rand(4,5)

l=loss(X,Y,W, 1, 2)
print l



