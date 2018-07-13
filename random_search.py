
# coding: utf-8

# In[1]:


from util import get_spiral, get_clouds
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import keras


# In[2]:


def random_search():
    # get the data and split into train/test
    X, Y = get_spiral()
    # X, Y = get_clouds()
    X, Y = shuffle(X, Y)
    Ntrain = int(0.7*len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # starting hyperparameters
    M = 20
    nHidden = 2
    log_lr = -4
    log_l2 = -2 # since we always want it to be positive
    max_tries = 30

    # loop through all possible hyperparameter settings
    best_validation_rate = 0
    best_hls = None
    best_lr = None
    best_l2 = None
    for _ in range(max_tries):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=M, input_dim=Xtrain.shape[1], activation='relu', 
                                             kernel_regularizer=keras.regularizers.l2(10**log_l2)))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        adam = keras.optimizers.Adam(lr=10**log_lr)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        hist = model.fit(Xtrain, Ytrain, epochs=3000, batch_size=300, verbose=0, validation_data=(Xtest, Ytest))
        validation_accuracy = hist.history['val_acc'][-1]
        train_accuracy = hist.history['acc'][-1]
        print(
            "validation_accuracy: %.3f, train_accuracy: %.3f, settings: %s, %s, %s" %
            (validation_accuracy, train_accuracy, [M]*nHidden, log_lr, log_l2)
        )
        if validation_accuracy > best_validation_rate:
            best_validation_rate = validation_accuracy
            best_M = M
            best_nHidden = nHidden
            best_lr = log_lr
            best_l2 = log_l2

        # select new hyperparams
        nHidden = best_nHidden + np.random.randint(-1, 2) # -1, 0, or 1
        nHidden = max(1, nHidden)
        M = best_M + np.random.randint(-1, 2)*10
        M = max(10, M)
        log_lr = best_lr + np.random.randint(-1, 2)
        log_l2 = best_l2 + np.random.randint(-1, 2)
        print("Best validation_accuracy:", best_validation_rate)
        print("Best settings:")
        print("best_M:", best_M)
        print("best_nHidden:", best_nHidden)
        print("learning_rate:", best_lr)
        print("l2:", best_l2)


# In[3]:


random_search()

