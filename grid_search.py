
# coding: utf-8

# In[1]:


from util import get_spiral, get_clouds
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import keras


# In[25]:


def grid_search():
    # get the data and split into train/test
    X, Y = get_spiral()
    # X, Y = get_clouds()
    X, Y = shuffle(X, Y)
    Ntrain = int(0.7*len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # hyperparameters to try
    hidden_layer_sizes = [
    [300],
    [100,100],
    [50,50,50],
    ]
    learning_rates = [1e-4, 1e-3, 1e-2]
    l2_penalties = [0., 0.1, 1.0]

    # loop through all possible hyperparameter settings
    best_validation_rate = 0
    best_hls = None
    best_lr = None
    best_l2 = None
    for hls in hidden_layer_sizes:
        for lr in learning_rates:
            for l2 in l2_penalties:
                model = keras.models.Sequential()
                if len(hls) == 1:
                    model.add(keras.layers.Dense(units=hls[0], input_dim=Xtrain.shape[1], activation='relu', 
                                             kernel_regularizer=keras.regularizers.l2(l2)))
                if len(hls) == 2:
                    model.add(keras.layers.Dense(hls[0], input_dim=2, activation='relu', 
                                             kernel_regularizer=keras.regularizers.l2(l2)))
                    model.add(keras.layers.Dense(hls[1], activation='relu', 
                                             kernel_regularizer=keras.regularizers.l2(l2)))
                if len(hls) == 3:
                    model.add(keras.layers.Dense(hls[0], input_dim=2, activation='relu', 
                                                 kernel_regularizer=keras.regularizers.l2(l2)))
                    model.add(keras.layers.Dense(hls[1], activation='relu', 
                                             kernel_regularizer=keras.regularizers.l2(l2)))
                    model.add(keras.layers.Dense(hls[2], activation='relu', 
                                             kernel_regularizer=keras.regularizers.l2(l2)))
                model.add(keras.layers.Dense(1, activation='sigmoid'))
                adam = keras.optimizers.Adam(lr=lr)
                model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
                hist = model.fit(Xtrain, Ytrain, epochs=3000, batch_size=300, verbose=0, validation_data=(Xtest, Ytest))
                validation_accuracy = hist.history['val_acc'][-1]
                train_accuracy = hist.history['acc'][-1]
#                 validation_accuracy = np.mean(np.equal(Ytest, np.round(model.predict(Xtest))))
#                 train_accuracy = np.mean(np.equal(Ytrain, np.round(model.predict(Xtrain))))
#                 validation_accuracy = model.evaluate(Xtest, Ytest)
#                 train_accuracy = model.evaluate(Xtrain, Ytrain)
#                 model = ANN(hls)
#                 model.fit(Xtrain, Ytrain, learning_rate=lr, reg=l2, mu=0.99, epochs=3000, show_fig=False)
#                 validation_accuracy = model.score(Xtest, Ytest)
#                 train_accuracy = model.score(Xtrain, Ytrain)
                print(
                    "validation_accuracy: %.3f, train_accuracy: %.3f, settings: %s, %s, %s" %
                    (validation_accuracy, train_accuracy, hls, lr, l2)
                )
                if validation_accuracy > best_validation_rate:
                    best_validation_rate = validation_accuracy
                    best_hls = hls
                    best_lr = lr
                    best_l2 = l2
    print("Best validation_accuracy:", best_validation_rate)
    print("Best settings:")
    print("hidden_layer_sizes:", best_hls)
    print("learning_rate:", best_lr)
    print("l2:", best_l2)


# In[26]:


grid_search()


# In[15]:


X, Y = get_spiral()
# X, Y = get_clouds()
X, Y = shuffle(X, Y)
Ntrain = int(0.7*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

model = keras.models.Sequential()
model.add(keras.layers.Dense(300, input_dim=2, activation='relu', 
                             kernel_regularizer=keras.regularizers.l2(0)))
model.add(keras.layers.Dense(1, activation='sigmoid'))
adam = keras.optimizers.Adam(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

hist = model.fit(Xtrain, Ytrain, epochs=10, batch_size=300, verbose=0, validation_data=(Xtest, Ytest))
validation_accuracy = model.evaluate(Xtest, Ytest)
train_accuracy = model.evaluate(Xtrain, Ytrain)

hist.history


# In[3]:


y_pred = model.predict_classes(Xtest)
validation_accuracy = np.count_nonzero(y_pred == Ytest)/len(Ytest)


# In[9]:


Ytest


# In[7]:


loss, acc = model.evaluate(Xtest, Ytest)
print(loss, acc)


# In[5]:


np.count_nonzero(y_pred == Ytest)


# In[48]:


train_accuracy

