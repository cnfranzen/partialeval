import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Lambda
from keras import backend as K


def quadratic_loss(y, y_hat): 
    return np.power((y - y_hat), 2)

def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))

def create_bayesian_nn(input_shape, hidden_units, dropout_rate, activeF='elu'):
    model = keras.Sequential()
    
    # Input layer
    model.add(Dense(hidden_units, input_shape=input_shape, activation=activeF))
    
    # Dropout layer for Bayesian inference
    model.add(PermaDropout(rate= dropout_rate))
    
    # Output layer
    model.add(Dense(1))  # For regression, no activation function
    
    return model

def find_mean_var_bootstrap(X, reg_evaluator, num_bootstrap = 30):

    y_preds = np.zeros([len(X), num_bootstrap])

    for ind in range(num_bootstrap):
        y_preds[:, ind] = reg_evaluator(X).numpy().ravel()
    
    meanVec = y_preds.mean(axis=1)
    varVec = y_preds.var(axis=1)
    
    return meanVec, varVec
    

def risk_estimator_partial(y_true, y_pred, N,  acq_weights, loss_function, mode='unweighted'):
    '''
    N: total number of samples in D_test 
    M: number of selected samples (y_true, y_pred)
    acq_weights: sampling distribution for each index (M elements)
    mode: "unweighted" or "weighted" 
    '''
    
    M = y_true.shape[0]
    
    if mode == 'unweighted':
        v_i = np.ones(M)
    
    if mode == 'weighted':
        
        m = np.arange(1, M+1) 
        
        v_i = 1 + ((N-M)/(N-m)) * (1/((N-m+1) * acq_weights) - 1)
        
    
    l_i = loss_function(y_true, y_pred)
    
    R = (v_i * l_i).mean()
    
    return R

def surrogate_sampling(X_train, y_train, X_test, y_pred, M, input_shape, hidden_units,
                       dropout_rate, learning_rate, mode='BiasVar', loss=quadratic_loss):
    
    # keep track of selected indices and weights 
    N = X_test.shape[0]
    remaining_idx = np.arange(N, dtype=int)
    observed_idx = np.array([], dtype=int)
    weights = np.array([])

    # train the evaluator model 
    reg_evaluator = create_bayesian_nn(input_shape=input_shape, hidden_units=hidden_units, 
                                       dropout_rate=dropout_rate)
    reg_evaluator.compile(loss='mean_squared_error', 
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    reg_evaluator.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)
    
    # find the mean and variance associated with the evaluator 
    meanVec, varVec = find_mean_var_bootstrap(X_test, reg_evaluator)
    
    for _ in range(M):
        if mode == 'BiasVar': 
            probs = loss(y_pred[remaining_idx], meanVec[remaining_idx]) + varVec[remaining_idx]
        elif mode == 'Var':
            probs = varVec[remaining_idx]

        pmf = probs / probs.sum()
        sample = np.random.multinomial(1, pmf)
        idx = np.where(sample)[0][0]
        observed_idx = np.append(observed_idx, remaining_idx[idx])
        weights = np.append(weights, pmf[idx])
        remaining_idx =  np.delete(remaining_idx, idx)
        
    return observed_idx, weights 