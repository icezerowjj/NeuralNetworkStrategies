# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:00:48 2017

@author: zerow
"""

# Importing the libraries
import numpy as np
import pandas as pd
import xlrd
import time
import openpyxl
import matplotlib.pyplot as plt
import statsmodels.api as sm
# sklearn
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

#Function to read excel data
def readExcel(file_name):
    raw_data = pd.read_excel(file_name)
    dates_temp = raw_data.iloc[:, 0]
    dates = ExtractString(pd.DataFrame(dates_temp))
    raw_data.index = dates
    return raw_data
    
# Function to get the date from a vector of stampdates
def ExtractString(input_vec):
    out_vec = []
    num = input_vec.size
    for i in xrange(num):
        temp = input_vec.iloc[i, 0].timetuple()
        year = temp.tm_year
        t_month = temp.tm_mon
        t_day = temp.tm_mday
        if t_month < 10:
            month = '0' + str(t_month)
        else:
            month = str(t_month)
                
        if t_day < 10:
            day = '0' + str(t_day)
        else:
            day = str(t_day)
        
        out_vec.append(str(year) + '-' + month+ '-' + day)
    
    return out_vec
    
# Calculate the return of the stock price
def get_stkReturn(stock_price, n, max_days):
    '''
    Calculate log return ratio with close price
    '''
    #Get the returns of the stocks
    new_index = stock_price.iloc[n:,:].index
    days_num = np.shape(stock_price)[0] - n
    stock_num = np.shape(stock_price)[1]
    stock_return = pd.DataFrame(np.zeros([days_num , stock_num]), index = new_index)
    for i in xrange(days_num):
        stock_return.iloc[i,:] = np.array(stock_price.iloc[i + n,:]/stock_price.iloc[i,:] - 1)
    # Drop the first max_days num of days observation
    stock_return = stock_return.iloc[max_days - n:,:]
    return stock_return
    
#Get the market return
def get_mktReturn(mkt_price, n, max_days, test_dates):
    days_num = np.shape(mkt_price)[0] - n
    mkt_return = np.zeros(days_num)
    for i in xrange(days_num):
        mkt_return[i] = mkt_price[i + n] / mkt_price[i] - 1
    # Drop the first max_days num of days observation
    mkt_return = pd.DataFrame(mkt_return)
    mkt_return = mkt_return.iloc[max_days - n:,:]
    mkt_return = mkt_return.set_index(test_dates)
    return mkt_return
    
# Function to get the label of whether outperform
def getLabel(ret_dict, mkt_ret):
    stk_ret = ret_dict[0]
    new_index = stk_ret.iloc[1:,:].index
    days_num = np.shape(stk_ret)[0] - 1
    stock_num = np.shape(stk_ret)[1]
    Y = pd.DataFrame(np.zeros([days_num , stock_num]), index = new_index)
    for s in xrange(stock_num):
        for d in xrange(days_num - 1):
            if stk_ret.iloc[d + 1, s] >= mkt_ret.iloc[d + 1, 0]:
                Y.iloc[d, s] = 1
            else:
                Y.iloc[d, s] = 0
    return Y
    
# Function to get the prediction of the next day
def getPrediction(X_temp_train, Y_temp_train, X_temp_test, batch_size, nb_epoch, node_hide):
    num_fet = np.shape(X_temp_train)[1]
    # Standardization for X
    sc = StandardScaler()
    X_temp_train = sc.fit_transform(X_temp_train)
    # Initialising the ANN
    classifier = Sequential()   
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = node_hide, init = 'uniform', activation = 'tanh', input_dim = num_fet)) 
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = node_hide, init = 'uniform', activation = 'tanh'))
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Fitting the ANN to the Training set
    classifier.fit(X_temp_train, Y_temp_train, batch_size, nb_epoch)
    # Prediction
#    X_temp_test = sc.fit_transform(X_temp_test)
    Y_temp_pred = classifier.predict(X_temp_test)
    return Y_temp_pred
    
# Function to estimate Kalman beta
def getKalmanBeta(stk_close, mkt_close, mkt_ret, rf, A, K0, Q, R, P0):
    # Construct the Kalman filter for each stock every day
    kal_stk_ret = get_stkReturn(stk_close, 1, 2)
    kal_mkt_ret = get_mktReturn(mkt_close, 1, 2, kal_stk_ret.index)
    stock_num = np.shape(kal_stk_ret)[1]
    # Regression to initialize the beta
    reg_days = len(kal_mkt_ret) - len(mkt_ret)
    reg_X = kal_mkt_ret.iloc[:reg_days, 0]
    init_beta = []
    for j in xrange(stock_num):
        reg_X = sm.add_constant(reg_X)
        reg_Y = kal_stk_ret.iloc[:reg_days, j]
        ols_model = sm.OLS(reg_Y, reg_X)
        temp_results = ols_model.fit()
        init_beta.append(temp_results.params[0])
    new_index = kal_mkt_ret.iloc[reg_days:, 0].index
    new_stk_ret = kal_stk_ret.iloc[reg_days:, :]
    new_mkt_ret = kal_mkt_ret.iloc[reg_days:, 0]
    rf = pd.DataFrame(rf)
    rf = rf.fillna(value = 0)
    new_rf = rf.iloc[reg_days + 2:, 0]
    # Now do Kalman filter
    kal_beta = pd.DataFrame(np.zeros([len(new_index), stock_num]), index = new_index)
    for s in xrange(stock_num):
        beta_0 = init_beta[s]
        beta_minus = A * beta_0
        P_minus = A * A * P0 + Q
        beta_hat = beta_0
        P_hat = P0
        K = K0
        kal_beta.iloc[0, s] = beta_hat
        for d in xrange(len(new_index)):
            # Calculate the parameters
            beta_minus = A * beta_hat
            P_minus = A * A * P_hat + Q
            epslon = new_stk_ret.iloc[d, s] - new_rf[d] - beta_minus * (new_mkt_ret[d] - new_rf[d])
            # Update the params            
            K = (P_minus*(new_mkt_ret[d] - new_rf[d])) / (P_minus * (new_mkt_ret[d] - new_rf[d])**2 + R)
            beta_hat = beta_minus + K * epslon
            P_hat = (1 - K*(new_mkt_ret[d] - new_rf[d]))**2 * P_minus + K**2 * R
            # Store the estimate
            kal_beta.iloc[d, s] = beta_hat
    return kal_beta

# Function to merge the return as features
def mergeretX(stk_close, fet_days):
    ret_dict = dict()
    max_days = max(fet_days)
    fet_num = len(fet_days)
    # Initialize with the first array
    test_dates = get_stkReturn(stk_close, fet_days[0], max_days).index
    ret_np = get_stkReturn(stk_close, fet_days[0], max_days).values
    for i in xrange(fet_num):
        ret_dict[i] = get_stkReturn(stk_close, fet_days[i], max_days)
        if i > 0:
            ret_np = np.hstack((ret_np, ret_dict[i].values))
    # This is also the input features for our training
    ret_pd = pd.DataFrame(ret_np, index = test_dates)
    # Now prepare for the input dataset every day
    X_dict = dict()
    for i in xrange(len(test_dates)):
        for j in xrange(fet_num):
            if j == 0:
                temp_pd = pd.DataFrame(ret_dict[j].iloc[i, :])
            else:
                temp_pd = pd.concat([temp_pd, pd.DataFrame(ret_dict[j].iloc[i, :])], axis = 1)
        X_dict[i] = pd.DataFrame(temp_pd)
    return ret_pd, ret_dict, X_dict

# Function to merge all the features
def mergeFeatures(X_return, kal_beta):
    X_dict = dict()
    num_days = len(X_return)
    for d in xrange(num_days):
        X_dict[d] = pd.concat([X_return[d], pd.DataFrame(kal_beta.iloc[d, :])], axis = 1)
    return X_dict

# Function to do asset allocation
def getPosition(prob, n):
    days_num = len(prob)
    stock_num = np.shape(prob)[1]
    new_index = prob.index
    pos = pd.DataFrame(np.zeros([days_num, stock_num]), index = new_index)
    for d in xrange(days_num):
        temp_prob = prob.iloc[d, :]
        # Max n probs
        max_prob = temp_prob.argsort()[-n:][::-1]
        pos.iloc[d, max_prob] = 1.0 / n
        # Min n probs
        min_prob = temp_prob.argsort()[:n]
        pos.iloc[d, min_prob] = -1.0 / n
    return pos
    
# Function to get the daily return of strategy
def getDailyReturn(pos, daily_stk_ret):
    daily_return = []
    for d in xrange(len(pos)):
        temp_ret = pos.iloc[d, :].dot(daily_stk_ret.iloc[d, :])
        daily_return.append(temp_ret)
    return daily_return
    
# Function to output the results of backtesting
def getResults(daily_return, rf):
    daily_return = pd.DataFrame(daily_return)
    daily_return = daily_return.fillna(value = 0)
    d_mean = np.mean(daily_return)[0]
    d_std = np.std(daily_return)[0]
    Sharpe = (d_mean - np.mean(rf)) / d_std
    log_ret = np.log(daily_return + 1)
    cum_ret = np.exp(np.cumsum(log_ret)) - 1
    # Plot the cumulated returns
    plt.plot(cum_ret)
    plt.title('Performance')
    plt.ylabel('Cumulated Returns')
    plt.xlabel('Time')
    plt.show()
    print "The Sharpe ratio of this %d-day strategy is%8.2f\n" % (len(daily_return), Sharpe)
    return Sharpe, cum_ret


# Main function
    
if __name__ == "__main__":

    #Start time
    start_time = time.clock()
    
    '''
    Read in the data 
    '''   
    #Read the file into python price
    SP_file = "Last_Price.xlsx"
    raw_data = readExcel(SP_file)
    # Select part of the dataset as test sample
    raw_data = raw_data.iloc[-250:, :]
    raw_data = raw_data.drop(raw_data.columns[1:48], axis=1) 
    
    # Split the dataset into SP stocks, the index and Rf
    #Get the price matrix and market index
    stk_close = raw_data.drop(["Date","Pm","Rf"], axis = 1)
    mkt_close = raw_data.iloc[:,-2]
    rf = raw_data.iloc[:,-1]
    # Convert the dataset into a new one to be learned
#    fet_days = list([1, 2, 5, 9, 12, 26, 50, 60, 80, 100, 150, 200, 250]) 
#    fet_days = list([1, 2, 5, 12, 26, 60, 120])
    fet_days = list([1, 2, 5])
    ret_pd, ret_dict, X_return = mergeretX(stk_close, fet_days)
    # Generate the next-day market return dataset
    mkt_ret = get_mktReturn(mkt_close, 1, max(fet_days), ret_pd.index)
    # Create the dataset for Y
    Y = getLabel(ret_dict, mkt_ret)

    # Kalman filter for beta estimation
    # Kalman params
    A = 1.0
    K0 = 1
    Q = 0.002
    R = 0.003
    P0 = 0.0025
    # Calculate betas
    kal_beta = getKalmanBeta(stk_close, mkt_close, mkt_ret, rf, A, K0, Q, R, P0)
    # Merge with X_return features
    X_dict = mergeFeatures(X_return, kal_beta)
    del(X_dict[len(X_dict) - 1])
    
    # ANN Modeling
    # Paras for ANN
    batch_size = 10
    nb_epoch = 50
    node_hide = 4
    
    # Start backtesting
    for d in xrange(len(X_dict) - 2):
        print "Now train the NN for Day%d" %(d)
        X_temp_train = X_dict[d].values
        Y_temp_train = np.array(Y.iloc[d, :])
        X_temp_test = X_dict[d + 1].values
        pred = getPrediction(X_temp_train, Y_temp_train, X_temp_test, batch_size, nb_epoch, node_hide)
        if d == 0:
            stock_prob = pd.DataFrame(pred)
        else:
            stock_prob = pd.concat([stock_prob, pd.DataFrame(pd.DataFrame(pred))], axis = 1)
    stock_prob = pd.DataFrame(np.transpose(stock_prob.values))
    stock_prob = stock_prob.fillna(value = 0)

    # Get positions
    hold = 5
    pos = getPosition(stock_prob, hold)
    daily_stk_ret = ret_pd.iloc[:, :np.shape(stk_close)[1]]
    daily_return = getDailyReturn(pos, daily_stk_ret)
    Sharpe, cum_ret = getResults(daily_return, rf)
        
    
    #End time
    end_time = time.clock()
    print 'Time used is: ', end_time - start_time, ' seconds\n'    