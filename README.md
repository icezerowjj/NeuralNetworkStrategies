# Neural-Network-Strategies
This project utilizes models of neural network to develop trading strategies based on US stock past performance
Last_price.xlsx is the original data file for stock market including those stocks in S&P 500 as well as the S&P index data with risk-free rate in the last two columns. 

Strats_DNN.py is the python code for the implementation of DNN stragety which trains a DNN every day. This incorporates returns of different time intervals and Kalman Filter beta as features. And whether a certain stock outperforms the S&P index is the output label.
