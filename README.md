# TitanicSurvivalPrecidtion-CUDAProgramming
Built Titanic survival prediction using logistic regression and employed cuda practices such as device synchronization, kernel execution, device and host memory management leveraging parallel processing and reducing computation time for training by 80 percent.

## Kaggle Dataset and Competition
https://www.kaggle.com/c/titanic

## Description
* Built Titanic survival prediction using logistic regression and employed cuda practices such as device synchronization,
kernel execution, device and host memory management leveraging parallel processing and reducing computation time for
training by 80 percent.
* Designed a custom C++ library utilizing CUDA to execute matrix multiplications and advanced loss functions; improved
computational speed by 30%, enabling quicker model training and evaluation cycles for machine learning projects.

## Pre-requisites
* The device should have GPU capabilities and cuda should be present.
* You can replace kaggle titanic data with other classification data, but currently confusion matrix is supported only for binary classes, just comment or modify the code and the script can be used for any classification problem.
* Make sure Target Value is the second column in csv file. 

## How to run
* just execute .\main.exe then provide training data file path