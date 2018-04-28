# logistic_regression

What is this?
----------------------------------------------------------------------
It's a logistic regression tool which supports BGD/SGD/Mini-batch and common gradient descent
optimization algorithms, including momentum/NAG/Adagrad/RMSProp/Adam/Adadelta.

How to use it?
----------------------------------------------------------------------
"make" to build the project, there are two ways to use it:

1.  choose a variant of the LR model and tune it.

    1)  start

        ./bin/train train-dataset test-dataset

    2)  choose model and input hyper parameters.

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        please input [M] to create a new model:
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        M
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        please input model type (lr/lr_m/lr_nag/lr_ag/lr_rms/lr_adam/lr_adad):
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        lr
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        new model[M] or tune parameters[P]?
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        P
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        please input params separated by comma (iter,batch,alpha,lambda,beta_1,beta_2):
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        1,1,0.03,0.01,0,0

    3)  wait for training and evaluating

        ############ train-dataset ############
          --> auc: 0.835583
          --> logloss: 0.478467
          --> mse: 0.157758
        ############ test-dataset ############
          --> auc: 0.835003
          --> logloss: 0.478853
          --> mse: 0.157709

    4)  choose new model or tune this model again

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        new model[M] or tune parameters[P]?
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        P

2.  search the best combination of hyper parameters for each variant.
    ./bin/search train-dataset test-dataset

What is the meaning of each hyper parameters?
----------------------------------------------------------------------
There are 6 hyper parameters need to be initialized.

1.  iter:     number of iteration
2.  batch:    number of samples in each batch, let it be 1 if you need SGD
3.  alpha:    learning rate
4.  lambda:   L2 factor
5.  beta_1:   it has different meaning in different model
6.  beta_2:   it has different meaning in different model

Model Detail
----------------------------------------------------------------------
