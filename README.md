# Logistic Regression

What is it?
----------------------------------------------------------------------
It's a logistic regression tool which supports BGD/SGD/Mini-batch and common gradient descent
optimization algorithms, including Momentum/NAG/Adagrad/RMSProp/Adam/Adadelta.


How to use it?
----------------------------------------------------------------------
Compile the project by running "make", there are two ways to use it:

1.  choose a variant of the LR model and tune it.

    1)  start running.

        ./bin/train data/airbnb_train_dataset data/airbnb_test_dataset

    2)  choose a model and input hyper parameters.

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

    3)  wait for training and evaluating.

        ############ train-dataset ############

          --> auc: 0.835583

          --> logloss: 0.478467

          --> mse: 0.157758

        ############ test-dataset ############

          --> auc: 0.835003

          --> logloss: 0.478853

          --> mse: 0.157709

    4)  choose a new model or tune this model again.

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        new model[M] or tune parameters[P]?

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        P

2.  search the best combination of hyper parameters for each variant of model.

    1)  start running.

        ./bin/search data/airbnb_train_dataset data/airbnb_test_dataset

    2)  wait for training each model to search the best combination of hyper parameters.

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        please input model type (lr/lr_m/lr_nag/lr_ag/lr_rms/lr_adam/lr_adad):

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        lr

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        auc: 0.833465

        alpha: 0.01

        lambda: 0

        beta_1: 0

        beta_2: 0

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        logloss: 0.464706

        alpha: 0.03

        lambda: 0

        beta_1: 0

        beta_2: 0

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        please input model type (lr/lr_m/lr_nag/lr_ag/lr_rms/lr_adam/lr_adad):

        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


What is the meaning of each hyper parameter?
----------------------------------------------------------------------
There are 6 hyper parameters which need to be initialized.

1.  iter:     number of iteration

2.  batch:    number of samples in each batch, let it be 1 if you need SGD

3.  alpha:    learning rate

4.  lambda:   L2 factor

5.  beta_1:   decay rate, different meaning in different models

    @Momentum   v(t) = beta_1 * v(t-1) + alpha * (-1 * g(t))

    @NAG        same as Momentum

    @Adam       m(t) = beta_1 * m(t-1) + (1 - beta_1) * (-1 * g(t))

    @Adadelta   E[g(t)^2] = beta_1 * E[g(t-1)^2] + (1 - beta_1) * g(t)^2

6.  beta_2:   decay rate, different meaning in different models

    @Adam       v(t) = beta_2 * v(t-1) + (1 - beta_2) * g(t)^2

    @RMSProp    same as Adam


Overview of optimization algorithms
----------------------------------------------------------------------
I highly recommend this blog by Sebastian Ruder.

http://ruder.io/optimizing-gradient-descent/index.html#challenges


Code description
----------------------------------------------------------------------
util.h          common functions and typedef

sample.h        dataset class, initialized in lr_factory.h

lr.h            base class of lr

lr_[model].h    child class, need to rewrite _forward/_backward/_update

lr_factory.h    construct an lr instance

train.cc        choose a model and tune hyper parameters

search.cc       search the best hyper parameters for certain model
