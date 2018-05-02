all:
	g++ -D_DEBUG -Wall -O3 -std=c++11 ./src/train.cpp ./src/lr/lr.cpp ./src/lr/lr_momentum.cpp \
		./src/lr/lr_nag.cpp ./src/lr/lr_adagrad.cpp ./src/lr/lr_rmsprop.cpp ./src/lr/lr_adam.cpp \
		./src/lr/lr_adadelta.cpp ./src/lr/lr_factory.cpp ./src/lr/sample.cpp ./src/util/util.cpp \
		-I ./include/lr -I ./include/util -o train
	g++  -Wall -O3 -std=c++11 ./src/search.cpp ./src/lr/lr.cpp ./src/lr/lr_momentum.cpp \
		./src/lr/lr_nag.cpp ./src/lr/lr_adagrad.cpp ./src/lr/lr_rmsprop.cpp ./src/lr/lr_adam.cpp \
		./src/lr/lr_adadelta.cpp ./src/lr/lr_factory.cpp ./src/lr/sample.cpp ./src/util/util.cpp \
		-I ./include/lr -I ./include/util -o search

clean:
	-rm train search
