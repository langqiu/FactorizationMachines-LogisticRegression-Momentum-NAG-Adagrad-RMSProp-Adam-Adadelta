all:
	g++ -D_DEBUG -Wall -O3 -std=c++11 ./src/train.cc ./src/lr/lr_factory.cc \
		./src/lr/lr_adadelta.cc ./src/lr/lr_adam.cc ./src/lr/lr_rmsprop.cc \
		./src/lr/lr_adagrad.cc ./src/lr/lr_nag.cc ./src/lr/lr_momentum.cc \
		./src/lr/lr.cc ./src/lr/sample.cc ./src/util/util.cc -I ./src/lr -I ./src/util -o ./bin/train
	g++ -Wall -O3 -std=c++11 ./src/search.cc ./src/lr/lr_factory.cc \
		./src/lr/lr_adadelta.cc ./src/lr/lr_adam.cc ./src/lr/lr_rmsprop.cc \
		./src/lr/lr_adagrad.cc ./src/lr/lr_nag.cc ./src/lr/lr_momentum.cc \
		./src/lr/lr.cc ./src/lr/sample.cc ./src/util/util.cc -I ./src/lr -I ./src/util -o ./bin/search

.PHONY : clean

clean:
	-rm ./bin/train ./bin/search
