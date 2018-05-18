all:
	g++ -D_DEBUG -Wall -O3 -std=c++11 ./src/main/train.cpp ./src/util/util.cpp ./src/sample/sample.cpp \
		./src/model/model.cpp ./src/model/model_factory.cpp ./src/fm/fm.cpp ./src/fm/fm_fengchao.cpp ./src/lr/lr.cpp \
		./src/lr/lr_momentum.cpp ./src/lr/lr_nag.cpp ./src/lr/lr_adagrad.cpp ./src/lr/lr_rmsprop.cpp \
		./src/lr/lr_adam.cpp ./src/lr/lr_adadelta.cpp \
		-I ./include/util -I ./include/sample -I ./include/model -I ./include/fm -I ./include/lr -o train
	g++ -Wall -O3 -std=c++11 ./src/main/search.cpp ./src/util/util.cpp ./src/sample/sample.cpp \
		./src/model/model.cpp ./src/model/model_factory.cpp ./src/fm/fm.cpp ./src/fm/fm_fengchao.cpp ./src/lr/lr.cpp \
		./src/lr/lr_momentum.cpp ./src/lr/lr_nag.cpp ./src/lr/lr_adagrad.cpp ./src/lr/lr_rmsprop.cpp \
		./src/lr/lr_adam.cpp ./src/lr/lr_adadelta.cpp \
		-I ./include/util -I ./include/sample -I ./include/model -I ./include/fm -I ./include/lr -o search
clean:
	-rm train search
