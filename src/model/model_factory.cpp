#include "model_factory.h"
using namespace util;

namespace model {

  Model* ModelFactory::_p_model = NULL;
  std::string ModelFactory::_model_type = "";
  DataSet* ModelFactory::_p_train_dataset = NULL;
  DataSet* ModelFactory::_p_test_dataset = NULL;
  hash2index_type ModelFactory::_f_hash2index;
  index2hash_type ModelFactory::_f_index2hash;
  f_index_type ModelFactory::_f_size = 0;

  void ModelFactory::load_dataset(const path_type& train_path, const path_type& test_path) {
    // train dataset
    std::cout << std::endl;
    std::cout << "====> step load_train_dataset......" << std::endl;
    _p_train_dataset = new DataSet(train_path, TRAIN_DATASET_NAME);
    _p_train_dataset->init(_f_hash2index, _f_index2hash, _f_size);
    std::cout << "  --> feature-size:" << _f_size << std::endl;
    // test dataset
    std::cout << std::endl;
    std::cout << "====> step load_test_dataset......" << std::endl;
    _p_test_dataset = new DataSet(test_path, TEST_DATASET_NAME);
    _p_test_dataset->init(_f_hash2index, _f_index2hash, _f_size);
  }

  Model* ModelFactory::get_lr_instance(const std::string& model_type) {
    if (_model_type != model_type) {
      if (!_p_model) delete _p_model;  // delete the last model if needed
      if (model_type == LR_MODEL) { // lr
        _p_model = new LRModel(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size, model_type);
      } else if (model_type == LR_M_MODEL) { // lr momentum
        _p_model = new LRMomentumModel(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size, model_type);
      } else if (model_type == LR_NAG_MODEL) { // lr nag
        _p_model = new LRNAGModel(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size, model_type);
      } else if (model_type == LR_AG_MODEL) { // lr adagrad
        _p_model = new LRAdagradModel(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size, model_type);
      } else if (model_type == LR_RMS_MODEL) { // lr rmsprop
        _p_model = new LRRMSPropModel(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size, model_type);
      } else if (model_type == LR_ADAM_MODEL) { // lr adam
        _p_model = new LRAdamModel(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size, model_type);
      } else if (model_type == LR_ADAD_MODEL) { // lr adadelta
        _p_model = new LRAdadeltaModel(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size, model_type);
      } else if (model_type == FM_MODEL) { // fm
        _p_model = new FMModel(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size, model_type);
      } else if (model_type == FM_FC_MODEL) { // fm fengchao
        _p_model = new FMFengchaoModel(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size, model_type);
      } else if (model_type == FFM_FC_MODEL) { // ffm fengchao
        _p_model = new FFMFengchaoModel(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size, model_type);
      }
      _model_type = model_type; // model type rewrite
    }
    return _p_model;
  }

} // namespace model
