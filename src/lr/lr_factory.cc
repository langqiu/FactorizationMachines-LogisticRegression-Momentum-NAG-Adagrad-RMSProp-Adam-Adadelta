#include "lr_factory.h"
using namespace util;

namespace lr {

  LR* LRFactory::_p_lr = NULL;
  std::string LRFactory::_model_type = "";
  DataSet* LRFactory::_p_train_dataset = NULL;
  DataSet* LRFactory::_p_test_dataset = NULL;
  hash2index_type LRFactory::_f_hash2index;
  index2hash_type LRFactory::_f_index2hash;
  f_index_type LRFactory::_f_size = 0;

  void LRFactory::load_dataset(const path_type& train_path, const path_type& test_path) {
    // train dataset
    print_step("load_train_dataset");
    _p_train_dataset = new DataSet(train_path, TRAIN_DATASET_NAME);
    _p_train_dataset->init(_f_hash2index, _f_index2hash, _f_size);
    std::cout << "  --> feature-size:" << _f_size << std::endl;
    // test dataset
    print_step("load_test_dataset");
    _p_test_dataset = new DataSet(test_path, TEST_DATASET_NAME);
    _p_test_dataset->init(_f_hash2index, _f_index2hash, _f_size);
  }

  LR* LRFactory::get_lr_instance(const std::string& model_type) {
    if (_model_type != model_type) {
      if (!_p_lr) { delete _p_lr; } // delete the last model if needed
      if (model_type == LR_MODEL) { // lr
        _p_lr = new LR(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size);
      } else if (model_type == LR_M_MODEL) { // lr momentum
        _p_lr = new LRMomentum(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size);
      } else if (model_type == LR_NAG_MODEL) { // lr nag
        _p_lr = new LRNAG(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size);
      } else if (model_type == LR_AG_MODEL) { // lr adagrad
        _p_lr = new LRAdagrad(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size);
      } else if (model_type == LR_RMS_MODEL) { // lr rmsprop
        _p_lr = new LRRMSProp(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size);
      } else if (model_type == LR_ADAM_MODEL) { // lr adam
        _p_lr = new LRAdam(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size);
      } else if (model_type == LR_ADAD_MODEL) { // lr adadelta
        _p_lr = new LRAdadelta(_p_train_dataset, _p_test_dataset,
            _f_hash2index, _f_index2hash, _f_size);
      }
      _model_type = model_type; // model type rewrite
    }
    return _p_lr;
  }

} // namespace lr
