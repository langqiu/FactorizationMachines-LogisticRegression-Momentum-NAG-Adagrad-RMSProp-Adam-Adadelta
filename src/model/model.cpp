#include "model.h"
using namespace util;

namespace model {

  bool sort_by_param(const param_evaluate_type& l, const param_evaluate_type& r) {
    return std::abs(l.first) > std::abs(r.first);
  }

  Model::Model(DataSet* p_train_dataset, DataSet* p_test_dataset,
      const hash2index_type& f_hash2index, const index2hash_type& f_index2hash,
      const f_index_type& f_size, const std::string& model_type) :
    _iter_size(0), _batch_size(0),
    _curr_batch(0), _p_train_dataset(p_train_dataset),
    _p_test_dataset(p_test_dataset), _f_hash2index(f_hash2index),
    _f_index2hash(f_index2hash), _f_size(f_size),
    _model_type(model_type) {}

  Model::~Model() {
    if (_p_train_dataset) _p_train_dataset = NULL;
    if (_p_test_dataset) _p_test_dataset = NULL;
  }

  void Model::init(const size_t& iter_size, const size_t& batch_size,
      const param_type& alpha, const param_type& lambda,
      const param_type& beta_1, const param_type& beta_2,
      const param_type& fm_dims) {
  }

  void Model::train() {
    /*
     * use "this" pointer to avoid rewriting func in child class
     */
    _print_step("train");
    size_t data_size = _p_train_dataset->get_size(); // get size of train dataset
    time_type time_start = time_now(); // time before train
    for (size_t iter=0; iter<_iter_size; iter++) { // each iteration
      size_t batch_start = 0; // start from the begining of dataset
      while (batch_start < data_size) { // each batch
        _curr_batch++;
        size_t batch_end = batch_start + _batch_size; // calculate end
        if (batch_end >= data_size) batch_end = data_size; // end for the last batch
        this->_forward(batch_start, batch_end, _p_train_dataset);
        this->_backward(batch_start, batch_end);
        this->_update();
        batch_start = batch_end; // update start of the next batch
        /*
        if (_curr_batch % TRAIN_PRINT_INTERVAL == 0) {
          _print_mini_batch(_curr_batch);
          _cal_model_logloss();
          _cal_model_mse();
          _predict_dataset(_p_test_dataset); // predict test dataset
          _p_test_dataset->evaluate(); // evaluate test dataset
        }
        */
      }
    }
    time_type time_end = time_now(); // time after train
    _print_time_cost(time_end, time_start);
    this->_print_model_param(); // print model
  }

  void Model::evaluate() {
    _print_step("evaluate");
    _predict_dataset(_p_train_dataset); // predict train dataset
    _predict_dataset(_p_test_dataset); // predict test dataset
    _p_train_dataset->evaluate(); // evaluate train dataset
    _p_test_dataset->evaluate(); // evaluate test dataset
  }

  void Model::_forward(const size_t& l, const size_t& r, DataSet* p_data) {
  }

  void Model::_backward(const size_t& l, const size_t& r) {
  }

  void Model::_update() {
  }

  void Model::_print_model_param() {
  }

  void Model::_predict_dataset(DataSet* p_data) {
    /*
     * use "this" pointer to avoid rewriting func in child class
     */
    size_t data_size = p_data->get_size(); // get the size of dataset
    this->_forward(0, data_size, p_data); // predict the dataset
  }

  void Model::_cal_model_mse() {
#ifdef _DEBUG
    _predict_dataset(_p_train_dataset); // predict the dataset
    std::cout << "  --> mse " << _p_train_dataset->cal_mse() << std::endl;
#endif
  }

  void Model::_cal_model_logloss() {
#ifdef _DEBUG
    _predict_dataset(_p_train_dataset); // predict the dataset
    std::cout << "  --> logloss " << _p_train_dataset->cal_logloss() << std::endl;
#endif
  }

  void Model::_print_step(const std::string& step) {
#ifdef _DEBUG
    std::cout << std::endl;
    std::cout << "====> step " << _model_type << "-" << step << "......" << std::endl;
#endif
  }

  void Model::_print_mini_batch(const size_t& batch) {
#ifdef _DEBUG
    std::cout << std::endl;
    std::cout << "  --> mini-batch " << batch << std::endl;
#endif
  }

  void Model::_print_time_cost(time_type time_end, time_type time_start) {
#ifdef _DEBUG
    std::cout << "  --> time-cost " << time_diff(time_end, time_start) << std::endl; // time cost
#endif
  }

} // namespace model
