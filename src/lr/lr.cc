#include "lr.h"
using namespace util;

namespace lr {

  bool sort_by_param(const param_evaluate_type& l, const param_evaluate_type& r) {
    return std::abs(l.first) > std::abs(r.first);
  }

  void print_step(const std::string& step) {
#ifdef _DEBUG
    std::cout << std::endl;
    std::cout << "====> step " << step << "..." << std::endl;
#endif
  }

  void print_iteration(const size_t& iter) {
#ifdef _DEBUG
    std::cout << "  --> iteration " << iter << std::endl;
#endif
  }

  LR::LR(DataSet* p_train_dataset, DataSet* p_test_dataset,
      const hash2index_type& f_hash2index, const index2hash_type& f_index2hash,
      const f_index_type& f_size) :
    _alpha(0.0), _lambda(0.0),
    _beta_1(0.0), _beta_2(0.0),
    _iter_size(0), _batch_size(0),
    _curr_batch(0), _p_train_dataset(p_train_dataset),
    _p_test_dataset(p_test_dataset), _f_hash2index(f_hash2index),
    _f_index2hash(f_index2hash), _f_size(f_size) {}

  LR::~LR() {
    if (_p_train_dataset) _p_train_dataset = NULL;
    if (_p_test_dataset) _p_test_dataset = NULL;
  }

  void LR::init(const size_t& iter_size, const size_t& batch_size,
      const param_type& alpha, const param_type& lambda,
      const param_type& beta_1, const param_type& beta_2) {
    print_step("init...");
    // init model parameters
    _theta.clear();
    _theta.reserve(_f_size);
    for (size_t i=0; i<_f_size; i++) {
      _theta.push_back(0.0);
    }
    _theta_new = _theta;
    // init hyperparameters
    _alpha = alpha;
    _lambda = lambda;
    _beta_1 = beta_1;
    _beta_2 = beta_2;
    // init train parameters
    _iter_size = iter_size;
    _batch_size = batch_size;
    _curr_batch = 0;
    // init extra vectors
    _gradient = _theta;
    _delta = _theta;
    for (size_t i=0; i<_f_size; i++) {
      _delta[i] = 0.001;
    }
    _first_moment_vector = _theta;
    _second_moment_vector = _theta;
  }

  void LR::train() {
    /*
     * use "this" pointer to avoid rewriting train func
     */
    print_step("train...");
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
      }
      if (iter % MSE_INTERVAL == 0) {
        print_iteration(iter);
        _cal_model_logloss();
        //_cal_model_mse();
      }
    }
    time_type time_end = time_now(); // time after train
    time_diff(time_end, time_start); // print time cost
    _print_model_param(); // print model
  }

  void LR::evaluate() {
    print_step("evaluate");
    _predict_dataset(_p_train_dataset); // predict train dataset
    _predict_dataset(_p_test_dataset); // predict test dataset
    _p_train_dataset->evaluate(); // evaluate train dataset
    _p_test_dataset->evaluate(); // evaluate test dataset
  }

  void LR::_forward(const size_t& l, const size_t& r, DataSet* p_data) {
    /*
     * predict the sample in [l, r) of the dataset p_data
     */
#ifdef _DEBUG
    if (_curr_batch == 1) std::cout << "lr forward" << std::endl;
#endif
    auto& data = p_data->get_data(); // get dataset
    for (size_t i=l; i<r; i++) { // loop each sample in this batch
      score_type product = 0;
      auto& curr_sample = data[i]; // current sample
      for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) { // do product
        auto& curr_f = curr_sample._sparse_f_list[k];
        product += _theta[curr_f.first] * curr_f.second;
      }
      product += _theta[_f_size-1]; // bias
      curr_sample._score = 1 / (1 + ::exp(-1 * product)); // sigmoid
    }
  }

  void LR::_backward(const size_t& l, const size_t& r) {
    /*
     * theta(t) = theta(t-1) * (1 - alpha * lambda / m)
     *    + alpha * SUM( (y(i) - y'(i)) * x(i,j) ) / m
     * Only update those theta that show up in samples of the current batch,
     *    or it is wrong and costs a lot of time.
     *    2 seconds (78 seconds if not) to train 1 million samples with SGD
     */
#ifdef _DEBUG
    if (_curr_batch == 1) std::cout << "lr backward" << std::endl;
#endif
    auto& data = _p_train_dataset->get_data(); // get train dataset
    std::unordered_set<f_index_type> theta_updated; // record theta in BGD
    score_type factor = -1 * _alpha * _lambda / (r - l); // L2 delta factor
    for (size_t i=l; i<r; i++) {
      auto& curr_sample = data[i]; // current sample
      for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) { // loop features
        auto& curr_f = curr_sample._sparse_f_list[k]; // current feature
        _theta_new[curr_f.first] += _alpha * (curr_sample._label - curr_sample._score)
          * curr_f.second / (r -l); // delta from the logloss
        // delta from L2
        if (r - l == 1) { // SGD
          _theta_new[curr_f.first] += _theta[curr_f.first] * factor;
        } else if (theta_updated.find(curr_f.first) == theta_updated.end()) {
          _theta_new[curr_f.first] += _theta[curr_f.first] * factor; // BGD
          theta_updated.insert(curr_f.first); // record the index of theta that has showed up
        }
      }
      _theta_new[_f_size-1] += _alpha * (curr_sample._label - curr_sample._score)
        / (r - l); // delta from logloss for bias
    }
    _theta_new[_f_size-1] += _theta[_f_size-1] * factor; // delta from L2 for bias
  }

  void LR::_update() {
#ifdef _DEBUG
    if (_curr_batch == 1) std::cout << "lr update" << std::endl;
#endif
    _theta = _theta_new;
  }

  void LR::_print_model_param() {
#ifdef _DEBUG
    /*
     * print the most useful, useless features and model bias
     */
    std::vector<param_evaluate_type> parameters;
    for (size_t i=0; i<_f_size; i++) {
      parameters.push_back(param_evaluate_type(_theta[i], i));
    }
    sort(parameters.begin(), parameters.end(), sort_by_param); // sort by |theta(j)|
    std::cout << "############ model-top-params ############" << std::endl;
    for (size_t i=0; i<parameters.size() && i<MODEL_TOP_PARAMS; i++) {
      std::cout << _f_index2hash[parameters[i].second] << " " << parameters[i].first << std::endl;
    }
    for (int i=parameters.size()-1; i>=0 && parameters.size()-i<MODEL_TOP_PARAMS; i--) {
      std::cout << _f_index2hash[parameters[i].second] << " " << parameters[i].first << std::endl;
    }
    std::cout << "bias: " << _theta[_f_size-1] << std::endl;
#endif
  }

  void LR::_predict_dataset(DataSet* p_data) {
    size_t data_size = p_data->get_size(); // get the size of dataset
    _forward(0, data_size, p_data); // predict the dataset
  }

  void LR::_cal_model_mse() {
#ifdef _DEBUG
    _predict_dataset(_p_train_dataset); // predict the dataset
    std::cout << "  --> mse " << _p_train_dataset->cal_mse() << std::endl;
#endif
  }

  void LR::_cal_model_logloss() {
#ifdef _DEBUG
    _predict_dataset(_p_train_dataset); // predict the dataset
    std::cout << "  --> logloss " << _p_train_dataset->cal_logloss() << std::endl;
#endif
  }

} // namespace lr
