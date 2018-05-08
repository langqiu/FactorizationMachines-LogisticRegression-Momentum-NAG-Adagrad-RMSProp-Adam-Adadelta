#include "lr.h"
using namespace util;

namespace model {

  LRModel::LRModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
      const hash2index_type& f_hash2index, const index2hash_type& f_index2hash,
      const f_index_type& f_size, const std::string& model_type) :
    Model(p_train_dataset, p_test_dataset,
        f_hash2index, f_index2hash, f_size, model_type),
    _alpha(0.0), _lambda(0.0),
    _beta_1(0.0), _beta_2(0.0) {}

  void LRModel::init(const size_t& iter_size, const size_t& batch_size,
      const param_type& alpha, const param_type& lambda,
      const param_type& beta_1, const param_type& beta_2,
      const param_type& fm_dims) {
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

  void LRModel::_forward(const size_t& l, const size_t& r, DataSet* p_data) {
    /*
     * predict the sample in [l, r) of the dataset p_data
     */
    if (_curr_batch == 1) _print_step("forward");
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

  void LRModel::_backward(const size_t& l, const size_t& r) {
    /*
     * theta(t) = theta(t-1) * (1 - alpha * lambda / m)
     *    + alpha * SUM( (y(i) - y'(i)) * x(i,j) ) / m
     * Only update those theta that show up in samples of the current batch,
     *    or it is wrong and costs a lot of time.
     *    2 seconds (78 seconds if not) to train 1 million samples with SGD
     */
    if (_curr_batch == 1) _print_step("backward");
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

  void LRModel::_update() {
    if (_curr_batch == 1) _print_step("update");
    _theta = _theta_new;
  }

  void LRModel::_print_model_param() {
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

} // namespace model
