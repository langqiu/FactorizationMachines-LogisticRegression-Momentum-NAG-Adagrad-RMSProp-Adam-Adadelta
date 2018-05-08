#include "lr_adadelta.h"
using namespace util;

namespace model {

  LRAdadeltaModel::LRAdadeltaModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
      const hash2index_type& f_hash2index, const index2hash_type& f_index2hash,
      const f_index_type& f_size, const std::string& model_type) :
    LRModel(p_train_dataset, p_test_dataset,
        f_hash2index, f_index2hash, f_size, model_type),
    _adadelta_delta(ADADELTA_DELTA) {}

  void LRAdadeltaModel::_backward(const size_t& l, const size_t& r) {
    /*
     * attention! element-wise operation
     * g(t) = -1 * [g(logloss) + g(L2)]
     * r(t) = beta_1 * r(t-1) + (1 - beta_1) * g(t) * g(t)
     * theta(t) = theta(t-1) + (sqrt(d(t-1)) + delta) * g(t) / (sqrt(r(t)) + delta)
     * d(t) = beta_1 * d(t-1) + (1 - beta_1) * ::pow(theta(t) - theta(t-1), 2)
     */
    if (_curr_batch == 1) _print_step("backward");
    auto& data = _p_train_dataset->get_data(); // get train dataset
    std::unordered_set<f_index_type> theta_updated; // record theta in BGD
    _theta_updated_vector.clear(); // clear before backward
    // calculate -1 * gradient
    param_type factor = -1 * _lambda / (r - l); // L2 gradient factor
    for (size_t i=l; i<r; i++) {
      auto& curr_sample = data[i]; // current sample
      for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) { // loop features
        auto& curr_f = curr_sample._sparse_f_list[k]; // current feature
        _gradient[curr_f.first] += (curr_sample._label - curr_sample._score)
          * curr_f.second / (r - l); // gradient from logloss
        // gradient from L2
        if (r - l == 1) { // SGD
          _gradient[curr_f.first] += factor * _theta[curr_f.first];
          _theta_updated_vector.push_back(curr_f.first);
        } else if (theta_updated.find(curr_f.first) == theta_updated.end()) { // BGD
          _gradient[curr_f.first] += factor * _theta[curr_f.first];
          theta_updated.insert(curr_f.first); // record the index of feature that has showed up
          _theta_updated_vector.push_back(curr_f.first);
        }
      }
      _gradient[_f_size-1] += (curr_sample._label - curr_sample._score)
        / (r - l); // gradient from logloss for bias
    }
    _theta_updated_vector.push_back(_f_size-1);
    _gradient[_f_size-1] += factor * _theta[_f_size-1]; // gradient from L2 for bias
    // calculate second moment vector in history
    for (auto& index : _theta_updated_vector) {
      _second_moment_vector[index] = _beta_1 * _second_moment_vector[index]
        + (1 - _beta_1) * _gradient[index] * _gradient[index];
    }
    // calculate theta_new
    for (auto& index : _theta_updated_vector) {
      _theta_new[index] += (::sqrt(_delta[index]) + _adadelta_delta) * _gradient[index]
        / (::sqrt(_second_moment_vector[index]) + _adadelta_delta);
    }
    // calculate delta
    for (auto& index : _theta_updated_vector) {
      _delta[index] = _beta_1 * _delta[index]
        + (1 - _beta_1) * ::pow(_theta_new[index] - _theta[index], 2);
    }
  }

  void LRAdadeltaModel::_update() {
    if (_curr_batch == 1) _print_step("update");
    _theta = _theta_new;
    for (size_t i=0; i<_f_size; i++) {
      // do not set zero to moment vector, because features don't show up continuously between batchs
      _gradient[i] = 0.0;
    }
  }

} // namespace model
