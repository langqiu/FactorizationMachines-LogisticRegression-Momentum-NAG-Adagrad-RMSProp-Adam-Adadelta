#include "lr_momentum.h"
using namespace util;

namespace model {

  LRMomentumModel::LRMomentumModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
      const hash2index_type& f_hash2index, const index2hash_type& f_index2hash,
      const f_index_type& f_size, const std::string& model_type) :
    LRModel(p_train_dataset, p_test_dataset,
        f_hash2index, f_index2hash, f_size, model_type) {}

  void LRMomentumModel::_backward(const size_t& l, const size_t& r) {
    /*
     * attention! element-wise operation
     * g(t) = -1 * [g(logloss) + g(L2)]
     * v(t) = beta_1 * v(t-1) + alpha * g(t)
     * theta(t) = theta(t-1) + v(t)
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
    // accumulate gradient of history separately in each direction
    for (auto& index : _theta_updated_vector) {
      _first_moment_vector[index] = _beta_1 * _first_moment_vector[index]
        + _alpha * _gradient[index];
    }
    // calculate theta_new
    for (auto& index : _theta_updated_vector) {
      _theta_new[index] += _first_moment_vector[index];
    }
  }

  void LRMomentumModel::_update() {
    if (_curr_batch == 1) _print_step("update");
    _theta = _theta_new;
    for (size_t i=0; i<_f_size; i++) {
      // do not set zero to moment vector, because features don't show up continuously between batchs
      _gradient[i] = 0.0;
    }
  }

} // namespace model
