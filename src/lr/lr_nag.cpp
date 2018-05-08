#include "lr_nag.h"
using namespace util;

namespace model {

  LRNAGModel::LRNAGModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
      const hash2index_type& f_hash2index, const index2hash_type& f_index2hash,
      const f_index_type& f_size, const std::string& model_type) :
    LRMomentumModel(p_train_dataset, p_test_dataset,
        f_hash2index, f_index2hash, f_size, model_type) {}

  void LRNAGModel::_forward(const size_t& l, const size_t& r, DataSet* p_data) {
    /*
     * predict the sample in [l, r) of the dataset p_data
     * theta(t) = theta(t-1) + beta_1 * v(t-1)
     */
    if (_curr_batch == 1) _print_step("forward");
    // small step first
    if (p_data == _p_train_dataset) {
      for (auto& index : _theta_updated_vector) {
        _theta_new[index] += _beta_1 * _first_moment_vector[index];
      }
    }
    // predict y' by using _theta_new
    auto& data = p_data->get_data(); // get dataset
    for (size_t i=l; i<r; i++) { // loop each sample in this batch
      score_type product = 0;
      auto& curr_sample = data[i]; // current sample
      for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) { // do product
        auto& curr_f = curr_sample._sparse_f_list[k];
        product += _theta_new[curr_f.first] * curr_f.second;
      }
      product += _theta_new[_f_size-1]; // bias
      curr_sample._score = 1 / (1 + ::exp(-1 * product)); // sigmoid
    }
    // reset theta_new
    _theta_new = _theta;
  }

} // namespace model
