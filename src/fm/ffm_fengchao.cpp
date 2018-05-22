#include "ffm_fengchao.h"
using namespace util;

namespace model {

  FFMFengchaoModel::FFMFengchaoModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
      const hash2index_type& f_hash2index, const index2hash_type& f_index2hash,
      const f_index_type f_size, const std::string& model_type) :
    Model(p_train_dataset, p_test_dataset,
        f_hash2index, f_index2hash, f_size, model_type),
    _alpha_1(0.0), _alpha_2(0.0),
    _lambda_1(0.0), _lambda_2(0.0),
    _fm_dims(0) {}

  void FFMFengchaoModel::init(const size_t& iter_size, const size_t& batch_size,
      const param_type& alpha_vector, const param_type& lambda_vector,
      const param_type& alpha_bias, const param_type& lambda_bias,
      const param_type& fm_dims) {
    _fm_dims = static_cast<size_t>(fm_dims+1);
    // model parameters
    _feature_vector.clear();
    _feature_vector.reserve();
    for (size_t i=0; i<_f_size; i++) {
      std::vector<param_type> v;
      for (size_t k=0; k<_fm_dims-1; k++) {
        v.push_back((2 * unit_random() - 1) * INIT_RANGE);
      }
      v.push_back(0.0);
      _feature_vector.push_back(v);
    }
    _alpha_vector = alpha_vector;
    _alpha_bias = alpha_bias;
    _lambda_vector = lambda_vector;
    _lambda_bias = lambda_bias;
    // bad implementation
    _init_vector(_p_train_dataset);
    _init_vector(_p_test_dataset);
  }

  void FFMFengchaoModel::_forward(const size_t& l, const size_t& r, DataSet* p_data) {
    if (_curr_batch == 1) _print_step("forward");
    auto& data = p_data->get_data();
    for (size_t i=l; i<r; i++) {
      auto& curr_sample = data[i];
      for (size_t k=0; k<FIELD_SIZE; k++) {
        for (size_t x=0; x<_fm_dims; x++) {
          curr_sample._ffm_sum_vector[k][x] = 0.0;
        }
      }
    }
    for (size_t i=l; i<r; i++) {
      auto& curr_sample = data[i];
      for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) {
        auto& curr_f = curr_sample._sparse_f_list[i];
        for (size_t x=0; x<_fm_dims; x++) {
          curr_sample._ffm_sum_vector[curr_f.second][x] += _feature_vector[curr_f.first][x];
        }
      }
      score_type product = 0.0;
      for (size_t x=0; x<_fm_dims-1; x++) {
        product += curr_sample._ffm_sum_vector[0][x] * curr_sample._ffm_sum_vector[1][x];
      }
      product += curr_sample._ffm_sum_vector[0][_fm_dims-1];
      product += curr_sample._ffm_sum_vector[1][_fm_dims-1];
      curr_sample._score = 1.0 / (1 + ::exp(-1 * product));
    }
  }

  void FFMFengchaoModel::_backward(const size_t& l, const size_t& r) {
    if (_curr_batch == 1) _print_step("backward");
    auto& data = _p_train_dataset->get_data();
    std::unordered_set<f_index_type> updated_theta;
    _theta_updated_vector.clear();
    for (size_t i=l; i<r; i++) {
      auto& curr_sample = data[i];
      for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) {
        auto& curr_f = curr_sample._sparse_f_list[k];
        for (size_t x=0; x<_fm_dims; x++) {
          if (x == _fm_dims-1) {
            _feature_vector_new[curr_f.first][x] += _alpha_bias
              * (curr_sample._lable - curr_sample._score) / (r - l)
          } else {
            _feature_vector_new[curr_f.first][x] += _alpha_vector
              * (curr_sample._lable - curr_sample._score) / (r - l)
              * curr_sample._ffm_sum_vector[FIELD_SIZE-1-curr_f.second][x];
          }
        }
        if (r - l == 1) {
          _theta_updated_vector.push_back(curr_f.first);
        } else {
          if (updated_theta.find(curr_f.first) == updated_theta.end()) {
            theta_update.insert(curr_f.first);
            _theta_updated_vector.push_back(curr_f.first);
          }
        }
      }
    }
    for (auto& v : _theta_updated_vector) {
      score_type factor_vector = -1 * _alpha_vector * _lambda_vector / (r - l);
      score_type factor_bias = -1 * _alpha_bias * _lambda_bias / (r - l);
      for (size_t x=0; x<_fm_dims-1; x++) {
        _feature_vector_new[v][x] += factor_vector * _feature_vector[v][x];
      }
      _feature_vector_new[_fm_dims-1] += factor_bias * (_feature_vector[v][x]>0? 1 : -1);
    }
  }

  void FFMFengchaoModel::_update() {
    if (_curr_batch == 1) _print_step("update");
    for (auto& v : _theta_updated_vector) {
      for (size_t x=0; x<_fm_dims; x++) {
        _feature_vector[v][x] = _feature_vector_new[v][x];
      }
    }
  }

  void FFMFengchaoModel::_print_model_param() {
#ifdef _DEBUG
#endif
  }

  void FFMFengchaoModel::_init_vector(DataSet* p_data) {
    auto& data = p_data->get_data();
    size_t data_size = p_data->get_size();
    for (size_t i=0; i<data_size; i++) {
      data[i]._ffm_sum_vector.clear();
      for (size_t k=0; k<FIELD_SIZE; k++) {
        vector<param_type> v(_fm_dims, 0.0);
        data[i]._ffm_sum_vector.push_back(v);
      }
    }
  }

} // namespace model
