#include "fm.h"
using namespace util;

namespace model {

  FMModel::FMModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
      const hash2index_type& f_hash2index, const index2hash_type& f_index2hash,
      const f_index_type f_size, const std::string& model_type) :
    Model(p_train_dataset, p_test_dataset,
        f_hash2index, f_index2hash, f_size, model_type),
    _alpha_theta(0.0), _lambda_theta(0.0),
    _alpha_vector(0.0), _lambda_vector(0.0),
    _fm_dims(0) {}

  void FMModel::init(const size_t& iter_size, const size_t& batch_size,
      const param_type& alpha_theta, const param_type& lambda_theta,
      const param_type& alpha_vector, const param_type& lambda_vector,
      const param_type& fm_dims) {
    // init model parameters
    _theta.clear();
    _theta.reserve(_f_size);
    for (size_t i=0; i<_f_size; i++) {
      _theta.push_back(0.0);
    }
    _theta_new = _theta;
    _feature_vector.clear();
    _feature_vector.reserve(_f_size);
    for (size_t i=0; i<_f_size; i++) {
      std::vector<param_type> v(fm_dims, 0.0001);
      _feature_vector.push_back(v);
    }
    _feature_vector_new = _feature_vector;
    // init hyper parameters
    _alpha_theta = alpha_theta;
    _lambda_theta = lambda_theta;
    _alpha_vector = alpha_vector;
    _lambda_vector = lambda_vector;
    _fm_dims = static_cast<size_t>(fm_dims);
    // init train parameters
    _iter_size = iter_size;
    _batch_size = batch_size;
    _curr_batch = 0;
  }

  void FMModel::_forward(const size_t& l, const size_t& r, DataSet* p_data) {
    /*
     * predict the sample in [l, r) of the dataset p_data
     */
    if (_curr_batch == 1) _print_step("forward");
    auto& data = p_data->get_data(); // get dataset
    for (size_t i=l; i<r; i++) { // loop each sample in this batch
      score_type product = 0;
      auto& curr_sample = data[i]; // current sample
      // first order
      for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) { // do product
        auto& curr_f = curr_sample._sparse_f_list[k];
        product += _theta[curr_f.first] * curr_f.second;
      }
      product += _theta[_f_size-1]; // bias
      // cross
      for (size_t x=0; x<_fm_dims; x++) {
        score_type sum = 0.0;
        for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) {
          auto& curr_f = curr_sample._sparse_f_list[k];
          score_type dot = _feature_vector[curr_f.first][x] * curr_f.second;
          sum += dot;
          product += -0.5 * dot * dot;
        }
        product += 0.5 * sum * sum;
      }
      curr_sample._score = 1 / (1 + ::exp(-1 * product)); // sigmoid
    }
  }

  void FMModel::_backward(const size_t& l, const size_t& r) {
    if (_curr_batch == 1) _print_step("backward");
    auto& data = _p_train_dataset->get_data(); // get train dataset
    std::unordered_set<f_index_type> theta_updated; // record theta in BGD
    _theta_updated_vector.clear();
    score_type factor_theta = -1 * _alpha_theta * _lambda_theta / (r - l); // L2 delta factor
    score_type factor_vector = -1 * _alpha_vector * _lambda_vector / (r - l);
    for (size_t i=l; i<r; i++) {
      auto& curr_sample = data[i]; // current sample
      std::vector<score_type> vector_sum(_fm_dims, 0.0);
      for (size_t x=0; x<_fm_dims; x++) {
        for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) {
          auto& curr_f = curr_sample._sparse_f_list[k];
          vector_sum[x] += _feature_vector[curr_f.first][x] * curr_f.second;
        }
      }
      for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) { // loop features
        auto& curr_f = curr_sample._sparse_f_list[k]; // current feature
        _theta_new[curr_f.first] += _alpha_theta * (curr_sample._label - curr_sample._score)
          * curr_f.second / (r -l); // delta from the logloss
        // delta from L2
        if (r - l == 1) { // SGD
          _theta_new[curr_f.first] += _theta[curr_f.first] * factor_theta;
          _theta_updated_vector.push_back(curr_f.first);
        } else if (theta_updated.find(curr_f.first) == theta_updated.end()) {
          _theta_new[curr_f.first] += _theta[curr_f.first] * factor_theta; // BGD
          theta_updated.insert(curr_f.first); // record the index of theta that has showed up
          _theta_updated_vector.push_back(curr_f.first);
        }
        for (size_t x=0; x<_fm_dims; x++) {
          _feature_vector_new[curr_f.first][x] += _alpha_vector * curr_f.second
            * (curr_sample._label - curr_sample._score)
            * (vector_sum[x] - _feature_vector[curr_f.first][x] * curr_f.second);
        }
      }
      _theta_new[_f_size-1] += _alpha_theta * (curr_sample._label - curr_sample._score)
        / (r - l); // delta from logloss for bias
    }
    _theta_new[_f_size-1] += _theta[_f_size-1] * factor_theta; // delta from L2 for bias
    // vector L2
    for (auto index : _theta_updated_vector) {
      for (size_t x=0; x<_fm_dims; x++) {
        _feature_vector_new[index][x] += _feature_vector[index][x] * factor_vector;
      }
    }
  }

  void FMModel::_update() {
    if (_curr_batch == 1) _print_step("update");
    _theta = _theta_new;
    // 嵌套vector直接赋值性能极其差，此行代码影响30倍性能
    //_feature_vector = _feature_vector_new;
    for (auto index : _theta_updated_vector) {
      _feature_vector[index] = _feature_vector_new[index];
    }
  }

  void FMModel::_print_model_param() {
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
    for (int i=parameters.size()-1; i>=0 && parameters.size()-i<=MODEL_TOP_PARAMS; i--) {
      std::cout << _f_index2hash[parameters[i].second] << " " << parameters[i].first << std::endl;
    }
    std::cout << "bias: " << _theta[_f_size-1] << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    parameters.clear();
    for (size_t i=0; i<_f_size; i++) {
      param_type d = 0.0;
      for (size_t k=0; k<_fm_dims; k++) {
        d += _feature_vector[i][k] * _feature_vector[i][k];
      }
      parameters.push_back(param_evaluate_type(::sqrt(d), i));
    }
    sort(parameters.begin(), parameters.end(), sort_by_param); // sort by |theta(j)|
    for (size_t i=0; i<parameters.size() && i<MODEL_TOP_PARAMS; i++) {
      std::cout << _f_index2hash[parameters[i].second] << " " << parameters[i].first << std::endl;
    }
    for (int i=parameters.size()-1; i>=0 && parameters.size()-i<=MODEL_TOP_PARAMS; i--) {
      std::cout << _f_index2hash[parameters[i].second] << " " << parameters[i].first << std::endl;
    }
#endif
  }

} // namespace model
