#include "fm_fengchao.h"
using namespace util;

namespace model {

  FMFengchaoModel::FMFengchaoModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
      const hash2index_type& f_hash2index, const index2hash_type& f_index2hash,
      const f_index_type f_size, const std::string& model_type) :
    Model(p_train_dataset, p_test_dataset,
        f_hash2index, f_index2hash, f_size, model_type),
    _alpha(0.0), _fm_delta(FM_DELTA),
    _min_bound(MIN_BOUND), _max_bound(MAX_BOUND),
    _fm_dims(0) {}

  void FMFengchaoModel::init(const size_t& iter_size, const size_t& batch_size,
      const param_type& alpha, const param_type& reserve_1,
      const param_type& reserve_2, const param_type& reserve_3,
      const param_type& fm_dims) {
    // init model parameters
    _fm_dims = static_cast<size_t>(fm_dims+1);
    _feature_vector.clear();
    _feature_vector.reserve(_f_size);
    for (size_t i=0; i<_f_size; i++) {
      std::vector<param_type> v;
      for (size_t j=0; j<_fm_dims-1; j++) {
        score_type r = (2 * unit_random() -1) * INIT_RANGE;
        v.push_back(r);
      }
      v.push_back(0.0); // bias
      _feature_vector.push_back(v);
    }
    // init hyper parameters
    _alpha = alpha;
    // init train parameters
    _iter_size = iter_size;
    _batch_size = batch_size;
    _curr_batch = 0;
    // init extra vectors
    _second_moment_vector.clear();
    _second_moment_vector.reserve(_f_size);
    for (size_t i=0; i<_f_size; i++) {
      std::vector<util::param_type> v(_fm_dims, 0.0);
      _second_moment_vector.push_back(v);
    }
    _gradient = _second_moment_vector;
  }

  void FMFengchaoModel::_forward(const size_t& l, const size_t& r, DataSet* p_data) {
    /*
     * predict the sample in [l, r) of the dataset p_data
     */
    if (_curr_batch == 1) _print_step("forward");
    auto& data = p_data->get_data(); // get dataset
    for (size_t i=l; i<r; i++) { // loop each sample in this batch
      auto& curr_sample = data[i]; // current sample
      std::vector<param_type> feature_vector_sum(_fm_dims, 0.0);
      for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) {
        auto& curr_f = curr_sample._sparse_f_list[k];
        for (size_t x=0; x<_fm_dims; x++) {
          feature_vector_sum[x] += _feature_vector[curr_f.first][x];
        }
      }
      score_type product = 0.0;
      for (size_t x=0; x<_fm_dims/2; x++) {
        product += feature_vector_sum[x] * feature_vector_sum[x + _fm_dims/2];
      }
      product += feature_vector_sum[_fm_dims - 1];
      curr_sample._score = 1 / (1 + ::exp(-1 * product)); // sigmoid
    }
  }

  void FMFengchaoModel::_backward(const size_t& l, const size_t& r) {
    if (_curr_batch == 1) _print_step("backward");
    auto& data = _p_train_dataset->get_data(); // get train dataset
    std::unordered_set<f_index_type> theta_updated; // record theta in BGD
    std::vector<param_type> curr_theta_updated;
    for (size_t i=l; i<r; i++) {
      auto& curr_sample = data[i]; // current sample
      std::vector<param_type> feature_vector_sum(_fm_dims, 0.0);
      _theta_updated_vector.clear();
      for (size_t k=0; k<curr_sample._sparse_f_list.size(); k++) { // loop features
        auto& curr_f = curr_sample._sparse_f_list[k]; // current feature
        for (size_t x=0; x<_fm_dims; x++) {
          feature_vector_sum[x] += _feature_vector[curr_f.first][x];
        }
        _theta_updated_vector.push_back(curr_f.first);
        if (r - l > 1 && theta_updated.find(curr_f.first) == theta_updated.end()) {
          theta_updated.insert(curr_f.first);
          curr_theta_updated.push_back(curr_f.first);
        }
      }
      for (auto& v : _theta_updated_vector) {
        for (size_t x=0; x<_fm_dims; x++) {
          score_type gradient = curr_sample._label - curr_sample._score;
          if (x < _fm_dims - 1) gradient *= feature_vector_sum[(x + _fm_dims / 2) % (_fm_dims - 1)];
          _feature_vector[v][x] += _alpha * gradient * ::sqrt(_fm_delta)
            / ::sqrt(_fm_delta + _second_moment_vector[v][x]);
          if (_feature_vector[v][x] < _min_bound) _feature_vector[v][x] = _min_bound;
          if (_feature_vector[v][x] > _max_bound) _feature_vector[v][x] = _max_bound;
          if (r - l == 1) {
            _second_moment_vector[v][x] += gradient * gradient;
          } else {
            _gradient[v][x] += gradient;
          }
        }
      }
    }
    if (r - l == 1) return;
    for (auto& v : curr_theta_updated) {
      for (size_t x=0; x<_fm_dims; x++) {
        _second_moment_vector[v][x] += _gradient[v][x] * _gradient[v][x];
        _gradient[v][x] = 0.0;
      }
    }
  }

  void FMFengchaoModel::_update() {
    if (_curr_batch == 1) _print_step("update");
    // 嵌套vector直接赋值性能极其差，此行代码影响10倍性能
    //_feature_vector = _feature_vector_new;
  }

  void FMFengchaoModel::_print_model_param() {
#ifdef _DEBUG
    /*
     * print the most useful, useless features and model bias
     */
    std::vector<param_evaluate_type> parameters;
    std::cout << "############ model-top-params ############" << std::endl;
    for (size_t i=0; i<_f_size; i++) {
      if (_f_index2hash[i] == 0) continue;
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
