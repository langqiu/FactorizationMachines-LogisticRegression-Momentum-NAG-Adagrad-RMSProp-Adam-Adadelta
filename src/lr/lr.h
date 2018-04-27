#ifndef __H_LR_H__
#define __H_LR_H__
#define MSE_INTERVAL 10
#define MODEL_TOP_PARAMS 10
#include "util.h"
#include "sample.h"

namespace lr {

  bool sort_by_param(const util::param_evaluate_type& l, const util::param_evaluate_type& r);

  void print_step(const std::string& step);

  void print_iteration(const size_t& iter);

  class LR {
    public:
      LR(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size);
      virtual ~LR();
      void init(const size_t& iter_size, const size_t& batch_size,
          const util::param_type& alpha, const util::param_type& lambda,
          const util::param_type& beta_1, const util::param_type& beta_2);
      void train();
      void evaluate();
    protected:
      virtual void _forward(const size_t& l, const size_t& r, DataSet* p_data);
      virtual void _backward(const size_t& l, const size_t& r);
      virtual void _update();
      void _print_model_param();
      void _predict_dataset(DataSet* p_data);
      void _cal_model_mse();
      void _cal_model_logloss();
      // model parameters
      std::vector<util::param_type> _theta; // including bias at the end
      std::vector<util::param_type> _theta_new; // for update
      // hyperparameters
      util::param_type _alpha; // learning rate
      util::param_type _lambda; // L2 parameter
      util::param_type _beta_1; // for momentum/nag/adam
      util::param_type _beta_2; // for rmsprop/adam
      // train parameters
      size_t _iter_size; // iteration size
      size_t _batch_size; // batch size
      size_t _curr_batch; // current batch, for adam
      // extra vectors
      std::vector<util::f_index_type> _theta_updated_vector; // theta need to update in current batch
      std::vector<util::param_type> _gradient; // current gradient
      std::vector<util::param_type> _delta; // adadelta
      std::vector<util::param_type> _first_moment_vector; // for momentum/nag/adam/adadelta
      std::vector<util::param_type> _second_moment_vector; // for adagrad/rmsprop/adam
      // data
      DataSet* _p_train_dataset;
      DataSet* _p_test_dataset;
      // feature
      util::hash2index_type _f_hash2index; // map from hash id to index in vector
      util::index2hash_type _f_index2hash; // map from index in vector to hash id
      util::f_index_type _f_size; // feature size including bias
  };

} // namespace lr

#endif // __H_LR_H__
