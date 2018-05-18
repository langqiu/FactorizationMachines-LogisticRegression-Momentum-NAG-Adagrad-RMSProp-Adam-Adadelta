#ifndef __H_MODEL_H__
#define __H_MODEL_H__
#define TRAIN_PRINT_INTERVAL 10000
#define MODEL_TOP_PARAMS 10
#include "util.h"
#include "sample.h"

namespace model {

  bool sort_by_param(const util::param_evaluate_type& l, const util::param_evaluate_type& r);

  class Model {
    public:
      Model(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size, const std::string& model_type);
      virtual ~Model();
      virtual void init(const size_t& iter_size, const size_t& batch_size,
          const util::param_type& alpha, const util::param_type& lambda,
          const util::param_type& beta_1, const util::param_type& beta_2,
          const util::param_type& fm_dims);
      void train();
      void evaluate();
    protected:
      virtual void _forward(const size_t& l, const size_t& r, DataSet* p_data);
      virtual void _backward(const size_t& l, const size_t& r);
      virtual void _update();
      virtual void _print_model_param();
      void _predict_dataset(DataSet* p_data);
      void _cal_model_mse();
      void _cal_model_logloss();
      void _print_step(const std::string& step);
      void _print_mini_batch(const size_t& batch);
      // train parameters
      size_t _iter_size; // iteration size
      size_t _batch_size; // batch size
      size_t _curr_batch; // current batch
      // data
      DataSet* _p_train_dataset;
      DataSet* _p_test_dataset;
      // feature
      util::hash2index_type _f_hash2index; // map from hash id to index in vector
      util::index2hash_type _f_index2hash; // map from index in vector to hash id
      util::f_index_type _f_size; // feature size including bias
      // model type
      std::string _model_type; // lr/fm
  };

} // namespace model

#endif // __H_MODEL_H__
