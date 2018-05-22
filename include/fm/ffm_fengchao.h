#ifndef __H_FFM_FENGCHAO_H__
#define __H_FFM_FENGCHAO_H__
#define FIELD_SIZE 2
#define INIT_RANGE 0.0001
#include "model.h"

namespace model {

  class FFMFengchaoModel : public Model {
    public:
      FFMFengchaoModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type f_size, const std::string& model_type);
      void init(const size_t& iter_size, const size_t& batch_size,
          const util::param_type& _alpha_vector, const util::param_type& lambda_vector,
          const util::param_type& alpha_bias, const util::param_type& lambda_bias,
          const util::param_type& fm_dims);
    private:
      void _forward(const size_t& l, const size_t& r, DataSet* p_data);
      void _backward(const size_t& l, const size_t& r);
      void _update();
      void _print_model_param();
      void _init_vector(DataSet* p_data);
      // model parameters
      std::vector<std::vector<util::param_type>> _feature_vector;
      // hyper parameters
      util::param_type _alpha_vector;
      util::param_type _alpha_bias;
      util::param_type _lambda_vector;
      util::param_type _lambda_bias;
      size_t _fm_dims;
      // extra vectors
      std::vector<util::param_type> _theta_updated_vector;
  };

} // namespace model

#endif // __H_FFM_FENGCHAO_H__
