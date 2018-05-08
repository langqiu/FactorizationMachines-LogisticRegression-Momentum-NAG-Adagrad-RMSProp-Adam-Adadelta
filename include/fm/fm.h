#ifndef __H_FM_H__
#define __H_FM_H__
#include "model.h"

namespace model {

  class FMModel : public Model {
    public:
      FMModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type f_size, const std::string& model_type);
      void init(const size_t& iter_size, const size_t& batch_size,
          const util::param_type& alpha_theta, const util::param_type& lambda_theta,
          const util::param_type& alpha_vector, const util::param_type& lambda_vector,
          const util::param_type& fm_dims);
    private:
      void _forward(const size_t& l, const size_t& r, DataSet* p_data);
      void _backward(const size_t& l, const size_t& r);
      void _update();
      void _print_model_param();
      // model parameters
      std::vector<util::param_type> _theta;
      std::vector<util::param_type> _theta_new;
      std::vector<std::vector<util::param_type>> _feature_vector;
      std::vector<std::vector<util::param_type>> _feature_vector_new;
      // hyper parameters
      util::param_type _alpha_theta;
      util::param_type _lambda_theta;
      util::param_type _alpha_vector;
      util::param_type _lambda_vector;
      size_t _fm_dims;
      // extra vectors
      std::vector<util::f_index_type> _theta_updated_vector;
  };

} // namespace model

#endif // __H_FM_H__
