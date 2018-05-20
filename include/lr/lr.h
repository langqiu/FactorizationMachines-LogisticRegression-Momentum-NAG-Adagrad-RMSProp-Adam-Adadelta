#ifndef __H_LR_H__
#define __H_LR_H__
#include "model.h"

namespace model {

  class LRModel : public Model {
    public:
      LRModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size, const std::string& model_type);
      void init(const size_t& iter_size, const size_t& batch_size,
          const util::param_type& alpha, const util::param_type& lambda,
          const util::param_type& beta_1, const util::param_type& beta_2,
          const util::param_type& reserve_1);
    protected:
      void _forward(const size_t& l, const size_t& r, DataSet* p_data);
      void _backward(const size_t& l, const size_t& r);
      void _update();
      void _print_model_param();
      // model parameters
      std::vector<util::param_type> _theta; // including bias at the end
      std::vector<util::param_type> _theta_new; // for update
      // hyperparameters
      util::param_type _alpha; // learning rate
      util::param_type _lambda; // L2 parameter
      util::param_type _beta_1; // for momentum/nag/adam
      util::param_type _beta_2; // for rmsprop/adam
      // extra vectors
      std::vector<util::f_index_type> _theta_updated_vector; // theta need to update in current batch
      std::vector<util::param_type> _gradient; // current gradient
      std::vector<util::param_type> _delta; // adadelta
      std::vector<util::param_type> _first_moment_vector; // for momentum/nag/adam/adadelta
      std::vector<util::param_type> _second_moment_vector; // for adagrad/rmsprop/adam
  };

} // namespace model

#endif // __H_LR_H__
