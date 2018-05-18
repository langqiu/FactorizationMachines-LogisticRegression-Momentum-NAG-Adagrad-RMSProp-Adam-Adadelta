#ifndef __H_FM_FENGCHAO_H__
#define __H_FM_FENGCHAO_H__
#define FM_DELTA 10.0
#define MIN_BOUND -10.0
#define MAX_BOUND 10.0
#define INIT_RANGE 0.0001
#include "model.h"

namespace model {

  class FMFengchaoModel : public Model{
    public:
      FMFengchaoModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type f_size, const std::string& model_type);
      void init(const size_t& iter_size, const size_t& batch_size,
          const util::param_type& alpha, const util::param_type& reserve_1,
          const util::param_type& reserve_2, const util::param_type& reserve_3,
          const util::param_type& fm_dims);
    private:
      void _forward(const size_t& l, const size_t& r, DataSet* p_data);
      void _backward(const size_t& l, const size_t& r);
      void _update();
      void _print_model_param();
      // model parameters
      std::vector<std::vector<util::param_type>> _feature_vector;
      // hyper parameters
      util::param_type _alpha;
      util::param_type _fm_delta;
      util::param_type _min_bound;
      util::param_type _max_bound;
      size_t _fm_dims;
      // extra vectors
      std::vector<util::param_type> _theta_updated_vector;
      std::vector<std::vector<util::param_type>> _second_moment_vector;
  };

} // namespace model

#endif // __H_FM_FENGCHAO_H__
