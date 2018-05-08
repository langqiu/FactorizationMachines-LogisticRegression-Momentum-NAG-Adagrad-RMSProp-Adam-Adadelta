#ifndef __H_LR_RMSPROP_H__
#define __H_LR_RMSPROP_H__
#define RMSPROP_DELTA ::pow(10, -6)
#include "lr_adagrad.h"

namespace model {

  class LRRMSPropModel : public LRAdagradModel {
    public:
      LRRMSPropModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size, const std::string& model_type);
    private:
      void _backward(const size_t& l, const size_t& r);
      util::param_type _rmsprop_delta;
  };

} // namespace model

#endif // __H_LR_RMSPROP_H__
