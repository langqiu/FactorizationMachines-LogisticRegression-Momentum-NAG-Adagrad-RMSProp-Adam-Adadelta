#ifndef __H_LR_MOMENTUM_H__
#define __H_LR_MOMENTUM_H__
#include "lr.h"

namespace model {

  class LRMomentumModel : public LRModel {
    public:
      LRMomentumModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size, const std::string& model_type);
    protected:
      void _backward(const size_t& l, const size_t& r);
      void _update();
  };

} // namespace model

#endif // __H_LR_MOMENTUM_H__
