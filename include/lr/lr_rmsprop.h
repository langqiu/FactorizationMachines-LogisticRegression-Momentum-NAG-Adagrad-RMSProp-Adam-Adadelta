#ifndef __H_LR_RMSPROP_H__
#define __H_LR_RMSPROP_H__
#define RMSPROP_DELTA ::pow(10, -6)
#include "util.h"
#include "lr_adagrad.h"

namespace lr {

  class LRRMSProp : public LRAdagrad {
    public:
      LRRMSProp(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size);
    private:
      void _backward(const size_t& l, const size_t& r);
      util::param_type _rmsprop_delta;
  };

} // namespace lr

#endif // __H_LR_RMSPROP_H__
