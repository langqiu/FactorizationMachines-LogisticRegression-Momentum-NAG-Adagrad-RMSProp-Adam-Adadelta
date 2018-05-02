#ifndef __H_LR_MOMENTUM_H__
#define __H_LR_MOMENTUM_H__
#include "util.h"
#include "lr.h"

namespace lr {

  class LRMomentum : public LR {
    public:
      LRMomentum(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size);
    protected:
      void _backward(const size_t& l, const size_t& r);
      void _update();
  };

} // namespace lr

#endif // __H_LR_MOMENTUM_H__
