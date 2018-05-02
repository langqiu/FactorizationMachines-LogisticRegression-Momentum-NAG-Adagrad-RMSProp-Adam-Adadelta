#ifndef __H_LR_NAG_H__
#define __H_LR_NAG_H__
#include "util.h"
#include "lr_momentum.h"

namespace lr {

  class LRNAG : public LRMomentum {
    public:
      LRNAG(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size);
    private:
      void _forward(const size_t& l, const size_t& r, DataSet* p_data);
  };

} // namespace lr

#endif // __H_LR_NAG_H__
