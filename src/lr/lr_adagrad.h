#ifndef __H_LR_ADAGRAD_H__
#define __H_LR_ADAGRAD_H__
#define ADAGRAD_DELTA ::pow(10, -7)
#include "util.h"
#include "lr.h"

namespace lr {

  class LRAdagrad : public LR {
    public:
      LRAdagrad(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size);
    protected:
      void _backward(const size_t& l, const size_t& r);
      void _update();
    private:
      util::param_type _adagrad_delta;
  };

} // namespace lr

#endif // __H_LR_ADAGRAD_H__
