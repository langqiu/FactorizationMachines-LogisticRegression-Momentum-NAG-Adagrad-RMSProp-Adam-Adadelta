#ifndef __H_LR_ADAM_H__
#define __H_LR_ADAM_H__
#define ADAM_DELTA ::pow(10, -8)
#include "util.h"
#include "lr.h"

namespace lr {

  class LRAdam : public LR {
    public:
      LRAdam(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size);
    private:
      void _backward(const size_t& l, const size_t& r);
      void _update();
      util::param_type _adam_delta;
  };

} // namespace lr

#endif // __H_LR_ADAM_H__
