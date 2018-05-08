#ifndef __H_LR_ADAM_H__
#define __H_LR_ADAM_H__
#define ADAM_DELTA ::pow(10, -8)
#include "lr.h"

namespace model {

  class LRAdamModel : public LRModel {
    public:
      LRAdamModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size, const std::string& model_type);
    private:
      void _backward(const size_t& l, const size_t& r);
      void _update();
      util::param_type _adam_delta;
  };

} // namespace model

#endif // __H_LR_ADAM_H__
