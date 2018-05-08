#ifndef __H_LR_ADADELTA_H__
#define __H_LR_ADADELTA_H__
#define ADADELTA_DELTA ::pow(10, -7)
#include "lr.h"

namespace model {

  class LRAdadeltaModel : public LRModel {
    public:
      LRAdadeltaModel(DataSet* p_train_dataset, DataSet* p_test_dataset,
          const util::hash2index_type& f_hash2index, const util::index2hash_type& f_index2hash,
          const util::f_index_type& f_size, const std::string& model_type);
    protected:
      void _backward(const size_t& l, const size_t& r);
      void _update();
    private:
      util::param_type _adadelta_delta;
  };

} // namespace model

#endif // __H_LR_ADADELTA_H__
