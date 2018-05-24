#ifndef __H_MODEL_FACTORY_H__
#define __H_MODEL_FACTORY_H__
#define LR_MODEL "lr"
#define LR_M_MODEL "lr_m"
#define LR_NAG_MODEL "lr_nag"
#define LR_AG_MODEL "lr_ag"
#define LR_RMS_MODEL "lr_rms"
#define LR_ADAM_MODEL "lr_adam"
#define LR_ADAD_MODEL "lr_adad"
#define FM_MODEL "fm"
#define FM_FC_MODEL "fm_fc"
#define FFM_FC_MODEL "ffm_fc"
#include "lr.h"
#include "lr_momentum.h"
#include "lr_nag.h"
#include "lr_adagrad.h"
#include "lr_rmsprop.h"
#include "lr_adam.h"
#include "lr_adadelta.h"
#include "fm.h"
#include "fm_fengchao.h"
#include "ffm_fengchao.h"

namespace model {

  class ModelFactory {
    public:
      static Model* get_lr_instance(const std::string& model_type);
      static void load_dataset(const util::path_type& train_path, const util::path_type& test_path);
      inline static DataSet* get_test_dataset() { return _p_test_dataset; }
    private:
      // model
      static Model* _p_model;
      static std::string _model_type;
      // data
      static DataSet* _p_train_dataset;
      static DataSet* _p_test_dataset;
      // feature
      static util::hash2index_type _f_hash2index;
      static util::index2hash_type _f_index2hash;
      static util::f_index_type _f_size;
  };

} // namespace model

#endif // __H_MODEL_FACTORY_H__
