#ifndef __H_SAMPLE_H__
#define __H_SAMPLE_H__
#define TRAIN_DATASET_NAME "train-dataset"
#define TEST_DATASET_NAME "test-dataset"
#define SAMPLE_SHOW_SIZE 3
#define LABEL_MIDDLE 0.5
#define LIBSVM_SEP1 " "
#define LIBSVM_SEP2 ":"
#define LOGLOSS_DELTA 10e-10
#include "util.h"

namespace model {

  struct Data {
    util::label_type _label; // label
    util::score_type _score; // model_predict
    std::vector<util::sparse_f_type> _sparse_f_list; // sparse feature list
    Data(const util::label_type& label,
        const std::vector<util::sparse_f_type>& sparse_f_list) :
      _label(label), _score(0.0),
      _sparse_f_list(sparse_f_list) {}
  };

  class DataSet {
    public:
      DataSet(const util::path_type& data_path, const std::string& dataset_type);
      void init(util::hash2index_type& f_hash2index,
          util::index2hash_type& f_index2hash,
          util::f_index_type& f_size);
      void evaluate();
      util::score_type cal_auc();
      util::score_type cal_logloss();
      util::score_type cal_mse();
      inline std::vector<Data>& get_data() { return _data; }
      inline size_t get_size() { return _data_size; }
    private:
      void _get_base_info();
      util::path_type _data_path;
      std::string _dataset_type; // train/test-dataset
      std::vector<Data> _data; // data vector
      size_t _data_size;
      size_t _positive_size;
      util::score_type _ctr; // ctr of the dataset
      std::vector<util::predict_type> _predict_data; // store label and predict
      util::score_type _auc;
      util::score_type _mse;
      util::score_type _logloss;
  };

} // namespace model

#endif // __H_SAMPLE_H__
