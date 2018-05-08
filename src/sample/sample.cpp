#include "sample.h"
using namespace util;

namespace model {

  bool sort_by_score(const predict_type& l, const predict_type& r) {
    return l.first > r.first;
  }

  DataSet::DataSet(const path_type& data_path, const std::string& dataset_type) :
    _data_path(data_path), _dataset_type(dataset_type),
    _data_size(0), _positive_size(0),
    _ctr(0.0), _auc(0.0),
    _mse(0.0), _logloss(0.0) {}

  void DataSet::init(hash2index_type& f_hash2index,
      index2hash_type& f_index2hash,
      f_index_type& f_size) {
    /*
     * TODO multi-thread read file
     */
    std::ifstream ifs(_data_path); // read file
    std::string line;
    time_type time_start = time_now(); // time before load data
    while (getline(ifs, line)) { // loop the data file
      // get label and feature list
      std::vector<std::string> data_vector = split(line, LIBSVM_SEP1);
      if (data_vector.size() < 2) continue;
      label_type label = atof(data_vector[0].c_str());
      std::vector<sparse_f_type> feature_list;
      // parse the feature list
      for (size_t i=1; i<data_vector.size(); i++) { // loop the feature
        // parse feature hash_id and value
        std::vector<std::string> feature_pair = split(data_vector[i], LIBSVM_SEP2);
        if (feature_pair.size() != 2) continue;
        hash_id_type f_hash_id = strtoull(feature_pair[0].c_str(), NULL, 0);
        f_value_type f_value = atof(feature_pair[1].c_str());
        // map feature hash_id to feature index
        if (_dataset_type == TRAIN_DATASET_NAME
            && f_hash2index.find(f_hash_id) == f_hash2index.end()) {
          f_hash2index[f_hash_id] = f_size;
          f_index2hash[f_size] = f_hash_id;
          f_size++;
        }
        // add current feature to the feature list
        feature_list.push_back(sparse_f_type(f_hash2index[f_hash_id], f_value));
      }
      _data.push_back(Data(label, feature_list)); // add one sample
      _data_size++;
      if (label > LABEL_MIDDLE) _positive_size++;
    }
    time_type time_end = time_now(); // time after load data
    time_diff(time_end, time_start); // time cost of loading
    ifs.close(); // close handle of file
    _ctr = _positive_size * 1.0 / _data_size; // ctr of the dataset
    if (_dataset_type == TRAIN_DATASET_NAME) f_size++; // add bias feature
    _predict_data.reserve(_data_size); // predict data set size
    _get_base_info(); //print detailed info of the dataset
  }

  void DataSet::evaluate() {
#ifdef _DEBUG
    std::cout << "############ "<< _dataset_type << " ############" << std::endl;
    std::cout << "  --> auc: " << cal_auc() << std::endl;
    std::cout << "  --> logloss: " << cal_logloss() << std::endl;
    std::cout << "  --> mse: " << cal_mse() << std::endl;
#endif
  }

  score_type DataSet::cal_auc() {
    // prepare data
    _predict_data.clear();
    for (size_t i=0; i<_data_size; i++) {
      _predict_data.push_back(predict_type(_data[i]._score, _data[i]._label));
    }
    sort(_predict_data.begin(), _predict_data.end(), sort_by_score);
    // calculate auc
    _auc = 0.0;
    size_t positive = 0;
    size_t negative = 0;
    for (size_t i=0; i<_data_size; i++) {
      if (_predict_data[i].second > LABEL_MIDDLE) {
        positive++;
      } else {
        negative++;
        _auc += positive;
      }
    }
    _auc = _auc / positive / negative;
    return _auc;
  }

  score_type DataSet::cal_mse() {
    // calculate mse
    _mse = 0.0;
    for (size_t i=0; i<_data_size; i++) {
      _mse += ::pow(_data[i]._label - _data[i]._score, 2);
    }
    _mse /= _data_size;
    return _mse;
  }

  score_type DataSet::cal_logloss() {
    // calculate logloss
    _logloss = 0.0;
    for (size_t i=0; i<_data_size; i++) {
      _logloss += -1 * (_data[i]._label * ::log(_data[i]._score)
          + (1 - _data[i]._label) * ::log(1 - _data[i]._score));
    }
    _logloss /= _data_size;
    return _logloss;
  }

  void DataSet::_get_base_info() {
    std::cout << "############ "<< _dataset_type << " ############" << std::endl;
    std::cout << "  --> size: " << _data_size << std::endl;
    std::cout << "  --> ctr: " << _ctr << std::endl;
    // sample data from dataset
    for (size_t i=0; i<_data_size && i<SAMPLE_SHOW_SIZE; i++) {
      auto& curr_data = _data[i];
      std::cout << curr_data._label << " " << curr_data._score << " ";
      for (size_t j=0; j<curr_data._sparse_f_list.size(); j++) {
        std::cout << curr_data._sparse_f_list[j].first << ":" << curr_data._sparse_f_list[j].second << " ";
      }
      std::cout << std::endl;
    }
  }

} // namespace model
