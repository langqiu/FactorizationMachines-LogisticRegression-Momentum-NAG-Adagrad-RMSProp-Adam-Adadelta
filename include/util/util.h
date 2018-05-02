#ifndef __H_UTIL_H__
#define __H_UTIL_H__
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

namespace util {

  typedef float label_type;
  typedef float score_type;
  typedef float f_value_type;
  typedef float param_type;
  typedef char* path_type;
  typedef uint64_t hash_id_type;
  typedef uint32_t f_index_type;
  typedef std::pair<f_index_type, f_value_type> sparse_f_type;
  typedef std::unordered_map<hash_id_type, f_index_type> hash2index_type;
  typedef std::unordered_map<f_index_type, hash_id_type> index2hash_type;
  typedef std::pair<score_type, label_type> predict_type;
  typedef std::pair<param_type, f_index_type> param_evaluate_type;
  typedef std::chrono::time_point<std::chrono::system_clock> time_type;

  std::vector<std::string> split(const std::string& source, const std::string& separator);
  time_type time_now();
  void time_diff(const time_type& time_end, const time_type& time_start);

} // namespace util

#endif // __H_UTIL_H__
