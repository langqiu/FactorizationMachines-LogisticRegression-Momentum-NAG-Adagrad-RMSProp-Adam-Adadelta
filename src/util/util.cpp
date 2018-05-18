#include "util.h"

namespace util {

  std::vector<std::string> split(const std::string& source, const std::string& separator) {
    std::vector<std::string> ans;
    size_t l = 0;
    size_t r = source.find(separator, l);
    while (r != std::string::npos) {
      ans.push_back(source.substr(l, r-l));
      l = r + separator.size();
      r = source.find(separator, l);
    }
    ans.push_back(source.substr(l));
    return ans;
  }

  time_type time_now() {
    return std::chrono::system_clock::now();
  }

  void time_diff(const time_type& time_end, const time_type& time_start) {
#ifdef _DEBUG
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    double delta = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    std::cout << "  --> time-cost " << delta << std::endl; // time cost
#endif
  }

  score_type unit_random() {
    srand(time(0));
    return rand() % 100 / 101.0;
  }

} // namespace util
