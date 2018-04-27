#define PARAM_SEP ","
#include "util.h"
#include "lr_factory.h"

using namespace lr;
using namespace util;

void help_1() {
  std::cout << "---------------------------------------------------------------------" << std::endl;
  std::cout << "please input model type (" << LR_MODEL
    << "/" << LR_M_MODEL
    << "/" << LR_NAG_MODEL
    << "/" << LR_AG_MODEL
    << "/" << LR_RMS_MODEL
    << "/" << LR_ADAM_MODEL
    << "/" << LR_ADAD_MODEL
    << "):" << std::endl;
  std::cout << "---------------------------------------------------------------------" << std::endl;
}

void help_2() {
  std::cout << "---------------------------------------------------------------------" << std::endl;
  std::cout << "please input params separated by comma (iter,batch):" << std::endl;
  std::cout << "---------------------------------------------------------------------" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << argv[0] << " train_data test_data" << std::endl;
    return -1;
  }
  path_type train_path = argv[1];
  path_type test_path = argv[2];
  LRFactory::load_dataset(train_path, test_path);
  LR* lr_instance = NULL;
  std::string model_type;
  std::string command;
  //std::vector<std::string> v_model = std::vector<std::string>{LR_MODEL,LR_M_MODEL,LR_NAG_MODEL,LR_AG_MODEL,LR_RMS_MODEL,LR_ADAM_MODEL,LR_ADAD_MODEL};
  help_1();
  while (std::cin >> model_type) { // remove '\n' by default
  //for (auto& model_type : v_model) {
    lr_instance = LRFactory::get_lr_instance(model_type);
    if (!lr_instance) {
      help_1();
      continue;
    }
    help_2();
    std::cin >> command;
    std::cout << "searching......" << std::endl;
    std::vector<std::string> params_vector = split(command, PARAM_SEP);
    if (params_vector.size() < 2) continue;
    size_t iter = strtoull(params_vector[0].c_str(), NULL, 0);
    size_t batch = strtoull(params_vector[1].c_str(), NULL, 0);
    //size_t iter = 1;
    //size_t batch = 1;
    //std::cout << model_type << std::endl;
    score_type logloss_best = 1000.0;
    score_type auc_best = 0.5;
    std::vector<param_type> v_alpha = std::vector<param_type>{0,0.01,0.1,0.02,0};
    std::vector<param_type> v_lambda = std::vector<param_type>{0,0,0.03,0.01,0};
    std::vector<param_type> v_beta_1 = std::vector<param_type>{0,0,0,0.1,0};
    std::vector<param_type> v_beta_2 = std::vector<param_type>{0,0,0,0.03,0};
    if (model_type == LR_M_MODEL or model_type == LR_NAG_MODEL) {
      v_beta_1[2] = 0.55;
    } else if (model_type == LR_RMS_MODEL) {
      v_beta_2[1] = 0.9;
      v_beta_2[2] = 0.99;
    } else if (model_type == LR_ADAM_MODEL) {
      v_beta_1[1] = 0.8;
      v_beta_1[2] = 0.9;
      v_beta_1[3] = 0.03;
      v_beta_2[1] = 0.9;
      v_beta_2[2] = 0.99;
      v_beta_2[3] = 0.03;
    } else if (model_type == LR_ADAD_MODEL) {
      v_alpha[1] = 0;
      v_alpha[2] = 0;
      v_beta_1[1] = 0.8;
      v_beta_1[2] = 0.95;
      v_beta_1[3] = 0.03;
    }
    for (param_type alpha=v_alpha[1]; alpha<=v_alpha[2]; alpha+=v_alpha[3]) {
      for (param_type lambda=v_lambda[1]; lambda<=v_lambda[2]; lambda+=v_lambda[3]) {
        for (param_type beta_1=v_beta_1[1]; beta_1<=v_beta_1[2]; beta_1+=v_beta_1[3]) {
          for (param_type beta_2=v_beta_2[1]; beta_2<=v_beta_2[2]; beta_2+=v_beta_2[3]) {
            lr_instance->init(iter, batch, alpha, lambda, beta_1, beta_2);
            lr_instance->train();
            lr_instance->evaluate();
            score_type logloss = LRFactory::get_test_dataset()->cal_logloss();
            score_type auc = LRFactory::get_test_dataset()->cal_auc();
            if (auc_best < auc) {
              auc_best = auc;
              v_alpha[0] = alpha;
              v_lambda[0] = lambda;
              v_beta_1[0] = beta_1;
              v_beta_2[0] = beta_2;
            }
            if (logloss_best > logloss) {
              logloss_best = logloss;
              v_alpha[4] = alpha;
              v_lambda[4] = lambda;
              v_beta_1[4] = beta_1;
              v_beta_2[4] = beta_2;
            }
          }
        }
      }
    }
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "auc: " << auc_best << std::endl;
    std::cout << "alpha: " << v_alpha[0] << std::endl;
    std::cout << "lambda: " << v_lambda[0] << std::endl;
    std::cout << "beta_1: " << v_beta_1[0] << std::endl;
    std::cout << "beta_2: " << v_beta_2[0] << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "logloss: " << logloss_best << std::endl;
    std::cout << "alpha: " << v_alpha[4] << std::endl;
    std::cout << "lambda: " << v_lambda[4] << std::endl;
    std::cout << "beta_1: " << v_beta_1[4] << std::endl;
    std::cout << "beta_2: " << v_beta_2[4] << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    help_1();
  }
  return 0;
}
