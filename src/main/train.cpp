#define PARAM_SEP ","
#include "model_factory.h"

using namespace util;
using namespace model;

void help_1() {
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << "please input [M] to create a new model:" << std::endl;
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
}

void help_2() {
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << "please input model type (" << LR_MODEL
    << "/" << LR_M_MODEL
    << "/" << LR_NAG_MODEL
    << "/" << LR_AG_MODEL
    << "/" << LR_RMS_MODEL
    << "/" << LR_ADAM_MODEL
    << "/" << LR_ADAD_MODEL
    << "/" << FM_MODEL
    << "/" << FM_FC_MODEL
    << "):" << std::endl;
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
}

void help_3() {
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << "please input params separated by comma (iter,batch,alpha,lambda,beta_1,beta_2,fm_dims):" << std::endl;
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
}

void help_4() {
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
  std::cout << "new model[M] or tune parameters[P]?" << std::endl;
  std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << argv[0] << " train_data test_data" << std::endl;
    return -1;
  }
  path_type train_path = argv[1];
  path_type test_path = argv[2];
  ModelFactory::load_dataset(train_path, test_path);
  Model* lr_instance = NULL;
  std::string model_type;
  std::string command;
  help_1();
  while (std::cin >> command) {
    if (command == "M") {
      help_2();
      std::cin >> model_type; // remove '\n' by default
      lr_instance = ModelFactory::get_lr_instance(model_type);
    } else if (command == "P" && lr_instance) {
      help_3();
      std::cin >> command;
      std::vector<std::string> params_vector = split(command, PARAM_SEP);
      if (params_vector.size() < 7) continue;
      size_t iter = strtoull(params_vector[0].c_str(), NULL, 0);
      size_t batch = strtoull(params_vector[1].c_str(), NULL, 0);
      param_type alpha = atof(params_vector[2].c_str());
      param_type lambda = atof(params_vector[3].c_str());
      param_type beta_1 = atof(params_vector[4].c_str());
      param_type beta_2 = atof(params_vector[5].c_str());
      size_t k = atol(params_vector[6].c_str());
      lr_instance->init(iter, batch, alpha, lambda, beta_1, beta_2, k);
      lr_instance->train();
      lr_instance->evaluate();
    }
    help_4();
  }
  return 0;
}
