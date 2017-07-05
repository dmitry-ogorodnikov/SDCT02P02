#include "tools.h"
#include <iostream>
#include <functional>
#include <numeric>

using Eigen::VectorXd;
using std::vector;

//static
VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
  const vector<VectorXd> &ground_truth) {

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    std::cout << "Invalid estimation or ground_truth data" << std::endl;
    return rmse;
  }

  auto op2 = [](const VectorXd& a, const VectorXd& b)->VectorXd {
    const VectorXd residual = a - b;
    return residual.array()*residual.array();
  };

  rmse = std::inner_product(estimations.begin(), estimations.end(), ground_truth.begin(), rmse, std::plus<VectorXd>(), op2);
  rmse /= estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}
