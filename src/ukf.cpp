#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  //initial state dimension
  n_x_ = 5;

  //initial augmented state dimension
  n_aug_ = 7;

  //initial number of augmented sigma points
  amount_aug_points = 2 * n_aug_ + 1;
  
  //initial spreading parameter
  lambda_ = 3 - static_cast<int>(n_x_);

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_)*10;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3. / 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;
  Xsig_pred_ = MatrixXd(n_x_, amount_aug_points);
  weights_ = InitWeights(n_aug_, lambda_);
  time_us_ = 0;

  R_radar_ = InitRadarR(std_radr_, std_radphi_, std_radrd_);
  R_laser_ = InitLaserR(std_laspx_, std_laspy_);
  NIS_ = 0;

  outputLaser_.open("laserNIS.dat");
  outputRadar_.open("radarNIS.dat");
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {

  /*****************************************************************************
  *  Initialization
  ****************************************************************************/
  if (!is_initialized_) {
    std::cout << "UFK:\n";

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      const auto rho = meas_package.raw_measurements_[0];
      const auto phi = meas_package.raw_measurements_[1];
      const auto rho_dot = meas_package.raw_measurements_[2];
      x_(0) = rho*cos(phi);
      x_(1) = rho*sin(phi);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_(0) = meas_package.raw_measurements_[0];
      x_(1) = meas_package.raw_measurements_[1];
    }

    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
  *  Prediction
  ****************************************************************************/
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ ||
    meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    
    const double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
    time_us_ = meas_package.timestamp_;
    Prediction(dt);

    /*****************************************************************************
    *  Update
    ****************************************************************************/
    Update(meas_package);
    if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
      outputLaser_ << NIS_ << std::endl;
    }else if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      outputRadar_ << NIS_ << std::endl;
    }
    // print the output
    std::cout << "x_ = " << x_ << std::endl;
    std::cout << "P_ = " << P_ << std::endl;
  }
}

void UKF::Prediction(const double dt) {

  const MatrixXd Xsig_aug = AugmentedSigmaPoints();
  PredictSigmaPoints(Xsig_aug, dt);
  
  x_.fill(0);
  for (size_t i = 0; amount_aug_points > i; ++i) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0);
  for (size_t i = 0; amount_aug_points > i; ++i) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::Update(const MeasurementPackage& meas_package) {
  std::function<void(double&)> norm = [](double&) {return; };
  MatrixXd Zsig, R;

  if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    Zsig = TransformSig2MeasRadar();
    R = R_radar_;
    norm = NormalizeAngle;
  }else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    Zsig = TransformSig2MeasLaser();
    R = R_laser_;
  }
  const VectorXd z_pred = MeanPredictedMeas(Zsig);
  const MatrixXd S = CovariancePredictedMeas(z_pred, Zsig, norm) + R;
  const MatrixXd Tc = CrossCorrelation(z_pred, Zsig, norm);

  //Kalman gain K
  const MatrixXd K = Tc*S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  norm(z_diff(1));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  NIS_ = z_diff.transpose()*S.inverse()*z_diff;
}

MatrixXd UKF::AugmentedSigmaPoints() const {
  MatrixXd Xsig_aug = MatrixXd(n_aug_, amount_aug_points);
  
  //create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  //create augmented mean state
  x_aug.head(n_x_) = x_;

  //create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

  //create square root matrix
  const MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;

  const double coeff = sqrt(lambda_ + n_aug_);
  for (size_t i = 0; i < n_aug_; ++i)
  {
    Xsig_aug.col(i + 1) = x_aug + coeff * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - coeff * L.col(i);
  }

  return Xsig_aug;
}

void UKF::PredictSigmaPoints(const MatrixXd& Xsig_aug, const double dt) {
  //predict sigma points

  for (size_t i = 0; amount_aug_points > i; ++i)
  {
    //extract values for better readability
    const double p_x = Xsig_aug(0, i);
    const double p_y = Xsig_aug(1, i);
    const double v = Xsig_aug(2, i);
    const double yaw = Xsig_aug(3, i);
    const double yawd = Xsig_aug(4, i);
    const double nu_a = Xsig_aug(5, i);
    const double nu_yawdd = Xsig_aug(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd*dt) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*dt));
    }
    else {
      px_p = p_x + v*dt*cos(yaw);
      py_p = p_y + v*dt*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*dt;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*dt*dt * cos(yaw);
    py_p = py_p + 0.5*nu_a*dt*dt * sin(yaw);
    v_p = v_p + nu_a*dt;

    yaw_p = yaw_p + 0.5*nu_yawdd*dt*dt;
    yawd_p = yawd_p + nu_yawdd*dt;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

MatrixXd UKF::TransformSig2MeasRadar() const {
  MatrixXd Zsig = MatrixXd(3, amount_aug_points);

  //transform sigma points into measurement space
  for (size_t i = 0; amount_aug_points > i; ++i) {  //2n+1 simga points
    // extract values for better readibility
    const double p_x = Xsig_pred_(0, i);
    const double p_y = Xsig_pred_(1, i);
    const double v = Xsig_pred_(2, i);
    const double yaw = Xsig_pred_(3, i);

    const double v1 = cos(yaw)*v;
    const double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y); //r
    Zsig(1, i) = atan2(p_y, p_x); //phi
    Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y); //r_dot
  }
  return Zsig;
}

MatrixXd UKF::TransformSig2MeasLaser() const {
  MatrixXd Zsig = MatrixXd(2, amount_aug_points);

  //transform sigma points into measurement space
  for (size_t i = 0; amount_aug_points > i; ++i) {
    // measurement model
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }
  return Zsig;
}

VectorXd UKF::MeanPredictedMeas(const MatrixXd& Zsig) const {
  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(Zsig.rows());

  for (size_t i = 0; i < amount_aug_points; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  return z_pred;
}

template <typename NormFunc>
MatrixXd UKF::CovariancePredictedMeas(const VectorXd& zPred, const MatrixXd& Zsig, NormFunc norm) const {
  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(zPred.size(), zPred.size());
  
  for (size_t i = 0; amount_aug_points > i; ++i) {
    //residual
    VectorXd z_diff = Zsig.col(i) - zPred;

    //angle normalization in case processing of radar measurement
    //in case laser measurement this function does nothing
    norm(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  return S;
}

template <typename NormFunc>
MatrixXd UKF::CrossCorrelation(const VectorXd& zPred, const MatrixXd& Zsig, NormFunc norm) const {
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, zPred.size());
  for (size_t i = 0; amount_aug_points > i; ++i) {
    //residual
    VectorXd z_diff = Zsig.col(i) - zPred;
    
    //angle normalization in case processing of radar measurement
    //in case laser measurement this function does nothing
    norm(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  return Tc;
}

VectorXd UKF::InitWeights(const size_t n_aug, const double lambda) {
  //create vector for weights
  VectorXd weights = VectorXd(2 * n_aug + 1);
  
  // set weights
  weights(0) = lambda / (lambda + n_aug);
  
  const size_t amountSigmaPoints = 2 * n_aug + 1;
  const double weight = 0.5 / (n_aug + lambda);
  for (size_t i = 1; amountSigmaPoints > i; ++i) {
    weights(i) = weight;
  }

  return weights;
}

void UKF::NormalizeAngle(double& angle) {
  while (angle > M_PI) {
    angle -= 2.*M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.*M_PI;
  }
}

MatrixXd UKF::InitRadarR(const double std_radr, const double std_radphi, const double std_radrd) {
  MatrixXd R = MatrixXd(3, 3);
  R <<  std_radr*std_radr, 0, 0,
        0, std_radphi*std_radphi, 0,
        0, 0, std_radrd*std_radrd;
  return R;
}

MatrixXd UKF::InitLaserR(double std_laspx, double std_laspy) {
  MatrixXd R = MatrixXd(2, 2);
  R <<  std_laspx*std_laspx, 0,
        0, std_laspy*std_laspy;
  return R;
}
