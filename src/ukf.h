#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:
  std::ofstream outputLaser_, outputRadar_;
  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  size_t n_x_;

  ///* Augmented state dimension
  size_t n_aug_;

  ///*Amount of augmented sigma points
  size_t amount_aug_points;

  ///* Sigma point spreading parameter
  double lambda_;

  ///* Measurement noise covariance matrix
  MatrixXd R_radar_;
  MatrixXd R_laser_;

  ///*Normalized innovation squared
  double NIS_;

  /**
   * Constructor
   */
  explicit UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(const MeasurementPackage& meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(const double delta_t);

private:
  /**
   * Updates the state and the state covariance matrix using a measurement
   * @param meas_package The measurement at k+1
   */
  void Update(const MeasurementPackage& meas_package);

  /**
   * \brief Generate augmented sigma points by state and state covariance
   * \return Augmented sigma points matrix
   */
  MatrixXd AugmentedSigmaPoints() const;

  /**
   * \brief Predict sigma points
   * \param Xsig_aug 
   */
  void PredictSigmaPoints(const MatrixXd& Xsig_aug, const double dt);
  
  /**
   * \brief Transform sigma points into measurement space for radar sensor
   * \return Matrix for sigma points in measurement space
   */
  MatrixXd TransformSig2MeasRadar() const;

  /**
  * \brief Transform sigma points into measurement space for laser sensor
  * \return Matrix for sigma points in measurement space
  */
  MatrixXd TransformSig2MeasLaser() const;

  /**
   * \brief Create mean predicted measurement
   * \param Zsig Sigma points into measurement space
   * \return Mean predicted measurement
   */
  VectorXd MeanPredictedMeas(const MatrixXd& Zsig) const;

  /**
   * \brief Create measurement covariance matrix S
   * \tparam NormFunc normalization function type
   * \param zPred Mean predicted measurement
   * \param Zsig Sigma points into measurement space
   * \param norm normalization function for radar sensor
   * \return Measurement covariance matrix
   */
  template <typename NormFunc>
  MatrixXd CovariancePredictedMeas(const VectorXd& zPred, const MatrixXd& Zsig, NormFunc norm) const;

  /**
   * \brief Create cross correlation matrix T
   * \tparam NormFunc normalization function type 
   * \param zPred Mean predicted measurement
   * \param Zsig Sigma points into measurement space
   * \param norm normalization function for radar sensor
   * \return Cross correlation matrix T
   */
  template <typename NormFunc>
  MatrixXd CrossCorrelation(const VectorXd& zPred, const MatrixXd& Zsig, NormFunc norm) const;

  /**
   * \brief Create weights of sigma points for prediction of mean and covariance
   * \param n_aug Augmented state dimension
   * \param lambda Sigma point spreading parameter
   * \return Weights vector of sigma points
   */
  static VectorXd InitWeights(const size_t n_aug, const double lambda);

  /**
   * \brief Normalization angle [-pi;pi]
   * \param angle Angle in rad
   */
  static void NormalizeAngle(double& angle);

  /**
   * \brief Create measurement noise matrix for radar sensor
   * \param std_radr Radar measurement noise standard deviation radius in m
   * \param std_radphi Radar measurement noise standard deviation angle in rad
   * \param std_radrd Radar measurement noise standard deviation radius change in m/s
   * \return measurement noise matrix
   */
  static MatrixXd InitRadarR(const double std_radr, const double std_radphi, const double std_radrd);

  /**
   * \brief Create measurement noise matrix for laser sensor
   * \param std_laspx Laser measurement noise standard deviation position 1 in m
   * \param std_laspy Laser measurement noise standard deviation position 2 in m
   * \return measurement noise matrix
   */
  static MatrixXd InitLaserR(double std_laspx, double std_laspy);
};

#endif /* UKF_H */
