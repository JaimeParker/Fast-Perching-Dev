// This code is part of Fast-Perching, https://github.com/ZJU-FAST-Lab/Fast-Perching
// All copyright belongs to the authors.

// This is a ros-free version by Zhaohong Liu

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include "minco.hpp"
#include "poly_traj_utils.hpp"

namespace traj_opt {

class TrajOpt {
   public:
    bool pause_debug_ = false;
    // # pieces and # key points
    int N_, K_, dim_t_, dim_p_;
    // weight for time regularization term
    double rhoT_, rhoVt_;
    // collision avoiding and dynamics paramters
    double vmax_, amax_;
    double rhoP_, rhoV_, rhoA_;
    double rhoThrust_, rhoOmega_;
    double rhoPerchingCollision_;
    // landing parameters
    double v_plus_, robot_l_, robot_r_, platform_r_;
    // SE3 dynamic limitation parameters
    double thrust_max_, thrust_min_;
    double omega_max_, omega_yaw_max_;
    // MINCO Optimizer
    minco::MINCO_S4_Uniform mincoOpt_;
    Eigen::MatrixXd initS_;
    // duration of each piece of the trajectory
    Eigen::VectorXd t_;
    double* x_;

    std::vector<Eigen::Vector3d> tracking_ps_;
    std::vector<Eigen::Vector3d> tracking_visible_ps_;
    std::vector<double> tracking_thetas_;

   public:
    TrajOpt();
    ~TrajOpt() {}

    // Configuration methods
    void setDynamicLimits(double vmax, double amax, double thrust_max, double thrust_min,
                          double omega_max, double omega_yaw_max);
    void setRobotParameters(double v_plus, double robot_l, double robot_r, double platform_r);
    void setOptimizationWeights(double rhoT, double rhoVt, double rhoP, double rhoV,
                                double rhoA, double rhoThrust, double rhoOmega, double rhoPerchingCollision);
    void setIntegrationSteps(int K);
    void setDebugMode(bool pause_debug);

    int optimize(const double& delta = 1e-4);
    bool generate_traj(const Eigen::MatrixXd& iniState,
                       const Eigen::Vector3d& car_p,
                       const Eigen::Vector3d& car_v,
                       const Eigen::Quaterniond& land_q,
                       const int& N,
                       Trajectory& traj,
                       const double& t_replan = -1.0);

    bool feasibleCheck(Trajectory& traj);

    void addTimeIntPenalty(double& cost);
    bool grad_cost_v(const Eigen::Vector3d& v,
                     Eigen::Vector3d& gradv,
                     double& costv);
    bool grad_cost_thrust(const Eigen::Vector3d& a,
                          Eigen::Vector3d& grada,
                          double& costa);
    bool grad_cost_omega(const Eigen::Vector3d& a,
                         const Eigen::Vector3d& j,
                         Eigen::Vector3d& grada,
                         Eigen::Vector3d& gradj,
                         double& cost);
    bool grad_cost_omega_yaw(const Eigen::Vector3d& a,
                             const Eigen::Vector3d& j,
                             Eigen::Vector3d& grada,
                             Eigen::Vector3d& gradj,
                             double& cost);
    bool grad_cost_floor(const Eigen::Vector3d& p,
                         Eigen::Vector3d& gradp,
                         double& costp);
    bool grad_cost_perching_collision(const Eigen::Vector3d& pos,
                                      const Eigen::Vector3d& acc,
                                      const Eigen::Vector3d& car_p,
                                      Eigen::Vector3d& gradp,
                                      Eigen::Vector3d& grada,
                                      Eigen::Vector3d& grad_car_p,
                                      double& cost);
    bool check_collilsion(const Eigen::Vector3d& pos,
                          const Eigen::Vector3d& acc,
                          const Eigen::Vector3d& car_p);
};

}  // namespace traj_opt