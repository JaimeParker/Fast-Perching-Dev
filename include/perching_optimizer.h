// This is a refactored and improved version of the original optimizer implementation.
// Author: Zhaohong Liu

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#ifndef NDEBUG
#include <iostream>
#endif

#include "lbfgs_raw.hpp"
#include "minco.hpp"
#include "poly_traj_utils.hpp"
#include "traj_opt.h"

namespace traj_opt {

class PerchingOptimizer {
   public:
    PerchingOptimizer()
        : debug_pause_(false),
          num_pieces_(0),
          integration_steps_(20),
          time_var_dim_(0),
          waypoint_dim_(0),
          time_w_(1.0),
          tail_velocity_w_(-1.0),
          pos_w_(1.0),
          vel_w_(1.0),
          acc_w_(1.0),
          thrust_w_(1.0),
          omega_w_(1.0),
          perching_collision_w_(1.0),
          landing_speed_offset_(1.0),
          tail_length_(0.3),
          body_radius_(0.1),
          platform_radius_(0.5),
          max_thrust_(20.0),
          min_thrust_(2.0),
          max_omega_(3.0),
          max_yaw_omega_(2.0),
          max_vel_(10.0),
          max_acc_(10.0),
          optimization_vars_(nullptr),
          gravity_vec_(0.0, 0.0, -9.8),
          landing_vel_(Eigen::Vector3d::Zero()),
          landing_basis_x_(Eigen::Vector3d::Zero()),
          landing_basis_y_(Eigen::Vector3d::Zero()),
          initial_tail_angle_(0.0),
          initial_tail_velocity_params_(Eigen::Vector2d::Zero()),
          has_initial_guess_(false),
          thrust_mid_level_(0.0),
          thrust_half_range_(0.0),
          inner_loop_duration_(0.0),
          integral_duration_(0.0),
          iteration_count_(0) {
        target_pos_.setZero();
        target_vel_.setZero();
        landing_att_z_vec_.setZero();
    }

    ~PerchingOptimizer() {
        if (optimization_vars_ != nullptr) {
            delete[] optimization_vars_;
            optimization_vars_ = nullptr;
        }
    }

    void setDynamicLimits(double max_velocity,
                          double max_acceleration,
                          double max_thrust,
                          double min_thrust,
                          double max_omega,
                          double max_yaw_omega) {
        assert(max_velocity > 0.0 && "Max velocity must be positive");
        assert(max_acceleration > 0.0 && "Max acceleration must be positive");
        assert(max_thrust > min_thrust && "Max thrust must be greater than min thrust");
        assert(max_omega > 0.0 && "Max omega must be positive");
        assert(max_yaw_omega > 0.0 && "Max yaw omega must be positive");

        max_vel_ = max_velocity;
        max_acc_ = max_acceleration;
        max_thrust_ = max_thrust;
        min_thrust_ = min_thrust;
        max_omega_ = max_omega;
        max_yaw_omega_ = max_yaw_omega;
    }

    void setRobotParameters(double landing_speed_offset,
                            double tail_length,
                            double body_radius,
                            double platform_radius) {
        assert(landing_speed_offset > 0.0 && "Landing speed offset must be positive");
        assert(tail_length > 0.0 && "Tail length must be positive");
        assert(body_radius > 0.0 && "Body radius must be positive");
        assert(platform_radius > 0.0 && "Platform radius must be positive");

        landing_speed_offset_ = landing_speed_offset;
        tail_length_ = tail_length;
        body_radius_ = body_radius;
        platform_radius_ = platform_radius;
    }

    void setOptimizationWeights(double time_weight,
                                double tail_velocity_weight,
                                double position_weight,
                                double velocity_weight,
                                double acceleration_weight,
                                double thrust_weight,
                                double omega_weight,
                                double perching_collision_weight) {
        assert(time_weight >= 0.0 && "Time weight must be non-negative");
        assert(position_weight >= 0.0 && "Position weight must be non-negative");
        assert(velocity_weight >= 0.0 && "Velocity weight must be non-negative");
        assert(acceleration_weight >= 0.0 && "Acceleration weight must be non-negative");
        assert(thrust_weight >= 0.0 && "Thrust weight must be non-negative");
        assert(omega_weight >= 0.0 && "Omega weight must be non-negative");
        assert(perching_collision_weight >= 0.0 && "Perching collision weight must be non-negative");

        time_w_ = time_weight;
        tail_velocity_w_ = tail_velocity_weight;
        pos_w_ = position_weight;
        vel_w_ = velocity_weight;
        acc_w_ = acceleration_weight;
        thrust_w_ = thrust_weight;
        omega_w_ = omega_weight;
        perching_collision_w_ = perching_collision_weight;
    }

    void setIntegrationSteps(int integration_steps) {
        assert(integration_steps > 0 && "Integration steps must be positive");
        integration_steps_ = integration_steps;
    }

    void setDebugMode(bool debug_pause) {
        debug_pause_ = debug_pause;
    }

    bool generateTrajectory(const Eigen::MatrixXd& initial_state,
                            const Eigen::Vector3d& target_pos,
                            const Eigen::Vector3d& target_vel,
                            const Eigen::Quaterniond& landing_quat,
                            int num_pieces,
                            Trajectory& trajectory,
                            const double& replanning_time = -1.0) {
        assert(initial_state.rows() == 3 && initial_state.cols() == 4 && "Initial state must be 3x4 matrix");
        assert(num_pieces > 0 && "Number of pieces must be positive");
        assert(landing_quat.norm() > 0.99 && landing_quat.norm() < 1.01 && "Landing quaternion must be unit quaternion");

#ifndef NDEBUG
        std::cout << "[PerchingOptimizer] Starting trajectory generation with " << num_pieces << " pieces" << std::endl;
#endif

        num_pieces_ = num_pieces;
        time_var_dim_ = 1;
        waypoint_dim_ = num_pieces_ - 1;

        const int variable_count = time_var_dim_ + 3 * waypoint_dim_ + 1 + 2;
        if (optimization_vars_ != nullptr) {
            delete[] optimization_vars_;
            optimization_vars_ = nullptr;
        }
        optimization_vars_ = new double[variable_count];

        double& log_time_var = optimization_vars_[0];
        Eigen::Map<Eigen::MatrixXd> intermediate_waypoints(optimization_vars_ + time_var_dim_, 3, waypoint_dim_);
        double& tail_angle = optimization_vars_[time_var_dim_ + 3 * waypoint_dim_];
        Eigen::Map<Eigen::Vector2d> tail_velocity_params(optimization_vars_ + time_var_dim_ + 3 * waypoint_dim_ + 1);

        target_pos_ = target_pos;
        target_vel_ = target_vel;

        quaternionToZAxis(landing_quat, landing_att_z_vec_);

        thrust_mid_level_ = (max_thrust_ + min_thrust_) / 2.0;
        thrust_half_range_ = (max_thrust_ - min_thrust_) / 2.0;

        landing_vel_ = target_vel_ - landing_att_z_vec_ * landing_speed_offset_;

        landing_basis_x_ = landing_att_z_vec_.cross(Eigen::Vector3d(0.0, 0.0, 1.0));
        if (landing_basis_x_.squaredNorm() == 0.0) {
            landing_basis_x_ = landing_att_z_vec_.cross(Eigen::Vector3d(0.0, 1.0, 0.0));
        }
        landing_basis_x_.normalize();
        landing_basis_y_ = landing_att_z_vec_.cross(landing_basis_x_);
        landing_basis_y_.normalize();

        tail_velocity_params.setConstant(0.0);

        initial_state_matrix_ = initial_state;

        minco_optimizer_.reset(num_pieces_);

        tail_angle = 0.0;

        bool reuse_initial_guess = has_initial_guess_ && replanning_time > 0.0 &&
                                   replanning_time < initial_trajectory_.getTotalDuration();
        if (reuse_initial_guess) {
            double initial_total_duration = initial_trajectory_.getTotalDuration() - replanning_time;
            log_time_var = logC2(initial_total_duration / num_pieces_);

            for (int i = 1; i < num_pieces_; ++i) {
                double segment_time = (static_cast<double>(i) / num_pieces_) * initial_total_duration;
                intermediate_waypoints.col(i - 1) = initial_trajectory_.getPos(segment_time + replanning_time);
            }
            tail_angle = initial_tail_angle_;
            tail_velocity_params = initial_tail_velocity_params_;
        } else {
            Eigen::MatrixXd initial_boundary = initial_state_matrix_;
            Eigen::MatrixXd final_boundary(3, 4);
            final_boundary.col(0) = target_pos_;
            final_boundary.col(1) = target_vel_;
            final_boundary.col(2) = forwardThrust(tail_angle, thrust_half_range_, thrust_mid_level_) * landing_att_z_vec_ + gravity_vec_;
            final_boundary.col(3).setZero();

            double boundary_duration = (final_boundary.col(0) - initial_boundary.col(0)).norm() / max_vel_;
            CoefficientMat coefficient_matrix;
            double max_omega = 0.0;

            do {
                boundary_duration += 1.0;
                final_boundary.col(0) = target_pos_ + target_vel_ * boundary_duration;
                solveBoundaryValueProblem(boundary_duration, initial_boundary, final_boundary, coefficient_matrix);
                std::vector<double> durations{boundary_duration};
                std::vector<CoefficientMat> coefficients{coefficient_matrix};
                Trajectory boundary_traj(durations, coefficients);
                max_omega = getMaxOmega(boundary_traj, gravity_vec_);
            } while (max_omega > 1.5 * max_omega_);

            Eigen::VectorXd polynomial_terms(8);
            polynomial_terms(7) = 1.0;
            for (int i = 1; i < num_pieces_; ++i) {
                double segment_time = (static_cast<double>(i) / num_pieces_) * boundary_duration;
                for (int j = 6; j >= 0; --j) {
                    polynomial_terms(j) = polynomial_terms(j + 1) * segment_time;
                }
                intermediate_waypoints.col(i - 1) = coefficient_matrix * polynomial_terms;
            }
            log_time_var = logC2(boundary_duration / num_pieces_);
        }

        lbfgs::lbfgs_parameter_t lbfgs_params;
        lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
        lbfgs_params.mem_size = 32;
        lbfgs_params.past = 3;
        lbfgs_params.g_epsilon = 0.0;
        lbfgs_params.min_step = 1e-16;
        lbfgs_params.delta = 1e-4;
        lbfgs_params.line_search_type = 0;

        double min_objective = 0.0;
        int optimization_result = 0;

        inner_loop_duration_ = 0.0;
        integral_duration_ = 0.0;

        iteration_count_ = 0;

#ifndef NDEBUG
        auto tic = std::chrono::steady_clock::now();
#endif

        optimization_result = lbfgs::lbfgs_optimize(
            variable_count,
            optimization_vars_,
            &min_objective,
            &PerchingOptimizer::objectiveFunction,
            nullptr,
            &PerchingOptimizer::earlyExitCallback,
            this,
            &lbfgs_params);

#ifndef NDEBUG
        auto toc = std::chrono::steady_clock::now();
        std::cout << "[PerchingOptimizer] Optimization result: " << optimization_result << std::endl;
        std::cout << "[PerchingOptimizer] Optimization time: " << (toc - tic).count() * 1e-6 << "ms" << std::endl;
#endif

        if (debug_pause_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        if (optimization_result < 0) {
            delete[] optimization_vars_;
            optimization_vars_ = nullptr;
            return false;
        }

        double piece_duration = expC2(log_time_var);
        double total_duration = num_pieces_ * piece_duration;

#ifndef NDEBUG
        std::cout << "[PerchingOptimizer] Final result - duration: " << total_duration
                  << ", tail_angle: " << tail_angle << std::endl;
#endif

        Eigen::Vector3d tail_velocity;
        computeTailVelocity(tail_velocity_params, landing_vel_, landing_basis_x_, landing_basis_y_, tail_velocity);

        Eigen::MatrixXd tail_state(3, 4);
        tail_state.col(0) = target_pos_ + target_vel_ * total_duration + landing_att_z_vec_ * tail_length_;
        tail_state.col(1) = tail_velocity;
        tail_state.col(2) = forwardThrust(tail_angle, thrust_half_range_, thrust_mid_level_) * landing_att_z_vec_ + gravity_vec_;
        tail_state.col(3).setZero();

        minco_optimizer_.generate(initial_state_matrix_, tail_state, intermediate_waypoints, piece_duration);
        trajectory = minco_optimizer_.getTraj();

#ifndef NDEBUG
        double max_omega = getMaxOmega(trajectory, gravity_vec_);
        double max_thrust = trajectory.getMaxThrust();
        std::cout << "[PerchingOptimizer] Trajectory stats - maxOmega: " << max_omega
                  << ", maxThrust: " << max_thrust << std::endl;
#endif

        initial_trajectory_ = trajectory;
        initial_tail_angle_ = tail_angle;
        initial_tail_velocity_params_ = tail_velocity_params;
        has_initial_guess_ = true;

        delete[] optimization_vars_;
        optimization_vars_ = nullptr;

        return true;
    }

    bool feasibleCheck(Trajectory& trajectory) {
        double dt = 0.01;
        for (double t = 0.0; t < trajectory.getTotalDuration(); t += dt) {
            Eigen::Vector3d pos = trajectory.getPos(t);
            Eigen::Vector3d acc = trajectory.getAcc(t);
            Eigen::Vector3d jer = trajectory.getJer(t);
            Eigen::Vector3d thrust = acc - gravity_vec_;
            Eigen::Vector3d zb_dot = getNormalizationJacobian(thrust) * jer;
            double omega12 = zb_dot.norm();
            if (omega12 > max_omega_ + 0.2) {
                return false;
            }
            if (pos.z() < 0.1) {
                return false;
            }
        }
        return true;
    }

    std::vector<Eigen::Vector3d> tracking_ps_;
    std::vector<Eigen::Vector3d> tracking_visible_ps_;
    std::vector<double> tracking_thetas_;

   private:
    static bool quaternionToZAxis(const Eigen::Quaterniond& quat,
                                  Eigen::Vector3d& z_axis) {
        Eigen::Matrix3d rotation = quat.toRotationMatrix();
        z_axis = rotation.col(2);
        return true;
    }

    static Eigen::Vector3d normalizeVector(const Eigen::Vector3d& vec) {
        return vec.normalized();
    }

    static Eigen::Matrix3d getNormalizationJacobian(const Eigen::Vector3d& vec) {
        double norm_sq = vec.squaredNorm();
        return (Eigen::Matrix3d::Identity() - vec * vec.transpose() / norm_sq) / std::sqrt(norm_sq);
    }

    static Eigen::Matrix3d getNormalizationHessian(const Eigen::Vector3d& vec,
                                                   const Eigen::Vector3d& dir) {
        double norm_sq = vec.squaredNorm();
        double norm_cu = norm_sq * vec.norm();
        Eigen::Matrix3d term = (3.0 * vec * vec.transpose() / norm_sq - Eigen::Matrix3d::Identity());
        return (term * dir * vec.transpose() - vec * dir.transpose() -
                vec.dot(dir) * Eigen::Matrix3d::Identity()) /
               norm_cu;
    }

    static double smoothedL1(const double& value,
                             double& gradient) {
        static double mu = 0.01;
        if (value < 0.0) {
            return 0.0;
        } else if (value > mu) {
            gradient = 1.0;
            return value - 0.5 * mu;
        } else {
            const double ratio = value / mu;
            const double ratio_sq = ratio * ratio;
            const double mu_minus_half_value = mu - 0.5 * value;
            gradient = ratio_sq * ((-0.5) * ratio + 3.0 * mu_minus_half_value / mu);
            return mu_minus_half_value * ratio_sq * ratio;
        }
    }

    static double smoothedZeroOne(const double& value,
                                  double& gradient) {
        static double mu = 0.01;
        static double mu4 = mu * mu * mu * mu;
        static double inv_mu4 = 1.0 / mu4;
        if (value < -mu) {
            gradient = 0.0;
            return 0.0;
        } else if (value < 0.0) {
            double y = value + mu;
            double y2 = y * y;
            gradient = y2 * (mu - 2.0 * value) * inv_mu4;
            return 0.5 * y2 * y * (mu - value) * inv_mu4;
        } else if (value < mu) {
            double y = value - mu;
            double y2 = y * y;
            gradient = y2 * (mu + 2.0 * value) * inv_mu4;
            return 0.5 * y2 * y * (mu + value) * inv_mu4 + 1.0;
        } else {
            gradient = 0.0;
            return 1.0;
        }
    }

    static double expC2(double time_var) {
        return time_var > 0.0 ? ((0.5 * time_var + 1.0) * time_var + 1.0)
                              : 1.0 / ((0.5 * time_var - 1.0) * time_var + 1.0);
    }

    static double logC2(double duration) {
        return duration > 1.0 ? (std::sqrt(2.0 * duration - 1.0) - 1.0)
                              : (1.0 - std::sqrt(2.0 / duration - 1.0));
    }

    static double gradTimeTransform(double time_var) {
        if (time_var > 0.0) {
            return time_var + 1.0;
        } else {
            double denom = (0.5 * time_var - 1.0) * time_var + 1.0;
            return (1.0 - time_var) / (denom * denom);
        }
    }

    static double forwardThrust(double tail_angle,
                                double thrust_half_range,
                                double thrust_mid_level) {
        return thrust_half_range * std::sin(tail_angle) + thrust_mid_level;
    }

    static double propagateThrustGradient(double tail_angle,
                                          double thrust_gradient,
                                          double thrust_half_range) {
        return thrust_half_range * std::cos(tail_angle) * thrust_gradient;
    }

    static void computeTailVelocity(const Eigen::Vector2d& tail_velocity_params,
                                    const Eigen::Vector3d& landing_velocity,
                                    const Eigen::Vector3d& basis_x,
                                    const Eigen::Vector3d& basis_y,
                                    Eigen::Vector3d& tail_velocity) {
        tail_velocity = landing_velocity + tail_velocity_params.x() * basis_x + tail_velocity_params.y() * basis_y;
    }

    static double objectiveFunction(void* ptr_optimizer,
                                    const double* vars,
                                    double* grads,
                                    int n) {
        auto* optimizer = static_cast<PerchingOptimizer*>(ptr_optimizer);
        optimizer->iteration_count_++;

        const double& log_time_var = vars[0];
        double& grad_log_time = grads[0];

        Eigen::Map<const Eigen::MatrixXd> intermediate_waypoints(vars + optimizer->time_var_dim_, 3, optimizer->waypoint_dim_);
        Eigen::Map<Eigen::MatrixXd> grad_intermediate_waypoints(grads + optimizer->time_var_dim_, 3, optimizer->waypoint_dim_);

        const double& tail_angle = vars[optimizer->time_var_dim_ + optimizer->waypoint_dim_ * 3];
        double& grad_tail_angle = grads[optimizer->time_var_dim_ + optimizer->waypoint_dim_ * 3];

        Eigen::Map<const Eigen::Vector2d> tail_velocity_params(vars + optimizer->time_var_dim_ + optimizer->waypoint_dim_ * 3 + 1);
        Eigen::Map<Eigen::Vector2d> grad_tail_velocity_params(grads + optimizer->time_var_dim_ + optimizer->waypoint_dim_ * 3 + 1);

        double piece_duration = expC2(log_time_var);
        Eigen::Vector3d tail_velocity;
        computeTailVelocity(tail_velocity_params,
                            optimizer->landing_vel_,
                            optimizer->landing_basis_x_,
                            optimizer->landing_basis_y_,
                            tail_velocity);

        Eigen::MatrixXd tail_state(3, 4);
        tail_state.col(0) = optimizer->target_pos_ + optimizer->target_vel_ * optimizer->num_pieces_ * piece_duration +
                            optimizer->landing_att_z_vec_ * optimizer->tail_length_;
        tail_state.col(1) = tail_velocity;
        tail_state.col(2) = forwardThrust(tail_angle, optimizer->thrust_half_range_, optimizer->thrust_mid_level_) *
                                optimizer->landing_att_z_vec_ +
                            optimizer->gravity_vec_;
        tail_state.col(3).setZero();

        auto tic = std::chrono::steady_clock::now();
        optimizer->minco_optimizer_.generate(optimizer->initial_state_matrix_, tail_state, intermediate_waypoints, piece_duration);

        double cost = optimizer->minco_optimizer_.getTrajSnapCost();
        optimizer->minco_optimizer_.calGrads_CT();

        auto toc = std::chrono::steady_clock::now();
        optimizer->inner_loop_duration_ += (toc - tic).count();

        tic = std::chrono::steady_clock::now();
        optimizer->addTimeIntegralPenalty(cost);
        toc = std::chrono::steady_clock::now();
        optimizer->integral_duration_ += (toc - tic).count();

        tic = std::chrono::steady_clock::now();
        optimizer->minco_optimizer_.calGrads_PT();
        toc = std::chrono::steady_clock::now();
        optimizer->inner_loop_duration_ += (toc - tic).count();

        optimizer->minco_optimizer_.gdT += optimizer->minco_optimizer_.gdTail.col(0).dot(optimizer->num_pieces_ * optimizer->target_vel_);
        Eigen::Vector3d grad_tail_velocity = optimizer->minco_optimizer_.gdTail.col(1);
        double thrust_gradient = optimizer->minco_optimizer_.gdTail.col(2).dot(optimizer->landing_att_z_vec_);
        grad_tail_angle = propagateThrustGradient(tail_angle, thrust_gradient, optimizer->thrust_half_range_);

        grad_tail_velocity_params.setZero();
        if (optimizer->tail_velocity_w_ > -1.0) {
            grad_tail_velocity_params.x() = grad_tail_velocity.dot(optimizer->landing_basis_x_);
            grad_tail_velocity_params.y() = grad_tail_velocity.dot(optimizer->landing_basis_y_);
            double tail_velocity_norm_sq = tail_velocity_params.squaredNorm();
            cost += optimizer->tail_velocity_w_ * tail_velocity_norm_sq;
            grad_tail_velocity_params += optimizer->tail_velocity_w_ * 2.0 * tail_velocity_params;
        }

        optimizer->minco_optimizer_.gdT += optimizer->time_w_;
        cost += optimizer->time_w_ * piece_duration;
        grad_log_time = optimizer->minco_optimizer_.gdT * gradTimeTransform(log_time_var);

        grad_intermediate_waypoints = optimizer->minco_optimizer_.gdP;

        return cost;
    }

    static int earlyExitCallback(void* ptr_optimizer,
                                 const double* vars,
                                 const double* grads,
                                 const double fx,
                                 const double xnorm,
                                 const double gnorm,
                                 const double step,
                                 int n,
                                 int k,
                                 int ls) {
        auto* optimizer = static_cast<PerchingOptimizer*>(ptr_optimizer);
        if (optimizer->debug_pause_) {
            const double& log_time_var = vars[0];
            Eigen::Map<const Eigen::MatrixXd> intermediate_waypoints(vars + optimizer->time_var_dim_, 3, optimizer->waypoint_dim_);
            const double& tail_angle = vars[optimizer->time_var_dim_ + optimizer->waypoint_dim_ * 3];
            Eigen::Map<const Eigen::Vector2d> tail_velocity_params(vars + optimizer->time_var_dim_ + optimizer->waypoint_dim_ * 3 + 1);

            double piece_duration = expC2(log_time_var);
            double total_duration = optimizer->num_pieces_ * piece_duration;
            Eigen::Vector3d tail_velocity;
            computeTailVelocity(tail_velocity_params,
                                optimizer->landing_vel_,
                                optimizer->landing_basis_x_,
                                optimizer->landing_basis_y_,
                                tail_velocity);

            Eigen::MatrixXd tail_state(3, 4);
            tail_state.col(0) = optimizer->target_pos_ + optimizer->target_vel_ * total_duration +
                                optimizer->landing_att_z_vec_ * optimizer->tail_length_;
            tail_state.col(1) = tail_velocity;
            tail_state.col(2) = forwardThrust(tail_angle, optimizer->thrust_half_range_, optimizer->thrust_mid_level_) *
                                    optimizer->landing_att_z_vec_ +
                                optimizer->gravity_vec_;
            tail_state.col(3).setZero();

            optimizer->minco_optimizer_.generate(optimizer->initial_state_matrix_, tail_state, intermediate_waypoints, piece_duration);
            auto traj = optimizer->minco_optimizer_.getTraj();
            std::vector<Eigen::Vector3d> intermediate_points;
            for (const auto& piece : traj) {
                const auto& duration = piece.getDuration();
                for (int i = 0; i < optimizer->integration_steps_; ++i) {
                    double time = duration * i / optimizer->integration_steps_;
                    intermediate_points.push_back(piece.getPos(time));
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return 0;
    }

    static void solveBoundaryValueProblem(const double& duration,
                                          const Eigen::MatrixXd initial_state,
                                          const Eigen::MatrixXd final_state,
                                          CoefficientMat& coefficient_matrix) {
        double t1 = duration;
        double t2 = t1 * t1;
        double t3 = t2 * t1;
        double t4 = t2 * t2;
        double t5 = t3 * t2;
        double t6 = t3 * t3;
        double t7 = t4 * t3;

        CoefficientMat boundary_conditions;
        boundary_conditions.leftCols(4) = initial_state;
        boundary_conditions.rightCols(4) = final_state;

        coefficient_matrix.col(0) =
            (boundary_conditions.col(7) / 6.0 + boundary_conditions.col(3) / 6.0) * t3 +
            (-2.0 * boundary_conditions.col(6) + 2.0 * boundary_conditions.col(2)) * t2 +
            (10.0 * boundary_conditions.col(5) + 10.0 * boundary_conditions.col(1)) * t1 +
            (-20.0 * boundary_conditions.col(4) + 20.0 * boundary_conditions.col(0));
        coefficient_matrix.col(1) =
            (-0.5 * boundary_conditions.col(7) - boundary_conditions.col(3) / 1.5) * t3 +
            (6.5 * boundary_conditions.col(6) - 7.5 * boundary_conditions.col(2)) * t2 +
            (-34.0 * boundary_conditions.col(5) - 36.0 * boundary_conditions.col(1)) * t1 +
            (70.0 * boundary_conditions.col(4) - 70.0 * boundary_conditions.col(0));
        coefficient_matrix.col(2) =
            (0.5 * boundary_conditions.col(7) + boundary_conditions.col(3)) * t3 +
            (-7.0 * boundary_conditions.col(6) + 10.0 * boundary_conditions.col(2)) * t2 +
            (39.0 * boundary_conditions.col(5) + 45.0 * boundary_conditions.col(1)) * t1 +
            (-84.0 * boundary_conditions.col(4) + 84.0 * boundary_conditions.col(0));
        coefficient_matrix.col(3) =
            (-boundary_conditions.col(7) / 6.0 - boundary_conditions.col(3) / 1.5) * t3 +
            (2.5 * boundary_conditions.col(6) - 5.0 * boundary_conditions.col(2)) * t2 +
            (-15.0 * boundary_conditions.col(5) - 20.0 * boundary_conditions.col(1)) * t1 +
            (35.0 * boundary_conditions.col(4) - 35.0 * boundary_conditions.col(0));
        coefficient_matrix.col(4) = boundary_conditions.col(3) / 6.0;
        coefficient_matrix.col(5) = boundary_conditions.col(2) / 2.0;
        coefficient_matrix.col(6) = boundary_conditions.col(1);
        coefficient_matrix.col(7) = boundary_conditions.col(0);

        coefficient_matrix.col(0) = coefficient_matrix.col(0) / t7;
        coefficient_matrix.col(1) = coefficient_matrix.col(1) / t6;
        coefficient_matrix.col(2) = coefficient_matrix.col(2) / t5;
        coefficient_matrix.col(3) = coefficient_matrix.col(3) / t4;
    }

    static double getMaxOmega(Trajectory& trajectory,
                              const Eigen::Vector3d& gravity_vec) {
        double dt = 0.01;
        double max_omega = 0.0;
        for (double t = 0.0; t < trajectory.getTotalDuration(); t += dt) {
            Eigen::Vector3d acc = trajectory.getAcc(t);
            Eigen::Vector3d jer = trajectory.getJer(t);
            Eigen::Vector3d thrust = acc - gravity_vec;
            Eigen::Vector3d zb_dot = getNormalizationJacobian(thrust) * jer;
            double omega12 = zb_dot.norm();
            if (omega12 > max_omega) {
                max_omega = omega12;
            }
        }
        return max_omega;
    }

    void addTimeIntegralPenalty(double& cost) {
        Eigen::Vector3d pos, vel, acc, jer, snap;
        Eigen::Vector3d grad_temp_pos, grad_temp_vel, grad_temp_acc, grad_temp_jer, grad_temp_car;
        Eigen::Vector3d grad_pos_total, grad_vel_total, grad_acc_total, grad_jer_total;
        double cost_temp = 0.0;
        double cost_inner = 0.0;

        Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
        double sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7;
        double step = 0.0;
        double alpha = 0.0;
        Eigen::Matrix<double, 8, 3> grad_coeff;
        double grad_time = 0.0;
        double omega_weight = 0.0;

        int inner_loop = integration_steps_ + 1;
        step = minco_optimizer_.t(1) / integration_steps_;

        sigma1 = 0.0;

        for (int j = 0; j < inner_loop; ++j) {
            sigma2 = sigma1 * sigma1;
            sigma3 = sigma2 * sigma1;
            sigma4 = sigma2 * sigma2;
            sigma5 = sigma4 * sigma1;
            sigma6 = sigma4 * sigma2;
            sigma7 = sigma4 * sigma3;
            beta0 << 1.0, sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7;
            beta1 << 0.0, 1.0, 2.0 * sigma1, 3.0 * sigma2, 4.0 * sigma3, 5.0 * sigma4, 6.0 * sigma5, 7.0 * sigma6;
            beta2 << 0.0, 0.0, 2.0, 6.0 * sigma1, 12.0 * sigma2, 20.0 * sigma3, 30.0 * sigma4, 42.0 * sigma5;
            beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * sigma1, 60.0 * sigma2, 120.0 * sigma3, 210.0 * sigma4;
            beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * sigma1, 360.0 * sigma2, 840.0 * sigma3;
            alpha = 1.0 / integration_steps_ * j;
            omega_weight = (j == 0 || j == inner_loop - 1) ? 0.5 : 1.0;

            for (int i = 0; i < num_pieces_; ++i) {
                const auto& coeff_block = minco_optimizer_.c.block<8, 3>(i * 8, 0);

                pos = coeff_block.transpose() * beta0;
                vel = coeff_block.transpose() * beta1;
                acc = coeff_block.transpose() * beta2;
                jer = coeff_block.transpose() * beta3;
                snap = coeff_block.transpose() * beta4;

                grad_pos_total.setZero();
                grad_vel_total.setZero();
                grad_acc_total.setZero();
                grad_jer_total.setZero();
                grad_temp_car.setZero();
                cost_inner = 0.0;

                if (computeFloorCost(pos, grad_temp_pos, cost_temp)) {
                    grad_pos_total += grad_temp_pos;
                    cost_inner += cost_temp;
                }

                if (computeVelocityCost(vel, grad_temp_vel, cost_temp)) {
                    grad_vel_total += grad_temp_vel;
                    cost_inner += cost_temp;
                }

                if (computeThrustCost(acc, grad_temp_acc, cost_temp)) {
                    grad_acc_total += grad_temp_acc;
                    cost_inner += cost_temp;
                }

                if (computeOmegaCost(acc, jer, grad_temp_acc, grad_temp_jer, cost_temp)) {
                    grad_acc_total += grad_temp_acc;
                    grad_jer_total += grad_temp_jer;
                    cost_inner += cost_temp;
                }

                if (computeOmegaYawCost(acc, jer, grad_temp_acc, grad_temp_jer, cost_temp)) {
                    grad_acc_total += grad_temp_acc;
                    grad_jer_total += grad_temp_jer;
                    cost_inner += cost_temp;
                }

                double duration_to_now = (i + alpha) * minco_optimizer_.t(1);
                Eigen::Vector3d car_pos = target_pos_ + target_vel_ * duration_to_now;
                if (computePerchingCollisionCost(pos, acc, car_pos,
                                                 grad_temp_pos, grad_temp_acc, grad_temp_car,
                                                 cost_temp)) {
                    grad_pos_total += grad_temp_pos;
                    grad_acc_total += grad_temp_acc;
                    cost_inner += cost_temp;
                }
                double grad_car_time = grad_temp_car.dot(target_vel_);

                grad_coeff = beta0 * grad_pos_total.transpose();
                grad_time = grad_pos_total.transpose() * vel;
                grad_coeff += beta1 * grad_vel_total.transpose();
                grad_time += grad_vel_total.transpose() * acc;
                grad_coeff += beta2 * grad_acc_total.transpose();
                grad_time += grad_acc_total.transpose() * jer;
                grad_coeff += beta3 * grad_jer_total.transpose();
                grad_time += grad_jer_total.transpose() * snap;
                grad_time += grad_car_time;

                minco_optimizer_.gdC.block<8, 3>(i * 8, 0) += omega_weight * step * grad_coeff;
                minco_optimizer_.gdT += omega_weight * (cost_inner / integration_steps_ + alpha * step * grad_time);
                minco_optimizer_.gdT += i * omega_weight * step * grad_car_time;
                cost += omega_weight * step * cost_inner;
            }
            sigma1 += step;
        }
    }

    bool computeVelocityCost(const Eigen::Vector3d& velocity,
                             Eigen::Vector3d& grad_velocity,
                             double& cost_velocity) {
        double velocity_penalty = velocity.squaredNorm() - max_vel_ * max_vel_;
        if (velocity_penalty > 0.0) {
            double gradient = 0.0;
            cost_velocity = smoothedL1(velocity_penalty, gradient);
            grad_velocity = vel_w_ * gradient * 2.0 * velocity;
            cost_velocity *= vel_w_;
            return true;
        }
        return false;
    }

    bool computeThrustCost(const Eigen::Vector3d& acceleration,
                           Eigen::Vector3d& grad_acceleration,
                           double& cost_acceleration) {
        bool has_penalty = false;
        grad_acceleration.setZero();
        cost_acceleration = 0.0;
        Eigen::Vector3d thrust = acceleration - gravity_vec_;

        double max_penalty = thrust.squaredNorm() - max_thrust_ * max_thrust_;
        if (max_penalty > 0.0) {
            double gradient = 0.0;
            cost_acceleration = thrust_w_ * smoothedL1(max_penalty, gradient);
            grad_acceleration = thrust_w_ * 2.0 * gradient * thrust;
            has_penalty = true;
        }

        double min_penalty = min_thrust_ * min_thrust_ - thrust.squaredNorm();
        if (min_penalty > 0.0) {
            double gradient = 0.0;
            cost_acceleration = thrust_w_ * smoothedL1(min_penalty, gradient);
            grad_acceleration = -thrust_w_ * 2.0 * gradient * thrust;
            has_penalty = true;
        }

        return has_penalty;
    }

    bool computeOmegaCost(const Eigen::Vector3d& acceleration,
                          const Eigen::Vector3d& jerk,
                          Eigen::Vector3d& grad_acceleration,
                          Eigen::Vector3d& grad_jerk,
                          double& cost) {
        Eigen::Vector3d thrust = acceleration - gravity_vec_;
        Eigen::Vector3d zb_dot = getNormalizationJacobian(thrust) * jerk;
        double omega_sq = zb_dot.squaredNorm();
        double penalty = omega_sq - max_omega_ * max_omega_;
        if (penalty > 0.0) {
            double gradient = 0.0;
            cost = smoothedL1(penalty, gradient);

            Eigen::Vector3d grad_zb_dot = 2.0 * zb_dot;
            grad_jerk = getNormalizationJacobian(thrust).transpose() * grad_zb_dot;
            grad_acceleration = getNormalizationHessian(thrust, jerk).transpose() * grad_zb_dot;

            cost *= omega_w_;
            gradient *= omega_w_;
            grad_acceleration *= gradient;
            grad_jerk *= gradient;

            return true;
        }
        return false;
    }

    bool computeOmegaYawCost(const Eigen::Vector3d& acceleration,
                             const Eigen::Vector3d& jerk,
                             Eigen::Vector3d& grad_acceleration,
                             Eigen::Vector3d& grad_jerk,
                             double& cost) {
        (void)acceleration;
        (void)jerk;
        (void)grad_acceleration;
        (void)grad_jerk;
        (void)cost;
        return false;
    }

    bool computeFloorCost(const Eigen::Vector3d& position,
                          Eigen::Vector3d& grad_position,
                          double& cost_position) {
        static double z_floor = 0.4;
        double penalty = z_floor - position.z();
        if (penalty > 0.0) {
            double gradient = 0.0;
            cost_position = smoothedL1(penalty, gradient);
            cost_position *= pos_w_;
            grad_position.setZero();
            grad_position.z() = -pos_w_ * gradient;
            return true;
        }
        return false;
    }

    bool computePerchingCollisionCost(const Eigen::Vector3d& position,
                                      const Eigen::Vector3d& acceleration,
                                      const Eigen::Vector3d& car_position,
                                      Eigen::Vector3d& grad_position,
                                      Eigen::Vector3d& grad_acceleration,
                                      Eigen::Vector3d& grad_car_position,
                                      double& cost) {
        static double eps = 1e-6;

        double distance_sq = (position - car_position).squaredNorm();
        double safe_radius = platform_radius_ + body_radius_;
        double safe_radius_sq = safe_radius * safe_radius;
        double penalty_distance = safe_radius_sq - distance_sq;
        penalty_distance /= safe_radius_sq;
        double gradient_distance = 0.0;
        double smoothing = smoothedZeroOne(penalty_distance, gradient_distance);
        if (smoothing == 0.0) {
            return false;
        }

        Eigen::Vector3d grad_position_distance = gradient_distance * 2.0 * (car_position - position);
        Eigen::Vector3d grad_car_position_distance = -grad_position_distance;

        Eigen::Vector3d plane_normal = -landing_att_z_vec_;
        double plane_offset = plane_normal.dot(car_position);

        Eigen::Vector3d thrust = acceleration - gravity_vec_;
        Eigen::Vector3d body_z_vec = normalizeVector(thrust);

        Eigen::Matrix<double, 2, 3> matrix_btrt;
        double body_z_x = body_z_vec.x();
        double body_z_y = body_z_vec.y();
        double body_z_z = body_z_vec.z();

        double inv_one_plus_z = 1.0 / (1.0 + body_z_z);

        matrix_btrt(0, 0) = 1.0 - body_z_x * body_z_x * inv_one_plus_z;
        matrix_btrt(0, 1) = -body_z_x * body_z_y * inv_one_plus_z;
        matrix_btrt(0, 2) = -body_z_x;
        matrix_btrt(1, 0) = -body_z_x * body_z_y * inv_one_plus_z;
        matrix_btrt(1, 1) = 1.0 - body_z_y * body_z_y * inv_one_plus_z;
        matrix_btrt(1, 2) = -body_z_y;

        Eigen::Vector2d v2 = matrix_btrt * plane_normal;
        double v2_norm = std::sqrt(v2.squaredNorm() + eps);
        double penalty = plane_normal.dot(position) - (tail_length_ - 0.005) * plane_normal.dot(body_z_vec) -
                         plane_offset + body_radius_ * v2_norm;

        if (penalty > 0.0) {
            double gradient = 0.0;
            cost = smoothedL1(penalty, gradient);

            grad_position = plane_normal;
            grad_car_position = -plane_normal;
            Eigen::Vector2d grad_v2 = body_radius_ * v2 / v2_norm;

            Eigen::Matrix<double, 2, 3> dM_dax, dM_day, dM_daz;
            double inv_one_plus_z_sq = inv_one_plus_z * inv_one_plus_z;

            dM_dax(0, 0) = -2.0 * body_z_x * inv_one_plus_z;
            dM_dax(0, 1) = -body_z_y * inv_one_plus_z;
            dM_dax(0, 2) = -1.0;
            dM_dax(1, 0) = -body_z_y * inv_one_plus_z;
            dM_dax(1, 1) = 0.0;
            dM_dax(1, 2) = 0.0;

            dM_day(0, 0) = 0.0;
            dM_day(0, 1) = -body_z_x * inv_one_plus_z;
            dM_day(0, 2) = 0.0;
            dM_day(1, 0) = -body_z_x * inv_one_plus_z;
            dM_day(1, 1) = -2.0 * body_z_y * inv_one_plus_z;
            dM_day(1, 2) = -1.0;

            dM_daz(0, 0) = body_z_x * body_z_x * inv_one_plus_z_sq;
            dM_daz(0, 1) = body_z_x * body_z_y * inv_one_plus_z_sq;
            dM_daz(0, 2) = 0.0;
            dM_daz(1, 0) = body_z_x * body_z_y * inv_one_plus_z_sq;
            dM_daz(1, 1) = body_z_y * body_z_y * inv_one_plus_z_sq;
            dM_daz(1, 2) = 0.0;

            Eigen::Matrix<double, 2, 3> dv2_dzb;
            dv2_dzb.col(0) = dM_dax * plane_normal;
            dv2_dzb.col(1) = dM_day * plane_normal;
            dv2_dzb.col(2) = dM_daz * plane_normal;

            Eigen::Vector3d grad_body_z = dv2_dzb.transpose() * grad_v2 - tail_length_ * plane_normal;

            grad_acceleration = getNormalizationJacobian(thrust).transpose() * grad_body_z;

            gradient *= smoothing;
            grad_position_distance *= cost;
            grad_car_position_distance *= cost;
            cost *= smoothing;
            grad_position = gradient * grad_position + grad_position_distance;
            grad_acceleration *= gradient;
            grad_car_position = gradient * grad_car_position + grad_car_position_distance;

            cost *= perching_collision_w_;
            grad_position *= perching_collision_w_;
            grad_acceleration *= perching_collision_w_;
            grad_car_position *= perching_collision_w_;

            return true;
        }
        return false;
    }

    bool checkCollision(const Eigen::Vector3d& position,
                        const Eigen::Vector3d& acceleration,
                        const Eigen::Vector3d& car_position) {
        if ((position - car_position).norm() > platform_radius_) {
            return false;
        }

        static double eps = 1e-6;

        Eigen::Vector3d plane_normal = -landing_att_z_vec_;
        double plane_offset = plane_normal.dot(car_position);

        Eigen::Vector3d thrust = acceleration - gravity_vec_;
        Eigen::Vector3d body_z_vec = normalizeVector(thrust);

        Eigen::Matrix<double, 2, 3> matrix_btrt;
        double body_z_x = body_z_vec.x();
        double body_z_y = body_z_vec.y();
        double body_z_z = body_z_vec.z();

        double inv_one_plus_z = 1.0 / (1.0 + body_z_z);

        matrix_btrt(0, 0) = 1.0 - body_z_x * body_z_x * inv_one_plus_z;
        matrix_btrt(0, 1) = -body_z_x * body_z_y * inv_one_plus_z;
        matrix_btrt(0, 2) = -body_z_x;
        matrix_btrt(1, 0) = -body_z_x * body_z_y * inv_one_plus_z;
        matrix_btrt(1, 1) = 1.0 - body_z_y * body_z_y * inv_one_plus_z;
        matrix_btrt(1, 2) = -body_z_y;

        Eigen::Vector2d v2 = matrix_btrt * plane_normal;
        double v2_norm = std::sqrt(v2.squaredNorm() + eps);
        double penalty = plane_normal.dot(position) - (tail_length_ - 0.005) * plane_normal.dot(body_z_vec) -
                         plane_offset + body_radius_ * v2_norm;

        return penalty > 0.0;
    }

    bool debug_pause_;
    int num_pieces_;
    int integration_steps_;
    int time_var_dim_;
    int waypoint_dim_;

    double time_w_;
    double tail_velocity_w_;
    double pos_w_;
    double vel_w_;
    double acc_w_;
    double thrust_w_;
    double omega_w_;
    double perching_collision_w_;

    double landing_speed_offset_;
    double tail_length_;
    double body_radius_;
    double platform_radius_;

    double max_thrust_;
    double min_thrust_;
    double max_omega_;
    double max_yaw_omega_;
    double max_vel_;
    double max_acc_;

    minco::MINCO_S4_Uniform minco_optimizer_;
    Eigen::MatrixXd initial_state_matrix_;
    Eigen::VectorXd piece_durations_;
    double* optimization_vars_;

    Eigen::Vector3d target_pos_;
    Eigen::Vector3d target_vel_;
    Eigen::Vector3d landing_att_z_vec_;
    Eigen::Vector3d gravity_vec_;
    Eigen::Vector3d landing_vel_;
    Eigen::Vector3d landing_basis_x_;
    Eigen::Vector3d landing_basis_y_;

    Trajectory initial_trajectory_;
    double initial_tail_angle_;
    Eigen::Vector2d initial_tail_velocity_params_;
    bool has_initial_guess_;

    double thrust_mid_level_;
    double thrust_half_range_;

    double inner_loop_duration_;
    double integral_duration_;
    int iteration_count_;
};

}  // namespace traj_opt