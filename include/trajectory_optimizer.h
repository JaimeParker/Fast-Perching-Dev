// This is a refactored version of traj_opt by Zhaohong Liu
// Bugs may exist

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include "lbfgs_raw.hpp"
#include "minco.hpp"

namespace traj_opt {

class TrajectoryOptimizer {
   public:
    // Constructor with configuration parameters
    TrajectoryOptimizer() : num_pieces_(0), integration_steps_(20), time_dimension_(1), waypoint_dimension_(0), optimization_variables_(nullptr), max_velocity_(10.0), max_acceleration_(10.0), max_thrust_(20.0), min_thrust_(2.0), max_angular_velocity_(3.0), max_yaw_angular_velocity_(2.0), velocity_plus_(1.0), robot_length_(0.3), robot_radius_(0.1), platform_radius_(0.5), time_weight_(1.0), velocity_tail_weight_(-1.0), position_weight_(1.0), velocity_weight_(1.0), acceleration_weight_(1.0), thrust_weight_(1.0), angular_velocity_weight_(1.0), perching_collision_weight_(1.0), gravity_(0, 0, -9.8), has_initial_guess_(false), thrust_middle_(0.0), thrust_half_(0.0), timing_inner_loop_(0.0), timing_integral_(0.0), iteration_count_(0), debug_mode_(false), current_position_(Eigen::Vector3d::Zero()), current_velocity_(Eigen::Vector3d::Zero()), tail_quaternion_vector_(Eigen::Vector3d::Zero()), landing_velocity_(Eigen::Vector3d::Zero()), velocity_tangent_x_(Eigen::Vector3d::Zero()), velocity_tangent_y_(Eigen::Vector3d::Zero()), initial_tail_thrust_(0.0), initial_velocity_tangent_(Eigen::Vector2d::Zero()) {}

    ~TrajectoryOptimizer() {
        if (optimization_variables_) {
            delete[] optimization_variables_;
        }
    }

    // Main trajectory generation function
    bool generateTrajectory(const Eigen::MatrixXd& initial_state,
                            const Eigen::Vector3d& target_position,
                            const Eigen::Vector3d& target_velocity,
                            const Eigen::Quaterniond& landing_quaternion,
                            const int& num_pieces,
                            Trajectory& result_trajectory,
                            const double& replan_time = -1.0) {
        std::cout << "[TrajectoryOptimizer] generateTrajectory START" << std::endl;
        std::cout << "[TrajectoryOptimizer] Input initial_state:\n"
                  << initial_state << std::endl;
        std::cout << "[TrajectoryOptimizer] Input target_position: " << target_position.transpose() << std::endl;
        std::cout << "[TrajectoryOptimizer] Input target_velocity: " << target_velocity.transpose() << std::endl;
        std::cout << "[TrajectoryOptimizer] Input landing_quaternion: [" << landing_quaternion.w() << ", " << landing_quaternion.x() << ", " << landing_quaternion.y() << ", " << landing_quaternion.z() << "]" << std::endl;
        std::cout << "[TrajectoryOptimizer] Input num_pieces: " << num_pieces << std::endl;

        num_pieces_ = num_pieces;
        time_dimension_ = 1;
        waypoint_dimension_ = num_pieces_ - 1;

        // Allocate optimization variables: time + waypoints + tail_thrust + tail_velocity_tangent
        optimization_variables_ = new double[time_dimension_ + 3 * waypoint_dimension_ + 1 + 2];

        double& time_var = optimization_variables_[0];
        Eigen::Map<Eigen::MatrixXd> waypoints(optimization_variables_ + time_dimension_, 3, waypoint_dimension_);
        double& tail_thrust = optimization_variables_[time_dimension_ + 3 * waypoint_dimension_];
        Eigen::Map<Eigen::Vector2d> velocity_tangent(optimization_variables_ + time_dimension_ + 3 * waypoint_dimension_ + 1);

        // Set current state
        current_position_ = target_position;
        current_velocity_ = target_velocity;
        std::cout << "[TrajectoryOptimizer] Set current_position_: " << current_position_.transpose() << std::endl;
        std::cout << "[TrajectoryOptimizer] Set current_velocity_: " << current_velocity_.transpose() << std::endl;

        // Convert quaternion to vector
        quaternionToVector(landing_quaternion, tail_quaternion_vector_);
        std::cout << "[TrajectoryOptimizer] tail_quaternion_vector_: " << tail_quaternion_vector_.transpose() << std::endl;

        // Set thrust parameters
        thrust_middle_ = (max_thrust_ + min_thrust_) / 2.0;
        thrust_half_ = (max_thrust_ - min_thrust_) / 2.0;
        std::cout << "[TrajectoryOptimizer] thrust_middle_: " << thrust_middle_ << ", thrust_half_: " << thrust_half_ << std::endl;

        // Calculate landing velocity
        landing_velocity_ = current_velocity_ - tail_quaternion_vector_ * velocity_plus_;
        std::cout << "[TrajectoryOptimizer] landing_velocity_: " << landing_velocity_.transpose() << std::endl;

        // Calculate tangent vectors
        velocity_tangent_x_ = tail_quaternion_vector_.cross(Eigen::Vector3d(0, 0, 1));
        if (velocity_tangent_x_.squaredNorm() == 0) {
            velocity_tangent_x_ = tail_quaternion_vector_.cross(Eigen::Vector3d(0, 1, 0));
        }
        velocity_tangent_x_.normalize();
        velocity_tangent_y_ = tail_quaternion_vector_.cross(velocity_tangent_x_);
        velocity_tangent_y_.normalize();
        std::cout << "[TrajectoryOptimizer] velocity_tangent_x_: " << velocity_tangent_x_.transpose() << std::endl;
        std::cout << "[TrajectoryOptimizer] velocity_tangent_y_: " << velocity_tangent_y_.transpose() << std::endl;

        velocity_tangent.setConstant(0.0);
        std::cout << "[TrajectoryOptimizer] velocity_tangent initialized: " << velocity_tangent.transpose() << std::endl;

        // Set boundary conditions
        initial_state_matrix_ = initial_state;
        std::cout << "[TrajectoryOptimizer] initial_state_matrix_:\n"
                  << initial_state_matrix_ << std::endl;

        // Initialize MINCO optimizer
        minco_optimizer_.reset(num_pieces_);

        // Update static variables for use in static callback functions
        updateStaticVariables();

        tail_thrust = 0.0;
        std::cout << "[TrajectoryOptimizer] tail_thrust initialized: " << tail_thrust << std::endl;

        // Initial guess logic
        bool use_warm_start = has_initial_guess_ && replan_time > 0 &&
                              replan_time < initial_trajectory_.getTotalDuration();

        if (use_warm_start) {
            // Use previous trajectory as warm start
            double remaining_time = initial_trajectory_.getTotalDuration() - replan_time;
            time_var = getLogarithmicC2(remaining_time / num_pieces_);
            tail_thrust = initial_tail_thrust_;
            velocity_tangent = initial_velocity_tangent_;

            // Extract waypoints from previous trajectory
            for (int i = 0; i < waypoint_dimension_; ++i) {
                double sample_time = replan_time + (i + 1) * remaining_time / num_pieces_;
                waypoints.col(i) = initial_trajectory_.getPos(sample_time);
            }
        } else {
            // Initialize with feasible trajectory using boundary value problem
            Eigen::MatrixXd initial_bvp = initial_state_matrix_;
            Eigen::MatrixXd final_bvp(3, 4);
            final_bvp.col(0) = current_position_;
            final_bvp.col(1) = current_velocity_;
            final_bvp.col(2) = getForwardThrust(tail_thrust) * tail_quaternion_vector_ + gravity_;
            final_bvp.col(3).setZero();

            // Start with distance-based time estimate
            double T_bvp = (final_bvp.col(0) - initial_bvp.col(0)).norm() / max_velocity_;
            CoefficientMat coeff_matrix;
            double max_omega = 0;

            // Iteratively increase T_bvp until max angular velocity is feasible
            do {
                T_bvp += 1.0;
                final_bvp.col(0) = current_position_ + current_velocity_ * T_bvp;
                solveBoundaryValueProblem(T_bvp, initial_bvp, final_bvp, coeff_matrix);

                // Create temporary trajectory to check angular velocity
                std::vector<double> durations{T_bvp};
                std::vector<CoefficientMat> coeffs{coeff_matrix};
                Trajectory temp_traj(durations, coeffs);
                max_omega = getMaximumAngularVelocity(temp_traj);
            } while (max_omega > 1.5 * max_angular_velocity_);

            // Extract waypoints using polynomial evaluation
            Eigen::VectorXd time_powers(8);
            time_powers(7) = 1.0;
            for (int i = 0; i < waypoint_dimension_; ++i) {
                double sample_time = (i + 1.0) / num_pieces_ * T_bvp;
                for (int j = 6; j >= 0; j--) {
                    time_powers(j) = time_powers(j + 1) * sample_time;
                }
                waypoints.col(i) = coeff_matrix * time_powers;
            }

            time_var = getLogarithmicC2(T_bvp / num_pieces_);
        }

        // Setup L-BFGS parameters
        lbfgs::lbfgs_parameter_t lbfgs_params;
        lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
        lbfgs_params.mem_size = 32;
        lbfgs_params.past = 3;
        lbfgs_params.g_epsilon = 0.0;
        lbfgs_params.min_step = 1e-16;
        lbfgs_params.delta = 1e-4;
        lbfgs_params.line_search_type = 0;

        double min_objective;

        auto start_time = std::chrono::steady_clock::now();
        timing_inner_loop_ = 0.0;
        timing_integral_ = 0.0;
        iteration_count_ = 0;

        // Run optimization
        int optimization_result = lbfgs::lbfgs_optimize(
            time_dimension_ + 3 * waypoint_dimension_ + 1 + 2,
            optimization_variables_,
            &min_objective,
            &getObjectiveFunction,
            nullptr,
            &getEarlyExitCondition,
            this,
            &lbfgs_params);

        auto end_time = std::chrono::steady_clock::now();

        if (debug_mode_) {
            std::cout << "Optimization result: " << optimization_result << std::endl;
            std::cout << "Optimization time: " << (end_time - start_time).count() * 1e-6 << "ms" << std::endl;
        }

        if (optimization_result < 0) {
            return false;
        }

        // Generate final trajectory
        double delta_time = getExponentialC2(time_var);
        std::cout << "[TrajectoryOptimizer] Final optimization result - time_var: " << time_var << ", delta_time: " << delta_time << std::endl;
        std::cout << "[TrajectoryOptimizer] Final tail_thrust: " << tail_thrust << ", velocity_tangent: " << velocity_tangent.transpose() << std::endl;

        Eigen::Vector3d tail_velocity;
        getForwardTailVelocity(velocity_tangent, tail_velocity);

        Eigen::MatrixXd tail_state(3, 4);
        tail_state.col(0) = current_position_ + current_velocity_ * num_pieces_ * delta_time +
                            tail_quaternion_vector_ * robot_length_;
        tail_state.col(1) = tail_velocity;
        tail_state.col(2) = getForwardThrust(tail_thrust) * tail_quaternion_vector_ + gravity_;
        tail_state.col(3).setZero();

        std::cout << "[TrajectoryOptimizer] gravity_: " << gravity_.transpose() << std::endl;
        std::cout << "[TrajectoryOptimizer] thrust_middle_: " << thrust_middle_ << ", thrust_half_: " << thrust_half_ << std::endl;
        std::cout << "[TrajectoryOptimizer] tail_thrust: " << tail_thrust << ", sin(tail_thrust): " << sin(tail_thrust) << std::endl;
        std::cout << "[TrajectoryOptimizer] Manual calculation: " << thrust_half_ << " * " << sin(tail_thrust) << " + " << thrust_middle_ << " = " << (thrust_half_ * sin(tail_thrust) + thrust_middle_) << std::endl;
        std::cout << "[TrajectoryOptimizer] getForwardThrust(tail_thrust) * tail_quaternion_vector_: " << (getForwardThrust(tail_thrust) * tail_quaternion_vector_).transpose() << std::endl;
        std::cout << "[TrajectoryOptimizer] Final tail_state matrix:\n"
                  << tail_state << std::endl;
        std::cout << "[TrajectoryOptimizer] Final getForwardThrust(tail_thrust): " << getForwardThrust(tail_thrust) << std::endl;
        std::cout << "[TrajectoryOptimizer] Final tail_velocity: " << tail_velocity.transpose() << std::endl;

        minco_optimizer_.generate(initial_state_matrix_, tail_state, waypoints, delta_time);
        result_trajectory = minco_optimizer_.getTraj();

        // Store for warm start
        initial_trajectory_ = result_trajectory;
        initial_tail_thrust_ = tail_thrust;
        initial_velocity_tangent_ = velocity_tangent;
        has_initial_guess_ = true;

        delete[] optimization_variables_;
        optimization_variables_ = nullptr;

        return true;
    }

    // Configuration setters
    void setDynamicLimits(double max_velocity, double max_acceleration,
                          double max_thrust, double min_thrust,
                          double max_angular_velocity, double max_yaw_angular_velocity) {
        max_velocity_ = max_velocity;
        max_acceleration_ = max_acceleration;
        max_thrust_ = max_thrust;
        min_thrust_ = min_thrust;
        max_angular_velocity_ = max_angular_velocity;
        max_yaw_angular_velocity_ = max_yaw_angular_velocity;

        std::cout << "[TrajectoryOptimizer] setDynamicLimits: max_velocity=" << max_velocity_ << ", max_acceleration=" << max_acceleration_
                  << ", max_thrust=" << max_thrust_ << ", min_thrust=" << min_thrust_
                  << ", max_angular_velocity=" << max_angular_velocity_ << ", max_yaw_angular_velocity=" << max_yaw_angular_velocity_ << std::endl;
    }

    void setRobotParameters(double velocity_plus, double robot_length,
                            double robot_radius, double platform_radius) {
        velocity_plus_ = velocity_plus;
        robot_length_ = robot_length;
        robot_radius_ = robot_radius;
        platform_radius_ = platform_radius;

        std::cout << "[TrajectoryOptimizer] setRobotParameters: velocity_plus=" << velocity_plus_ << ", robot_length=" << robot_length_
                  << ", robot_radius=" << robot_radius_ << ", platform_radius=" << platform_radius_ << std::endl;
    }

    void setOptimizationWeights(double time_weight, double velocity_tail_weight,
                                double position_weight, double velocity_weight,
                                double acceleration_weight, double thrust_weight,
                                double angular_velocity_weight, double perching_collision_weight) {
        time_weight_ = time_weight;
        velocity_tail_weight_ = velocity_tail_weight;
        position_weight_ = position_weight;
        velocity_weight_ = velocity_weight;
        acceleration_weight_ = acceleration_weight;
        thrust_weight_ = thrust_weight;
        angular_velocity_weight_ = angular_velocity_weight;
        perching_collision_weight_ = perching_collision_weight;

        std::cout << "[TrajectoryOptimizer] setOptimizationWeights: time_weight=" << time_weight_ << ", velocity_tail_weight=" << velocity_tail_weight_
                  << ", position_weight=" << position_weight_ << ", velocity_weight=" << velocity_weight_ << ", acceleration_weight=" << acceleration_weight_
                  << ", thrust_weight=" << thrust_weight_ << ", angular_velocity_weight=" << angular_velocity_weight_
                  << ", perching_collision_weight=" << perching_collision_weight_ << std::endl;
    }

    void setIntegrationParameters(int integration_steps) {
        integration_steps_ = integration_steps;

        std::cout << "[TrajectoryOptimizer] setIntegrationParameters: integration_steps=" << integration_steps_ << std::endl;
    }

    void setDebugMode(bool enable_debug) {
        debug_mode_ = enable_debug;
    }

    // Feasibility checking
    bool checkFeasibility(Trajectory& trajectory) {
        double dt = 0.01;
        for (double t = 0; t < trajectory.getTotalDuration(); t += dt) {
            Eigen::Vector3d velocity = trajectory.getVel(t);
            Eigen::Vector3d acceleration = trajectory.getAcc(t);

            // Check velocity limits
            if (velocity.norm() > max_velocity_) {
                return false;
            }

            // Check thrust limits
            Eigen::Vector3d thrust = acceleration - gravity_;
            if (thrust.norm() > max_thrust_ || thrust.norm() < min_thrust_) {
                return false;
            }

            // Check angular velocity limits
            Eigen::Vector3d jerk = trajectory.getJer(t);
            Eigen::Vector3d thrust_normalized = normalizeVector(thrust);
            Eigen::Vector3d angular_velocity_vector = getNormalizationDerivative(thrust) * jerk;
            if (angular_velocity_vector.norm() > max_angular_velocity_) {
                return false;
            }
        }
        return true;
    }

    // Collision checking
    bool checkCollision(const Eigen::Vector3d& position,
                        const Eigen::Vector3d& acceleration,
                        const Eigen::Vector3d& target_position) {
        if ((position - target_position).norm() > platform_radius_) {
            return false;
        }

        static double eps = 1e-6;

        Eigen::Vector3d plane_normal = -tail_quaternion_vector_;
        double plane_offset = plane_normal.dot(target_position);

        Eigen::Vector3d thrust_force = acceleration - gravity_;
        Eigen::Vector3d body_z = normalizeVector(thrust_force);

        // Compute rotation matrix from body frame to world frame
        Eigen::MatrixXd body_to_world_rotation(2, 3);
        double a = body_z.x();
        double b = body_z.y();
        double c = body_z.z();
        double c_inv = 1.0 / (1.0 + c);

        body_to_world_rotation(0, 0) = 1.0 - a * a * c_inv;
        body_to_world_rotation(0, 1) = -a * b * c_inv;
        body_to_world_rotation(0, 2) = -a;
        body_to_world_rotation(1, 0) = -a * b * c_inv;
        body_to_world_rotation(1, 1) = 1.0 - b * b * c_inv;
        body_to_world_rotation(1, 2) = -b;

        Eigen::Vector2d projected_normal = body_to_world_rotation * plane_normal;
        double projected_normal_norm = sqrt(projected_normal.squaredNorm() + eps);

        double penetration = plane_normal.dot(position) - robot_length_ * plane_normal.dot(body_z) -
                             plane_offset + robot_radius_ * projected_normal_norm;

        return penetration > 0;
    }

   private:
    // Core optimization parameters
    int num_pieces_;
    int integration_steps_;
    int time_dimension_;
    int waypoint_dimension_;
    double* optimization_variables_;

    // Dynamic limits
    double max_velocity_;
    double max_acceleration_;
    double max_thrust_;
    double min_thrust_;
    double max_angular_velocity_;
    double max_yaw_angular_velocity_;

    // Robot physical parameters
    double velocity_plus_;
    double robot_length_;
    double robot_radius_;
    double platform_radius_;

    // Optimization weights
    double time_weight_;
    double velocity_tail_weight_;
    double position_weight_;
    double velocity_weight_;
    double acceleration_weight_;
    double thrust_weight_;
    double angular_velocity_weight_;
    double perching_collision_weight_;

    // Internal state variables
    Eigen::Vector3d current_position_;
    Eigen::Vector3d current_velocity_;
    Eigen::Vector3d tail_quaternion_vector_;
    Eigen::Vector3d gravity_;
    Eigen::Vector3d landing_velocity_;
    Eigen::Vector3d velocity_tangent_x_;
    Eigen::Vector3d velocity_tangent_y_;

    // Optimization state
    Trajectory initial_trajectory_;
    double initial_tail_thrust_;
    Eigen::Vector2d initial_velocity_tangent_;
    bool has_initial_guess_;

    double thrust_middle_;
    double thrust_half_;

    // Timing variables
    double timing_inner_loop_;
    double timing_integral_;
    int iteration_count_;

    // Debug
    bool debug_mode_;

    // MINCO optimizer
    minco::MINCO_S4_Uniform minco_optimizer_;
    Eigen::MatrixXd initial_state_matrix_;

    // Helper function for polynomial evaluation
    Eigen::Vector3d evaluatePolynomial(const CoefficientMat& coeffs, double t) {
        Eigen::Vector3d result = Eigen::Vector3d::Zero();
        double t_power = 1.0;
        for (int i = 7; i >= 0; --i) {
            result += coeffs.col(i) * t_power;
            if (i > 0) t_power *= t;
        }
        return result;
    }

    // Static helper functions for optimization
    static bool quaternionToVector(const Eigen::Quaterniond& quaternion, Eigen::Vector3d& vector) {
        Eigen::MatrixXd rotation_matrix = quaternion.toRotationMatrix();
        vector = rotation_matrix.col(2);
        return true;
    }

    static Eigen::Vector3d normalizeVector(const Eigen::Vector3d& vector) {
        return vector.normalized();
    }

    static Eigen::MatrixXd getNormalizationDerivative(const Eigen::Vector3d& vector) {
        double squared_norm = vector.squaredNorm();
        return (Eigen::MatrixXd::Identity(3, 3) - vector * vector.transpose() / squared_norm) / sqrt(squared_norm);
    }

    static Eigen::MatrixXd getSecondNormalizationDerivative(const Eigen::Vector3d& vector, const Eigen::Vector3d& direction) {
        double squared_norm = vector.squaredNorm();
        double cubed_norm = squared_norm * vector.norm();
        Eigen::MatrixXd A = (3.0 * vector * vector.transpose() / squared_norm - Eigen::MatrixXd::Identity(3, 3));
        return (A * direction * vector.transpose() - vector * direction.transpose() -
                vector.dot(direction) * Eigen::MatrixXd::Identity(3, 3)) /
               cubed_norm;
    }

    // Smoothing functions
    // static double getSmoothedL1(const double& value, double& gradient) {
    //     static double mu = 0.01;
    //     if (value < 0.0) {
    //         gradient = 0.0;
    //         return 0.0;
    //     } else if (value > mu) {
    //         gradient = 1.0;
    //         return value - 0.5 * mu;
    //     } else {
    //         const double normalized_value = value / mu;
    //         const double squared_normalized = normalized_value * normalized_value;
    //         const double mu_minus_half_x = mu - 0.5 * value;
    //         gradient = squared_normalized * ((-0.5) * normalized_value + 3.0 * mu_minus_half_x / mu);
    //         return mu_minus_half_x * squared_normalized * normalized_value;
    //     }
    // }

    static double getSmoothedL1(const double& value, double& gradient) {
        // IMPORTANT: This version intentionally replicates a bug from the original code
        // where the gradient is NOT set to 0 for value < 0. This is necessary for 1-to-1 matching.
        static double mu = 0.01;
        if (value < 0.0) {
            return 0.0;
        } else if (value > mu) {
            gradient = 1.0;
            return value - 0.5 * mu;
        } else {
            const double normalized_value = value / mu;
            const double squared_normalized = normalized_value * normalized_value;
            const double mu_minus_half_x = mu - 0.5 * value;
            gradient = squared_normalized * ((-0.5) * normalized_value + 3.0 * mu_minus_half_x / mu);
            return mu_minus_half_x * squared_normalized * normalized_value;
        }
    }

    static double getSmoothed01(const double& value, double& gradient) {
        static double mu = 0.01;
        static double mu4 = mu * mu * mu * mu;
        static double mu4_inv = 1.0 / mu4;

        if (value < -mu) {
            gradient = 0.0;
            return 0.0;
        } else if (value < 0.0) {
            double y = value + mu;
            double y2 = y * y;
            gradient = y2 * (mu - 2.0 * value) * mu4_inv;
            return 0.5 * y2 * y * (mu - value) * mu4_inv;
        } else if (value < mu) {
            double y = value - mu;
            double y2 = y * y;
            gradient = y2 * (mu + 2.0 * value) * mu4_inv;
            return 0.5 * y2 * y * (mu + value) * mu4_inv + 1.0;
        } else {
            gradient = 0.0;
            return 1.0;
        }
    }

    // Time transformation functions
    static double getExponentialC2(double time) {
        return time > 0.0 ? ((0.5 * time + 1.0) * time + 1.0)
                          : 1.0 / ((0.5 * time - 1.0) * time + 1.0);
    }

    static double getLogarithmicC2(double duration) {
        return duration > 1.0 ? (sqrt(2.0 * duration - 1.0) - 1.0)
                              : (1.0 - sqrt(2.0 / duration - 1.0));
    }

    static double getTimeGradient(double time) {
        if (time > 0.0) {
            return time + 1.0;
        } else {
            double denominator_sqrt = (0.5 * time - 1.0) * time + 1.0;
            return (1.0 - time) / (denominator_sqrt * denominator_sqrt);
        }
    }

    // Thrust transformation functions
    static double getForwardThrust(const double& thrust_angle) {
        // Access static variables from the optimizer instance
        return getStaticThrustHalf() * sin(thrust_angle) + getStaticThrustMiddle();
    }

    static void addThrustLayerGradient(const double& thrust_angle,
                                       const double& thrust_gradient,
                                       double& angle_gradient) {
        angle_gradient = getStaticThrustHalf() * cos(thrust_angle) * thrust_gradient;
    }

    static void getForwardTailVelocity(const Eigen::Ref<const Eigen::Vector2d>& tangent_velocity,
                                       Eigen::Ref<Eigen::Vector3d> tail_velocity) {
        tail_velocity = getStaticLandingVelocity() + tangent_velocity.x() * getStaticVelocityTangentX() +
                        tangent_velocity.y() * getStaticVelocityTangentY();
    }

    // Static variable accessors (needed for static functions)
    static double getStaticThrustMiddle() { return static_thrust_middle_; }
    static double getStaticThrustHalf() { return static_thrust_half_; }
    static Eigen::Vector3d getStaticLandingVelocity() { return static_landing_velocity_; }
    static Eigen::Vector3d getStaticVelocityTangentX() { return static_velocity_tangent_x_; }
    static Eigen::Vector3d getStaticVelocityTangentY() { return static_velocity_tangent_y_; }
    static Eigen::Vector3d getStaticCurrentPosition() { return static_current_position_; }
    static Eigen::Vector3d getStaticCurrentVelocity() { return static_current_velocity_; }
    static Eigen::Vector3d getStaticTailQuaternionVector() { return static_tail_quaternion_vector_; }
    static Eigen::Vector3d getStaticGravity() { return static_gravity_; }

    // Static variables for use in static functions
    static thread_local double static_thrust_middle_;
    static thread_local double static_thrust_half_;
    static thread_local Eigen::Vector3d static_landing_velocity_;
    static thread_local Eigen::Vector3d static_velocity_tangent_x_;
    static thread_local Eigen::Vector3d static_velocity_tangent_y_;
    static thread_local Eigen::Vector3d static_current_position_;
    static thread_local Eigen::Vector3d static_current_velocity_;
    static thread_local Eigen::Vector3d static_tail_quaternion_vector_;
    static thread_local Eigen::Vector3d static_gravity_;
    static thread_local int static_iteration_count_;

    // Update static variables before optimization
    void updateStaticVariables() {
        static_thrust_middle_ = thrust_middle_;
        static_thrust_half_ = thrust_half_;
        static_landing_velocity_ = landing_velocity_;
        static_velocity_tangent_x_ = velocity_tangent_x_;
        static_velocity_tangent_y_ = velocity_tangent_y_;
        static_current_position_ = current_position_;
        static_current_velocity_ = current_velocity_;
        static_tail_quaternion_vector_ = tail_quaternion_vector_;
        static_gravity_ = gravity_;
        static_iteration_count_ = 0;

        std::cout << "[TrajectoryOptimizer] updateStaticVariables:" << std::endl;
        std::cout << "  static_thrust_middle_: " << static_thrust_middle_ << std::endl;
        std::cout << "  static_thrust_half_: " << static_thrust_half_ << std::endl;
        std::cout << "  static_landing_velocity_: " << static_landing_velocity_.transpose() << std::endl;
        std::cout << "  static_tail_quaternion_vector_: " << static_tail_quaternion_vector_.transpose() << std::endl;
    }

    // Objective function and early exit callback
    static double getObjectiveFunction(void* optimizer_ptr,
                                       const double* variables,
                                       double* gradients,
                                       const int variable_count) {
        static_iteration_count_++;
        TrajectoryOptimizer& optimizer = *(TrajectoryOptimizer*)optimizer_ptr;

        const double& time_var = variables[0];
        double& time_gradient = gradients[0];
        Eigen::Map<const Eigen::MatrixXd> waypoints(variables + optimizer.time_dimension_, 3, optimizer.waypoint_dimension_);
        Eigen::Map<Eigen::MatrixXd> waypoint_gradients(gradients + optimizer.time_dimension_, 3, optimizer.waypoint_dimension_);
        const double& tail_thrust = variables[optimizer.time_dimension_ + optimizer.waypoint_dimension_ * 3];
        double& thrust_gradient = gradients[optimizer.time_dimension_ + optimizer.waypoint_dimension_ * 3];
        Eigen::Map<const Eigen::Vector2d> velocity_tangent(variables + optimizer.time_dimension_ + 3 * optimizer.waypoint_dimension_ + 1);
        Eigen::Map<Eigen::Vector2d> velocity_tangent_gradient(gradients + optimizer.time_dimension_ + 3 * optimizer.waypoint_dimension_ + 1);

        double delta_time = getExponentialC2(time_var);
        Eigen::Vector3d tail_velocity, tail_velocity_gradient;
        getForwardTailVelocity(velocity_tangent, tail_velocity);

        Eigen::MatrixXd tail_state(3, 4);
        tail_state.col(0) = static_current_position_ + static_current_velocity_ * optimizer.num_pieces_ * delta_time +
                            static_tail_quaternion_vector_ * optimizer.robot_length_;
        tail_state.col(1) = tail_velocity;
        tail_state.col(2) = getForwardThrust(tail_thrust) * static_tail_quaternion_vector_ + static_gravity_;
        tail_state.col(3).setZero();

        auto start_time = std::chrono::steady_clock::now();
        optimizer.minco_optimizer_.generate(optimizer.initial_state_matrix_, tail_state, waypoints, delta_time);

        double cost = optimizer.minco_optimizer_.getTrajSnapCost();
        optimizer.minco_optimizer_.calGrads_CT();

        auto end_time = std::chrono::steady_clock::now();
        optimizer.timing_inner_loop_ += (end_time - start_time).count();

        start_time = std::chrono::steady_clock::now();
        optimizer.addTimeIntegrationPenalty(cost);
        end_time = std::chrono::steady_clock::now();
        optimizer.timing_integral_ += (end_time - start_time).count();

        start_time = std::chrono::steady_clock::now();
        optimizer.minco_optimizer_.calGrads_PT();
        end_time = std::chrono::steady_clock::now();
        optimizer.timing_inner_loop_ += (end_time - start_time).count();

        optimizer.minco_optimizer_.gdT += optimizer.minco_optimizer_.gdTail.col(0).dot(optimizer.num_pieces_ * static_current_velocity_);
        tail_velocity_gradient = optimizer.minco_optimizer_.gdTail.col(1);
        double thrust_force_gradient = optimizer.minco_optimizer_.gdTail.col(2).dot(static_tail_quaternion_vector_);
        addThrustLayerGradient(tail_thrust, thrust_force_gradient, thrust_gradient);

        if (optimizer.velocity_tail_weight_ > -1.0) {
            velocity_tangent_gradient.x() = tail_velocity_gradient.dot(static_velocity_tangent_x_);
            velocity_tangent_gradient.y() = tail_velocity_gradient.dot(static_velocity_tangent_y_);

            double velocity_tangent_penalty = velocity_tangent.squaredNorm();
            cost += optimizer.velocity_tail_weight_ * velocity_tangent_penalty;
            velocity_tangent_gradient += 2.0 * optimizer.velocity_tail_weight_ * velocity_tangent;
        }

        optimizer.minco_optimizer_.gdT += optimizer.time_weight_;
        cost += optimizer.time_weight_ * delta_time;
        time_gradient = optimizer.minco_optimizer_.gdT * getTimeGradient(time_var);

        waypoint_gradients = optimizer.minco_optimizer_.gdP;

        return cost;
    }

    static int getEarlyExitCondition(void* optimizer_ptr,
                                     const double* variables,
                                     const double* gradients,
                                     const double function_value,
                                     const double variable_norm,
                                     const double gradient_norm,
                                     const double step_size,
                                     int variable_count,
                                     int iteration,
                                     int line_search) {
        TrajectoryOptimizer& optimizer = *(TrajectoryOptimizer*)optimizer_ptr;
        if (optimizer.debug_mode_) {
            if (iteration % 10 == 0) {
                std::cout << "Iteration: " << iteration << ", Cost: " << function_value
                          << ", Gradient norm: " << gradient_norm << std::endl;
            }
        }
        return 0;  // Continue optimization
    }

    // Boundary value problem solver
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

        coefficient_matrix.col(0) = (boundary_conditions.col(7) / 6.0 + boundary_conditions.col(3) / 6.0) * t3 +
                                    (-2.0 * boundary_conditions.col(6) + 2.0 * boundary_conditions.col(2)) * t2 +
                                    (10.0 * boundary_conditions.col(5) + 10.0 * boundary_conditions.col(1)) * t1 +
                                    (-20.0 * boundary_conditions.col(4) + 20.0 * boundary_conditions.col(0));
        coefficient_matrix.col(1) = (-0.5 * boundary_conditions.col(7) - boundary_conditions.col(3) / 1.5) * t3 +
                                    (6.5 * boundary_conditions.col(6) - 7.5 * boundary_conditions.col(2)) * t2 +
                                    (-34.0 * boundary_conditions.col(5) - 36.0 * boundary_conditions.col(1)) * t1 +
                                    (70.0 * boundary_conditions.col(4) - 70.0 * boundary_conditions.col(0));
        coefficient_matrix.col(2) = (0.5 * boundary_conditions.col(7) + boundary_conditions.col(3)) * t3 +
                                    (-7.0 * boundary_conditions.col(6) + 10.0 * boundary_conditions.col(2)) * t2 +
                                    (39.0 * boundary_conditions.col(5) + 45.0 * boundary_conditions.col(1)) * t1 +
                                    (-84.0 * boundary_conditions.col(4) + 84.0 * boundary_conditions.col(0));
        coefficient_matrix.col(3) = (-boundary_conditions.col(7) / 6.0 - boundary_conditions.col(3) / 1.5) * t3 +
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

    // Maximum angular velocity calculation
    static double getMaximumAngularVelocity(Trajectory& trajectory) {
        double dt = 0.01;
        double max_angular_velocity = 0.0;
        for (double t = 0; t < trajectory.getTotalDuration(); t += dt) {
            Eigen::Vector3d acceleration = trajectory.getAcc(t);
            Eigen::Vector3d jerk = trajectory.getJer(t);
            Eigen::Vector3d thrust = acceleration - static_gravity_;
            Eigen::Vector3d angular_velocity_vector = getNormalizationDerivative(thrust) * jerk;
            double angular_velocity_magnitude = angular_velocity_vector.norm();
            if (angular_velocity_magnitude > max_angular_velocity) {
                max_angular_velocity = angular_velocity_magnitude;
            }
        }
        return max_angular_velocity;
    }

    // Time integration penalty
    void addTimeIntegrationPenalty(double& cost) {
        Eigen::Vector3d position, velocity, acceleration, jerk, snap;
        Eigen::Vector3d temp_gradient, temp_gradient2, temp_gradient3;
        Eigen::Vector3d position_gradient, velocity_gradient, acceleration_gradient, jerk_gradient;
        double temp_cost, inner_cost;
        Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
        double s1, s2, s3, s4, s5, s6, s7;
        double step, alpha, omega;
        Eigen::Matrix<double, 8, 3> gradient_violation_coeffs;
        double gradient_violation_time;

        int inner_loop_count = integration_steps_ + 1;
        step = minco_optimizer_.t(1) / integration_steps_;

        s1 = 0.0;

        for (int j = 0; j < inner_loop_count; ++j) {
            s2 = s1 * s1;
            s3 = s2 * s1;
            s4 = s2 * s2;
            s5 = s4 * s1;
            s6 = s4 * s2;
            s7 = s4 * s3;
            beta0 << 1.0, s1, s2, s3, s4, s5, s6, s7;
            beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6.0 * s5, 7.0 * s6;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3, 30.0 * s4, 42.0 * s5;
            beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2, 120.0 * s3, 210.0 * s4;
            beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1, 360.0 * s2, 840.0 * s3;
            alpha = 1.0 / integration_steps_ * j;
            omega = (j == 0 || j == inner_loop_count - 1) ? 0.5 : 1.0;

            for (int i = 0; i < num_pieces_; ++i) {
                const auto& c = minco_optimizer_.c.block<8, 3>(i * 8, 0);

                position = c.transpose() * beta0;
                velocity = c.transpose() * beta1;
                acceleration = c.transpose() * beta2;
                jerk = c.transpose() * beta3;
                snap = c.transpose() * beta4;

                position_gradient.setZero();
                velocity_gradient.setZero();
                acceleration_gradient.setZero();
                jerk_gradient.setZero();
                temp_gradient3.setZero();
                inner_cost = 0.0;

                // Apply constraints
                if (getFloorCostGradient(position, temp_gradient, temp_cost)) {
                    position_gradient += temp_gradient;
                    inner_cost += temp_cost;
                }

                if (getVelocityCostGradient(velocity, temp_gradient, temp_cost)) {
                    velocity_gradient += temp_gradient;
                    inner_cost += temp_cost;
                }

                if (getThrustCostGradient(acceleration, temp_gradient, temp_cost)) {
                    acceleration_gradient += temp_gradient;
                    inner_cost += temp_cost;
                }

                if (getAngularVelocityCostGradient(acceleration, jerk, temp_gradient, temp_gradient2, temp_cost)) {
                    acceleration_gradient += temp_gradient;
                    jerk_gradient += temp_gradient2;
                    inner_cost += temp_cost;
                }

                // if (getYawAngularVelocityCostGradient(acceleration, jerk, temp_gradient, temp_gradient2, temp_cost)) {
                //     acceleration_gradient += temp_gradient;
                //     jerk_gradient += temp_gradient2;
                //     inner_cost += temp_cost;
                // }

                double duration_to_now = (i + alpha) * minco_optimizer_.t(1);
                Eigen::Vector3d car_position = current_position_ + current_velocity_ * duration_to_now;
                if (getPerchingCollisionCostGradient(position, acceleration, car_position,
                                                     temp_gradient, temp_gradient2, temp_gradient3, temp_cost)) {
                    position_gradient += temp_gradient;
                    acceleration_gradient += temp_gradient2;
                    inner_cost += temp_cost;
                }
                double gradient_car_time = temp_gradient3.dot(current_velocity_);

                // DEBUG LOGGING START
                // if (static_iteration_count_ % 10 == 0 && j == integration_steps_ / 2) { // Log on specific iterations and at the midpoint of the piece
                //     double floor_c = 0, v_c = 0, thrust_c = 0, omega_c = 0, perch_c = 0;
                //     Eigen::Vector3d unused_grad, unused_grad2, unused_grad3;
                //     getFloorCostGradient(position, unused_grad, floor_c);
                //     getVelocityCostGradient(velocity, unused_grad, v_c);
                //     getThrustCostGradient(acceleration, unused_grad, thrust_c);
                //     getAngularVelocityCostGradient(acceleration, jerk, unused_grad, unused_grad2, omega_c);
                //     getPerchingCollisionCostGradient(position, acceleration, car_position, unused_grad, unused_grad2, unused_grad3, perch_c);
                //     printf("[REFACTORED] it:%d i:%d j:%d pos:%.2f,%.2f,%.2f costs(f,v,t,o,p): %.3f,%.3f,%.3f,%.3f,%.3f\n",
                //            static_iteration_count_, i, j, position.x(), position.y(), position.z(), floor_c, v_c, thrust_c, omega_c, perch_c);
                // }
                // DEBUG LOGGING END

                gradient_violation_coeffs = beta0 * position_gradient.transpose();
                gradient_violation_time = position_gradient.transpose() * velocity;
                gradient_violation_coeffs += beta1 * velocity_gradient.transpose();
                gradient_violation_time += velocity_gradient.transpose() * acceleration;
                gradient_violation_coeffs += beta2 * acceleration_gradient.transpose();
                gradient_violation_time += acceleration_gradient.transpose() * jerk;
                gradient_violation_coeffs += beta3 * jerk_gradient.transpose();
                gradient_violation_time += jerk_gradient.transpose() * snap;
                gradient_violation_time += gradient_car_time;

                minco_optimizer_.gdC.block<8, 3>(i * 8, 0) += omega * step * gradient_violation_coeffs;
                minco_optimizer_.gdT += omega * (inner_cost / integration_steps_ + alpha * step * gradient_violation_time);
                minco_optimizer_.gdT += i * omega * step * gradient_car_time;
                cost += omega * step * inner_cost;
            }
            s1 += step;
        }
    }

    // Cost and gradient functions
    bool getVelocityCostGradient(const Eigen::Vector3d& velocity,
                                 Eigen::Vector3d& velocity_gradient,
                                 double& velocity_cost) {
        double velocity_penalty = velocity.squaredNorm() - max_velocity_ * max_velocity_;
        if (velocity_penalty > 0) {
            double gradient_scalar = 0.0;
            velocity_cost = getSmoothedL1(velocity_penalty, gradient_scalar);
            velocity_gradient = velocity_weight_ * gradient_scalar * 2.0 * velocity;
            velocity_cost *= velocity_weight_;
            return true;
        }
        return false;
    }

    bool getThrustCostGradient(const Eigen::Vector3d& acceleration,
                               Eigen::Vector3d& acceleration_gradient,
                               double& acceleration_cost) {
        bool has_violation = false;
        acceleration_gradient.setZero();
        acceleration_cost = 0.0;

        Eigen::Vector3d thrust_force = acceleration - gravity_;
        double max_penalty = thrust_force.squaredNorm() - max_thrust_ * max_thrust_;
        if (max_penalty > 0) {
            double gradient_scalar = 0.0;
            acceleration_cost = thrust_weight_ * getSmoothedL1(max_penalty, gradient_scalar);
            acceleration_gradient = thrust_weight_ * 2.0 * gradient_scalar * thrust_force;
            has_violation = true;
        }

        double min_penalty = min_thrust_ * min_thrust_ - thrust_force.squaredNorm();
        if (min_penalty > 0) {
            double gradient_scalar = 0.0;
            // FIXME: this might be bug in the original code, but we keep it for now
            acceleration_cost = thrust_weight_ * getSmoothedL1(min_penalty, gradient_scalar);
            acceleration_gradient = -thrust_weight_ * 2.0 * gradient_scalar * thrust_force;
            has_violation = true;
        }

        return has_violation;
    }

    bool getAngularVelocityCostGradient(const Eigen::Vector3d& acceleration,
                                        const Eigen::Vector3d& jerk,
                                        Eigen::Vector3d& acceleration_gradient,
                                        Eigen::Vector3d& jerk_gradient,
                                        double& cost) {
        Eigen::Vector3d thrust_force = acceleration - gravity_;
        Eigen::Vector3d body_z_dot = getNormalizationDerivative(thrust_force) * jerk;
        double angular_velocity_12_squared = body_z_dot.squaredNorm();
        double penalty = angular_velocity_12_squared - max_angular_velocity_ * max_angular_velocity_;

        if (penalty > 0) {
            double gradient_scalar = 0.0;
            // FIXME: check potential mismatch with TrajOpt::grad_cost_omega here
            cost = angular_velocity_weight_ * getSmoothedL1(penalty, gradient_scalar);

            Eigen::Vector3d temp_gradient = angular_velocity_weight_ * gradient_scalar * 2.0 * body_z_dot;
            Eigen::MatrixXd derivative_matrix = getNormalizationDerivative(thrust_force);
            Eigen::MatrixXd second_derivative_matrix = getSecondNormalizationDerivative(thrust_force, jerk);

            jerk_gradient = derivative_matrix.transpose() * temp_gradient;
            acceleration_gradient = second_derivative_matrix.transpose() * temp_gradient;

            return true;
        }
        return false;
    }

    // bool getYawAngularVelocityCostGradient(const Eigen::Vector3d& acceleration,
    //                                       const Eigen::Vector3d& jerk,
    //                                       Eigen::Vector3d& acceleration_gradient,
    //                                       Eigen::Vector3d& jerk_gradient,
    //                                       double& cost) {
    //     // TODO: Implement yaw angular velocity constraint
    //     return false;
    // }

    bool getFloorCostGradient(const Eigen::Vector3d& position,
                              Eigen::Vector3d& position_gradient,
                              double& position_cost) {
        static double floor_height = 0.4;
        double penalty = floor_height - position.z();
        if (penalty > 0) {
            double gradient_scalar = 0.0;
            position_cost = position_weight_ * getSmoothedL1(penalty, gradient_scalar);
            position_gradient = Eigen::Vector3d(0, 0, -position_weight_ * gradient_scalar);
            return true;
        } else {
            position_gradient.setZero();
            position_cost = 0.0;
            return false;
        }
    }

    bool getPerchingCollisionCostGradient(const Eigen::Vector3d& pos,
                                          const Eigen::Vector3d& acc,
                                          const Eigen::Vector3d& car_p,
                                          Eigen::Vector3d& gradp,
                                          Eigen::Vector3d& grada,
                                          Eigen::Vector3d& grad_car_p,
                                          double& cost) {
        static double eps = 1e-6;

        double dist_sqr = (pos - car_p).squaredNorm();
        double safe_r = platform_radius_ + robot_radius_;
        double safe_r_sqr = safe_r * safe_r;
        double pen_dist = safe_r_sqr - dist_sqr;
        pen_dist /= safe_r_sqr;
        double grad_dist = 0;
        double var01 = getSmoothed01(pen_dist, grad_dist);
        if (var01 == 0) {
            return false;
        }
        Eigen::Vector3d gradp_dist = grad_dist * 2 * (car_p - pos);
        Eigen::Vector3d grad_carp_dist = -gradp_dist;

        Eigen::Vector3d a_i = -static_tail_quaternion_vector_;
        double b_i = a_i.dot(car_p);

        Eigen::Vector3d thrust_f = acc - static_gravity_;
        Eigen::Vector3d zb = normalizeVector(thrust_f);

        Eigen::MatrixXd BTRT(2, 3);
        double a = zb.x();
        double b = zb.y();
        double c = zb.z();

        double c_1 = 1.0 / (1 + c);

        BTRT(0, 0) = 1 - a * a * c_1;
        BTRT(0, 1) = -a * b * c_1;
        BTRT(0, 2) = -a;
        BTRT(1, 0) = -a * b * c_1;
        BTRT(1, 1) = 1 - b * b * c_1;
        BTRT(1, 2) = -b;

        Eigen::Vector2d v2 = BTRT * a_i;
        double v2_norm = sqrt(v2.squaredNorm() + eps);
        double pen = a_i.dot(pos) - (robot_length_ - 0.005) * a_i.dot(zb) - b_i + robot_radius_ * v2_norm;

        if (pen > 0) {
            double grad = 0;
            cost = getSmoothedL1(pen, grad);
            // gradients: pos, car_p, v2
            gradp = a_i;
            grad_car_p = -a_i;
            Eigen::Vector2d grad_v2 = robot_radius_ * v2 / v2_norm;

            Eigen::MatrixXd pM_pa(2, 3), pM_pb(2, 3), pM_pc(2, 3);
            double c2_1 = c_1 * c_1;

            pM_pa(0, 0) = -2 * a * c_1;
            pM_pa(0, 1) = -b * c_1;
            pM_pa(0, 2) = -1;
            pM_pa(1, 0) = -b * c_1;
            pM_pa(1, 1) = 0;
            pM_pa(1, 2) = 0;

            pM_pb(0, 0) = 0;
            pM_pb(0, 1) = -a * c_1;
            pM_pb(0, 2) = 0;
            pM_pb(1, 0) = -a * c_1;
            pM_pb(1, 1) = -2 * b * c_1;
            pM_pb(1, 2) = -1;

            pM_pc(0, 0) = a * a * c2_1;
            pM_pc(0, 1) = a * b * c2_1;
            pM_pc(0, 2) = 0;
            pM_pc(1, 0) = a * b * c2_1;
            pM_pc(1, 1) = b * b * c2_1;
            pM_pc(1, 2) = 0;

            Eigen::MatrixXd pv2_pzb(2, 3);
            pv2_pzb.col(0) = pM_pa * a_i;
            pv2_pzb.col(1) = pM_pb * a_i;
            pv2_pzb.col(2) = pM_pc * a_i;

            Eigen::Vector3d grad_zb = pv2_pzb.transpose() * grad_v2 - robot_length_ * a_i;

            grada = getNormalizationDerivative(thrust_f).transpose() * grad_zb;

            grad *= var01;
            gradp_dist *= cost;
            grad_carp_dist *= cost;
            cost *= var01;
            gradp = grad * gradp + gradp_dist;
            grada *= grad;
            grad_car_p = grad * grad_car_p + grad_carp_dist;

            cost *= perching_collision_weight_;
            gradp *= perching_collision_weight_;
            grada *= perching_collision_weight_;
            grad_car_p *= perching_collision_weight_;

            return true;
        }
        return false;
    }
};

// Static thread_local variable definitions
thread_local double TrajectoryOptimizer::static_thrust_middle_ = 0.0;
thread_local double TrajectoryOptimizer::static_thrust_half_ = 0.0;
thread_local Eigen::Vector3d TrajectoryOptimizer::static_landing_velocity_ = Eigen::Vector3d::Zero();
thread_local Eigen::Vector3d TrajectoryOptimizer::static_velocity_tangent_x_ = Eigen::Vector3d::Zero();
thread_local Eigen::Vector3d TrajectoryOptimizer::static_velocity_tangent_y_ = Eigen::Vector3d::Zero();
thread_local Eigen::Vector3d TrajectoryOptimizer::static_current_position_ = Eigen::Vector3d::Zero();
thread_local Eigen::Vector3d TrajectoryOptimizer::static_current_velocity_ = Eigen::Vector3d::Zero();
thread_local Eigen::Vector3d TrajectoryOptimizer::static_tail_quaternion_vector_ = Eigen::Vector3d::Zero();
thread_local Eigen::Vector3d TrajectoryOptimizer::static_gravity_ = Eigen::Vector3d(0, 0, -9.8);
thread_local int TrajectoryOptimizer::static_iteration_count_ = 0;

// Implementation note: This header extracts and refactors the trajectory optimization
// functionality from traj_opt_perching.cc, removing ROS dependencies and improving
// code clarity with better variable naming and consistent formatting.
//
// Key changes from original:
// - Removed ROS NodeHandle dependency
// - Renamed variables for clarity (e.g., car_p_ -> current_position_)
// - Used 4-space indentation consistently
// - Applied getSomething naming convention for getter functions
// - Used member_ format for class members
// - Extracted core optimization logic into clean, reusable class
// - Implemented all functions inline for header-only library usage

}  // namespace traj_opt
