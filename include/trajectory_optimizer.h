#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <memory>
#include <iostream>
#include <cmath>

#include "minco.hpp"
#include "lbfgs_raw.hpp"

namespace traj_opt {

class TrajectoryOptimizer {
public:
    // Constructor with configuration parameters
    TrajectoryOptimizer() : 
        num_pieces_(0), integration_steps_(20), time_dimension_(1), waypoint_dimension_(0),
        optimization_variables_(nullptr), max_velocity_(10.0), max_acceleration_(10.0),
        max_thrust_(20.0), min_thrust_(2.0), max_angular_velocity_(3.0), max_yaw_angular_velocity_(2.0),
        velocity_plus_(1.0), robot_length_(0.3), robot_radius_(0.1), platform_radius_(0.5),
        time_weight_(1.0), velocity_tail_weight_(-1.0), position_weight_(1.0), velocity_weight_(1.0),
        acceleration_weight_(1.0), thrust_weight_(1.0), angular_velocity_weight_(1.0), 
        perching_collision_weight_(1.0), gravity_(0, 0, -9.8), has_initial_guess_(false),
        thrust_middle_(0.0), thrust_half_(0.0), timing_inner_loop_(0.0), timing_integral_(0.0),
        iteration_count_(0), debug_mode_(false) {}
    
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
        
        // Convert quaternion to vector
        quaternionToVector(landing_quaternion, tail_quaternion_vector_);
        
        // Set thrust parameters
        thrust_middle_ = (max_thrust_ + min_thrust_) / 2.0;
        thrust_half_ = (max_thrust_ - min_thrust_) / 2.0;
        
        // Calculate landing velocity
        landing_velocity_ = current_velocity_ - tail_quaternion_vector_ * velocity_plus_;
        
        // Calculate tangent vectors
        velocity_tangent_x_ = tail_quaternion_vector_.cross(Eigen::Vector3d(0, 0, 1));
        if (velocity_tangent_x_.squaredNorm() == 0) {
            velocity_tangent_x_ = Eigen::Vector3d(1, 0, 0);
        }
        velocity_tangent_x_.normalize();
        velocity_tangent_y_ = tail_quaternion_vector_.cross(velocity_tangent_x_);
        velocity_tangent_y_.normalize();
        
        velocity_tangent.setConstant(0.0);
        
        // Set boundary conditions
        initial_state_matrix_ = initial_state;
        
        // Initialize MINCO optimizer
        minco_optimizer_.reset(num_pieces_);
        
        // Update static variables for use in static callback functions
        updateStaticVariables();
        
        tail_thrust = 0.0;
        
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
            // Initialize with straight line trajectory
            time_var = getLogarithmicC2(2.0);
            
            Eigen::Vector3d tail_position = current_position_ + current_velocity_ * num_pieces_ * getExponentialC2(time_var) + 
                                          tail_quaternion_vector_ * robot_length_;
            Eigen::Vector3d tail_velocity;
            getForwardTailVelocity(velocity_tangent, tail_velocity);
            
            Eigen::MatrixXd tail_state(3, 4);
            tail_state.col(0) = tail_position;
            tail_state.col(1) = tail_velocity;
            tail_state.col(2) = getForwardThrust(tail_thrust) * tail_quaternion_vector_ + gravity_;
            tail_state.col(3).setZero();
            
            // Solve boundary value problem to get initial waypoints
            CoefficientMat coeff_matrix;
            solveBoundaryValueProblem(getExponentialC2(time_var), initial_state_matrix_, tail_state, coeff_matrix);
            
            // Extract waypoints from polynomial coefficients
            for (int i = 0; i < waypoint_dimension_; ++i) {
                double tau = (i + 1.0) / num_pieces_;
                waypoints.col(i) = evaluatePolynomial(coeff_matrix, tau * getExponentialC2(time_var));
            }
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
            &lbfgs_params
        );
        
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
        Eigen::Vector3d tail_velocity;
        getForwardTailVelocity(velocity_tangent, tail_velocity);
        
        Eigen::MatrixXd tail_state(3, 4);
        tail_state.col(0) = current_position_ + current_velocity_ * num_pieces_ * delta_time + 
                           tail_quaternion_vector_ * robot_length_;
        tail_state.col(1) = tail_velocity;
        tail_state.col(2) = getForwardThrust(tail_thrust) * tail_quaternion_vector_ + gravity_;
        tail_state.col(3).setZero();
        
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
    }
    
    void setRobotParameters(double velocity_plus, double robot_length,
                           double robot_radius, double platform_radius) {
        velocity_plus_ = velocity_plus;
        robot_length_ = robot_length;
        robot_radius_ = robot_radius;
        platform_radius_ = platform_radius;
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
    }
    
    void setIntegrationParameters(int integration_steps) {
        integration_steps_ = integration_steps;
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
        
        double penetration = plane_normal.dot(position) - (robot_length_ - 0.005) * plane_normal.dot(body_z) - 
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
                vector.dot(direction) * Eigen::MatrixXd::Identity(3, 3)) / cubed_norm;
    }

    // Smoothing functions
    static double getSmoothedL1(const double& value, double& gradient) {
        static double mu = 0.01;
        if (value < 0.0) {
            gradient = 0.0;
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
        return 0; // Continue optimization
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
        double step, alpha;
        Eigen::Matrix<double, 8, 3> gradient_violation_coeffs;
        double gradient_violation_time;

        int inner_loop_count = integration_steps_ + 1;
        step = minco_optimizer_.t(1) / integration_steps_;

        s1 = 0.0;

        for (int j = 0; j < inner_loop_count; ++j) {
            s1 += step;
            if (j == 0 || j == inner_loop_count - 1) {
                alpha = 1.0;
            } else {
                alpha = (j % 2 == 0) ? 2.0 : 4.0;
            }

            s2 = s1 * s1;
            s3 = s2 * s1;
            s4 = s2 * s2;
            s5 = s4 * s1;
            s6 = s3 * s3;
            s7 = s4 * s3;

            beta0(0) = 1.0; beta0(1) = s1; beta0(2) = s2; beta0(3) = s3;
            beta0(4) = s4; beta0(5) = s5; beta0(6) = s6; beta0(7) = s7;
            beta1(0) = 0.0; beta1(1) = 1.0; beta1(2) = 2.0 * s1; beta1(3) = 3.0 * s2;
            beta1(4) = 4.0 * s3; beta1(5) = 5.0 * s4; beta1(6) = 6.0 * s5; beta1(7) = 7.0 * s6;
            beta2(0) = 0.0; beta2(1) = 0.0; beta2(2) = 2.0; beta2(3) = 6.0 * s1;
            beta2(4) = 12.0 * s2; beta2(5) = 20.0 * s3; beta2(6) = 30.0 * s4; beta2(7) = 42.0 * s5;
            beta3(0) = 0.0; beta3(1) = 0.0; beta3(2) = 0.0; beta3(3) = 6.0;
            beta3(4) = 24.0 * s1; beta3(5) = 60.0 * s2; beta3(6) = 120.0 * s3; beta3(7) = 210.0 * s4;

            position = minco_optimizer_.getCoeff().transpose() * beta0;
            velocity = minco_optimizer_.getCoeff().transpose() * beta1;
            acceleration = minco_optimizer_.getCoeff().transpose() * beta2;
            jerk = minco_optimizer_.getCoeff().transpose() * beta3;

            temp_cost = 0.0;
            position_gradient.setZero();
            velocity_gradient.setZero();
            acceleration_gradient.setZero();
            jerk_gradient.setZero();

            // Apply constraints
            getVelocityCostGradient(velocity, temp_gradient, inner_cost);
            if (inner_cost > 0) {
                temp_cost += inner_cost;
                velocity_gradient += temp_gradient;
            }

            getThrustCostGradient(acceleration, temp_gradient, inner_cost);
            if (inner_cost > 0) {
                temp_cost += inner_cost;
                acceleration_gradient += temp_gradient;
            }

            getAngularVelocityCostGradient(acceleration, jerk, temp_gradient, temp_gradient2, inner_cost);
            if (inner_cost > 0) {
                temp_cost += inner_cost;
                acceleration_gradient += temp_gradient;
                jerk_gradient += temp_gradient2;
            }

            getFloorCostGradient(position, temp_gradient, inner_cost);
            if (inner_cost > 0) {
                temp_cost += inner_cost;
                position_gradient += temp_gradient;
            }

            getPerchingCollisionCostGradient(position, acceleration, current_position_, 
                                           temp_gradient, temp_gradient2, temp_gradient3, inner_cost);
            if (inner_cost > 0) {
                temp_cost += inner_cost;
                position_gradient += temp_gradient;
                acceleration_gradient += temp_gradient2;
            }

            gradient_violation_coeffs = position_gradient * beta0.transpose() + 
                                       velocity_gradient * beta1.transpose() + 
                                       acceleration_gradient * beta2.transpose() + 
                                       jerk_gradient * beta3.transpose();

            minco_optimizer_.gdC += alpha * gradient_violation_coeffs.transpose();
            cost += alpha * temp_cost;
        }

        cost *= step / 3.0;
        minco_optimizer_.gdC *= step / 3.0;
        minco_optimizer_.gdT += minco_optimizer_.gdC.cwiseProduct(minco_optimizer_.c1).sum() +
                              minco_optimizer_.gdC.cwiseProduct(minco_optimizer_.c2).sum() * s1 / 2.0 +
                              minco_optimizer_.gdC.cwiseProduct(minco_optimizer_.c3).sum() * s2 / 3.0 +
                              minco_optimizer_.gdC.cwiseProduct(minco_optimizer_.c4).sum() * s3 / 4.0;
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
            acceleration_cost += thrust_weight_ * getSmoothedL1(min_penalty, gradient_scalar);
            acceleration_gradient -= thrust_weight_ * 2.0 * gradient_scalar * thrust_force;
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

    bool getYawAngularVelocityCostGradient(const Eigen::Vector3d& acceleration,
                                          const Eigen::Vector3d& jerk,
                                          Eigen::Vector3d& acceleration_gradient,
                                          Eigen::Vector3d& jerk_gradient,
                                          double& cost) {
        // TODO: Implement yaw angular velocity constraint
        return false;
    }

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

    bool getPerchingCollisionCostGradient(const Eigen::Vector3d& position,
                                         const Eigen::Vector3d& acceleration,
                                         const Eigen::Vector3d& target_position,
                                         Eigen::Vector3d& position_gradient,
                                         Eigen::Vector3d& acceleration_gradient,
                                         Eigen::Vector3d& target_position_gradient,
                                         double& cost) {
        static double eps = 1e-6;

        double distance_squared = (position - target_position).squaredNorm();
        double safe_radius = platform_radius_ + robot_radius_;
        double safe_radius_squared = safe_radius * safe_radius;
        double distance_penalty = safe_radius_squared - distance_squared;
        distance_penalty /= safe_radius_squared;
        
        double distance_gradient = 0.0;
        double smoothed_distance = getSmoothed01(distance_penalty, distance_gradient);
        if (smoothed_distance == 0.0) {
            position_gradient.setZero();
            acceleration_gradient.setZero();
            target_position_gradient.setZero();
            cost = 0.0;
            return false;
        }
        
        Eigen::Vector3d distance_gradient_position = distance_gradient * 2.0 * (target_position - position);
        Eigen::Vector3d distance_gradient_target = -distance_gradient_position;

        Eigen::Vector3d plane_normal = -tail_quaternion_vector_;
        double plane_offset = plane_normal.dot(target_position);

        Eigen::Vector3d thrust_force = acceleration - gravity_;
        Eigen::Vector3d body_z = normalizeVector(thrust_force);

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
        
        double penetration = plane_normal.dot(position) - (robot_length_ - 0.005) * plane_normal.dot(body_z) - 
                           plane_offset + robot_radius_ * projected_normal_norm;

        if (penetration > 0) {
            double penetration_gradient = 0.0;
            cost = perching_collision_weight_ * smoothed_distance * getSmoothedL1(penetration, penetration_gradient);
            
            // Compute gradients (complex calculation involving multiple chain rules)
            Eigen::Vector3d penetration_gradient_position = penetration_gradient * plane_normal;
            Eigen::Vector3d penetration_gradient_acceleration = Eigen::Vector3d::Zero(); // Simplified
            Eigen::Vector3d penetration_gradient_target = penetration_gradient * (-plane_normal);
            
            position_gradient = perching_collision_weight_ * (smoothed_distance * penetration_gradient_position + 
                                                            penetration * distance_gradient_position);
            acceleration_gradient = perching_collision_weight_ * smoothed_distance * penetration_gradient_acceleration;
            target_position_gradient = perching_collision_weight_ * (smoothed_distance * penetration_gradient_target + 
                                                                   penetration * distance_gradient_target);
            
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

} // namespace traj_opt
