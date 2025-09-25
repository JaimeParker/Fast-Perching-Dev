#include "traj_opt.h"
#include "trajectory_optimizer.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

int main() {
    std::cout << "Trajectory Optimization Example\n";
    
    // Example 1: Using the original TrajOpt class (ROS-free version)
    {
        std::cout << "\n=== Testing TrajOpt class ===\n";
        traj_opt::TrajOpt optimizer;
        
        // Configure parameters
        optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        optimizer.setRobotParameters(1.0, 0.3, 0.1, 0.5);
        optimizer.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        optimizer.setIntegrationSteps(20);
        optimizer.setDebugMode(false);
        
        // Set up initial state (position, velocity, acceleration, jerk)
        Eigen::MatrixXd initial_state(3, 4);
        initial_state.setZero();
        initial_state.col(0) = Eigen::Vector3d(0.0, 0.0, 2.0);  // Initial position
        initial_state.col(1) = Eigen::Vector3d(5.0, 0.0, 0.0);  // Initial velocity
        
        // Set up target
        Eigen::Vector3d target_position(10.0, 0.0, 1.0);
        Eigen::Vector3d target_velocity(2.0, 0.0, 0.0);
        Eigen::Quaterniond landing_quaternion(1.0, 0.0, 0.0, 0.0);  // Identity quaternion
        
        std::cout << "Initial state configured:\n";
        std::cout << "  Position: " << initial_state.col(0).transpose() << "\n";
        std::cout << "  Velocity: " << initial_state.col(1).transpose() << "\n";
        std::cout << "Target position: " << target_position.transpose() << "\n";
        std::cout << "Target velocity: " << target_velocity.transpose() << "\n";
        
        // Note: Actual trajectory generation would require proper MINCO implementation
        // This is just a demonstration of the API
        std::cout << "TrajOpt class configured successfully!\n";
    }
    
    // Example 2: Using the new TrajectoryOptimizer class
    {
        std::cout << "\n=== Testing TrajectoryOptimizer class ===\n";
        traj_opt::TrajectoryOptimizer optimizer;
        
        // Configure parameters
        optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        optimizer.setRobotParameters(1.0, 0.3, 0.1, 0.5);
        optimizer.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        optimizer.setIntegrationParameters(20);
        optimizer.setDebugMode(true);
        
        // Set up initial state
        Eigen::MatrixXd initial_state(3, 4);
        initial_state.setZero();
        initial_state.col(0) = Eigen::Vector3d(0.0, 0.0, 2.0);  // Initial position
        initial_state.col(1) = Eigen::Vector3d(5.0, 0.0, 0.0);  // Initial velocity
        
        // Set up target
        Eigen::Vector3d target_position(10.0, 0.0, 1.0);
        Eigen::Vector3d target_velocity(2.0, 0.0, 0.0);
        Eigen::Quaterniond landing_quaternion(1.0, 0.0, 0.0, 0.0);
        
        std::cout << "TrajectoryOptimizer configured successfully!\n";
        std::cout << "This demonstrates the ROS-free trajectory optimization API.\n";
        
        // Note: Actual trajectory generation would require proper implementation
        // Trajectory result_trajectory;
        // bool success = optimizer.generateTrajectory(initial_state, target_position, 
        //                                           target_velocity, landing_quaternion, 
        //                                           10, result_trajectory);
    }
    
    std::cout << "\nExample completed successfully!\n";
    std::cout << "Note: For full functionality, ensure MINCO and L-BFGS libraries are properly linked.\n";
    
    return 0;
}