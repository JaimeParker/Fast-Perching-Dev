#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "perching_optimizer.h"
#include "traj_opt.h"

// Helper function to compare trajectories
bool compareTrajectories(const Trajectory& traj1, const Trajectory& traj2, double tolerance = 1e-3) {
    if (traj1.getPieceNum() != traj2.getPieceNum()) {
        std::cout << "Different number of trajectory pieces: " << traj1.getPieceNum() << " vs " << traj2.getPieceNum() << std::endl;
        return false;
    }

    bool trajectories_match = true;
    double max_position_diff = 0.0;
    double max_velocity_diff = 0.0;

    // Compare trajectories at multiple sample points
    for (int i = 0; i < traj1.getPieceNum(); ++i) {
        double piece_duration = traj1[i].getDuration();
        int num_samples = 10;

        for (int j = 0; j <= num_samples; ++j) {
            double t = j * piece_duration / num_samples;

            Eigen::Vector3d pos1 = traj1[i].getPos(t);
            Eigen::Vector3d vel1 = traj1[i].getVel(t);
            Eigen::Vector3d pos2 = traj2[i].getPos(t);
            Eigen::Vector3d vel2 = traj2[i].getVel(t);

            double pos_diff = (pos1 - pos2).norm();
            double vel_diff = (vel1 - vel2).norm();

            max_position_diff = std::max(max_position_diff, pos_diff);
            max_velocity_diff = std::max(max_velocity_diff, vel_diff);

            if (pos_diff > tolerance || vel_diff > tolerance) {
                trajectories_match = false;
            }
        }
    }

    std::cout << "Max position difference: " << std::fixed << std::setprecision(6) << max_position_diff << std::endl;
    std::cout << "Max velocity difference: " << std::fixed << std::setprecision(6) << max_velocity_diff << std::endl;

    return trajectories_match;
}

// Helper function to print trajectory statistics
void printTrajectoryStats(const std::string& name, const Trajectory& traj) {
    if (traj.getPieceNum() == 0) {
        std::cout << name << ": Empty trajectory" << std::endl;
        return;
    }

    std::cout << name << " statistics:" << std::endl;
    std::cout << "  Pieces: " << traj.getPieceNum() << std::endl;
    std::cout << "  Total duration: " << std::fixed << std::setprecision(3) << traj.getTotalDuration() << "s" << std::endl;

    // Sample trajectory at start, middle, and end
    Eigen::Vector3d start_pos = traj.getPos(0.0);
    Eigen::Vector3d end_pos = traj.getPos(traj.getTotalDuration());
    Eigen::Vector3d start_vel = traj.getVel(0.0);
    Eigen::Vector3d end_vel = traj.getVel(traj.getTotalDuration());

    std::cout << "  Start position: [" << std::fixed << std::setprecision(3)
              << start_pos.x() << ", " << start_pos.y() << ", " << start_pos.z() << "]" << std::endl;
    std::cout << "  End position: [" << std::fixed << std::setprecision(3)
              << end_pos.x() << ", " << end_pos.y() << ", " << end_pos.z() << "]" << std::endl;
    std::cout << "  Start velocity: [" << std::fixed << std::setprecision(3)
              << start_vel.x() << ", " << start_vel.y() << ", " << start_vel.z() << "]" << std::endl;
    std::cout << "  End velocity: [" << std::fixed << std::setprecision(3)
              << end_vel.x() << ", " << end_vel.y() << ", " << end_vel.z() << "]" << std::endl;

    // Print max thrust if available
    try {
        double max_thrust = traj.getMaxThrust();
        std::cout << "  Max thrust: " << std::fixed << std::setprecision(3) << max_thrust << std::endl;
    } catch (...) {
        // getMaxThrust might not be available in all trajectory implementations
    }
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "        Perching Trajectory Optimization Comparison Test\n";
    std::cout << "=================================================================\n";

    // Common problem setup - perching scenario
    Eigen::MatrixXd initial_state(3, 4);
    initial_state.setZero();
    initial_state.col(0) = Eigen::Vector3d(0.0, 0.0, 3.0);   // Initial position (hovering above target)
    initial_state.col(1) = Eigen::Vector3d(2.0, 0.0, -0.5);  // Initial velocity (moving toward target)
    initial_state.col(2) = Eigen::Vector3d(0.0, 0.0, 0.0);   // Initial acceleration
    initial_state.col(3) = Eigen::Vector3d(0.0, 0.0, 0.0);   // Initial jerk

    // Moving target (car/platform)
    Eigen::Vector3d target_position(8.0, 2.0, 1.0);
    Eigen::Vector3d target_velocity(1.5, 0.5, 0.0);

    // Landing orientation (slightly tilted for perching)
    Eigen::Quaterniond landing_quaternion(0.9659, 0.0, 0.2588, 0.0);  // ~30 degree tilt around Y-axis

    int num_pieces = 6;

    std::cout << "\nPerching problem setup:\n";
    std::cout << "  Initial position: [" << initial_state.col(0).transpose() << "]\n";
    std::cout << "  Initial velocity: [" << initial_state.col(1).transpose() << "]\n";
    std::cout << "  Target position:  [" << target_position.transpose() << "]\n";
    std::cout << "  Target velocity:  [" << target_velocity.transpose() << "]\n";
    std::cout << "  Landing quaternion: [" << landing_quaternion.w() << ", "
              << landing_quaternion.x() << ", " << landing_quaternion.y() << ", " << landing_quaternion.z() << "]\n";
    std::cout << "  Number of pieces: " << num_pieces << "\n";

    // Test results storage
    Trajectory traj_opt_result;
    Trajectory perching_optimizer_result;
    bool traj_opt_success = false;
    bool perching_optimizer_success = false;
    double traj_opt_time = 0.0;
    double perching_optimizer_time = 0.0;

    // Test 1: Original TrajOpt class (traj_opt_perching.cc)
    {
        std::cout << "\n=== Testing Original TrajOpt Class (traj_opt_perching.cc) ===\n";
        traj_opt::TrajOpt optimizer;

        // Configure perching-specific parameters
        optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        optimizer.setRobotParameters(1.0, 0.3, 0.1, 0.5);
        optimizer.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        optimizer.setIntegrationSteps(20);
        optimizer.setDebugMode(false);

        std::cout << "Configuration complete. Starting optimization...\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        traj_opt_success = optimizer.generate_traj(
            initial_state,
            target_position,
            target_velocity,
            landing_quaternion,
            num_pieces,
            traj_opt_result);

        auto end_time = std::chrono::high_resolution_clock::now();
        traj_opt_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        if (traj_opt_success) {
            std::cout << "✓ Original TrajOpt optimization successful!\n";
            std::cout << "  Optimization time: " << std::fixed << std::setprecision(2) << traj_opt_time << " ms\n";
            printTrajectoryStats("Original TrajOpt", traj_opt_result);
        } else {
            std::cout << "✗ Original TrajOpt optimization failed!\n";
        }
    }

    // Test 2: New PerchingOptimizer class (perching_optimizer.h)
    {
        std::cout << "\n=== Testing New PerchingOptimizer Class (perching_optimizer.h) ===\n";
        traj_opt::PerchingOptimizer optimizer;

        // Configure identical parameters
        optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        optimizer.setRobotParameters(1.0, 0.3, 0.1, 0.5);
        optimizer.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        optimizer.setIntegrationSteps(20);
        optimizer.setDebugMode(false);

        std::cout << "Configuration complete. Starting optimization...\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        perching_optimizer_success = optimizer.generateTrajectory(
            initial_state,
            target_position,
            target_velocity,
            landing_quaternion,
            num_pieces,
            perching_optimizer_result);

        auto end_time = std::chrono::high_resolution_clock::now();
        perching_optimizer_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        if (perching_optimizer_success) {
            std::cout << "✓ PerchingOptimizer optimization successful!\n";
            std::cout << "  Optimization time: " << std::fixed << std::setprecision(2) << perching_optimizer_time << " ms\n";
            printTrajectoryStats("PerchingOptimizer", perching_optimizer_result);
        } else {
            std::cout << "✗ PerchingOptimizer optimization failed!\n";
        }
    }

    // Save trajectories to files for visualization
    if (traj_opt_success) {
        std::ofstream file1("../assets/original_trajectory.csv");
        file1 << "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z\n";

        double dt = 0.01;  // 10ms sampling
        for (double t = 0.0; t <= traj_opt_result.getTotalDuration(); t += dt) {
            Eigen::Vector3d pos = traj_opt_result.getPos(t);
            Eigen::Vector3d vel = traj_opt_result.getVel(t);
            Eigen::Vector3d acc = traj_opt_result.getAcc(t);

            file1 << t << "," << pos.x() << "," << pos.y() << "," << pos.z() << ","
                  << vel.x() << "," << vel.y() << "," << vel.z() << ","
                  << acc.x() << "," << acc.y() << "," << acc.z() << "\n";
        }
        file1.close();
        std::cout << "Original trajectory saved to ../assets/original_trajectory.csv\n";
    }

    if (perching_optimizer_success) {
        std::ofstream file2("../assets/perching_optimizer_trajectory.csv");
        file2 << "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z\n";

        double dt = 0.01;  // 10ms sampling
        for (double t = 0.0; t <= perching_optimizer_result.getTotalDuration(); t += dt) {
            Eigen::Vector3d pos = perching_optimizer_result.getPos(t);
            Eigen::Vector3d vel = perching_optimizer_result.getVel(t);
            Eigen::Vector3d acc = perching_optimizer_result.getAcc(t);

            file2 << t << "," << pos.x() << "," << pos.y() << "," << pos.z() << ","
                  << vel.x() << "," << vel.y() << "," << vel.z() << ","
                  << acc.x() << "," << acc.y() << "," << acc.z() << "\n";
        }
        file2.close();
        std::cout << "PerchingOptimizer trajectory saved to ../assets/perching_optimizer_trajectory.csv\n";
    }

    // Compare results
    std::cout << "\n=== Comparison Results ===\n";

    if (traj_opt_success && perching_optimizer_success) {
        std::cout << "Both optimizers succeeded. Comparing trajectories...\n";

        bool trajectories_similar = compareTrajectories(traj_opt_result, perching_optimizer_result, 1e-2);

        if (trajectories_similar) {
            std::cout << "✓ Trajectories are very similar! Both optimizers produced consistent results.\n";
            std::cout << "  This confirms the refactoring preserved the mathematical logic.\n";
        } else {
            std::cout << "\033[1;33m⚠ Trajectories differ significantly. This may indicate:\033[0m\n";
            std::cout << "    - Implementation differences in the refactoring\n";
            std::cout << "    - Different convergence criteria or numerical precision\n";
            std::cout << "    - Parameter mapping issues between the two classes\n";
        }

        // Performance comparison
        double time_ratio = perching_optimizer_time / traj_opt_time;
        std::cout << "\nPerformance comparison:\n";
        std::cout << "  Original TrajOpt time:    " << std::fixed << std::setprecision(2) << traj_opt_time << " ms\n";
        std::cout << "  PerchingOptimizer time:   " << std::fixed << std::setprecision(2) << perching_optimizer_time << " ms\n";
        std::cout << "  Speed ratio (New/Old):    " << std::fixed << std::setprecision(2) << time_ratio << "x\n";

        if (time_ratio < 0.9) {
            std::cout << "  → PerchingOptimizer is faster\n";
        } else if (time_ratio > 1.1) {
            std::cout << "  → Original TrajOpt is faster\n";
        } else {
            std::cout << "  → Performance is comparable\n";
        }

    } else if (traj_opt_success && !perching_optimizer_success) {
        std::cout << "\033[1;31m⚠ Only Original TrajOpt succeeded.\033[0m\n";
        std::cout << "  The PerchingOptimizer refactoring may have introduced bugs.\n";
        std::cout << "  Check the implementation in perching_optimizer.h\n";
    } else if (!traj_opt_success && perching_optimizer_success) {
        std::cout << "\033[1;31m⚠ Only PerchingOptimizer succeeded.\033[0m\n";
        std::cout << "  The original TrajOpt may have issues with this problem setup.\n";
    } else {
        std::cout << "\033[1;31m✗ Both optimizers failed.\033[0m\n";
        std::cout << "  Check problem setup, dependencies, and parameter values.\n";
    }

    // Feasibility checks if trajectories exist
    if (traj_opt_success) {
        std::cout << "\nRunning feasibility check on Original TrajOpt result...\n";
        traj_opt::TrajOpt feasibility_checker;
        feasibility_checker.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        bool is_feasible = feasibility_checker.feasibleCheck(traj_opt_result);
        std::cout << "  Original TrajOpt result feasible: " << (is_feasible ? "✓ Yes" : "✗ No") << std::endl;
    }

    if (perching_optimizer_success) {
        std::cout << "\nRunning feasibility check on PerchingOptimizer result...\n";
        traj_opt::PerchingOptimizer feasibility_checker;
        feasibility_checker.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        bool is_feasible = feasibility_checker.feasibleCheck(perching_optimizer_result);
        std::cout << "  PerchingOptimizer result feasible: " << (is_feasible ? "✓ Yes" : "✗ No") << std::endl;
    }

    // Additional validation: Check trajectory properties specific to perching
    std::cout << "\n=== Perching-Specific Validation ===\n";

    if (traj_opt_success) {
        std::cout << "Original TrajOpt trajectory analysis:\n";

        // Check if trajectory reaches near the target
        Eigen::Vector3d final_pos = traj_opt_result.getPos(traj_opt_result.getTotalDuration());
        Eigen::Vector3d expected_final_pos = target_position + target_velocity * traj_opt_result.getTotalDuration();
        double position_error = (final_pos - expected_final_pos).norm();

        std::cout << "  Expected final position: [" << expected_final_pos.transpose() << "]\n";
        std::cout << "  Actual final position:   [" << final_pos.transpose() << "]\n";
        std::cout << "  Position tracking error:  " << std::fixed << std::setprecision(4) << position_error << " m\n";

        if (position_error < 0.5) {
            std::cout << "  ✓ Good tracking performance\n";
        } else {
            std::cout << "  ⚠ Large tracking error\n";
        }
    }

    if (perching_optimizer_success) {
        std::cout << "\nPerchingOptimizer trajectory analysis:\n";

        // Check if trajectory reaches near the target
        Eigen::Vector3d final_pos = perching_optimizer_result.getPos(perching_optimizer_result.getTotalDuration());
        Eigen::Vector3d expected_final_pos = target_position + target_velocity * perching_optimizer_result.getTotalDuration();
        double position_error = (final_pos - expected_final_pos).norm();

        std::cout << "  Expected final position: [" << expected_final_pos.transpose() << "]\n";
        std::cout << "  Actual final position:   [" << final_pos.transpose() << "]\n";
        std::cout << "  Position tracking error:  " << std::fixed << std::setprecision(4) << position_error << " m\n";

        if (position_error < 0.5) {
            std::cout << "  ✓ Good tracking performance\n";
        } else {
            std::cout << "  ⚠ Large tracking error\n";
        }
    }

    std::cout << "\n=== Test Summary ===\n";
    if (traj_opt_success && perching_optimizer_success) {
        bool trajectories_similar = compareTrajectories(traj_opt_result, perching_optimizer_result, 1e-2);
        if (trajectories_similar) {
            std::cout << "🎉 \033[1;32mREFACTORING VALIDATION SUCCESSFUL!\033[0m\n";
            std::cout << "   The PerchingOptimizer class correctly implements the same\n";
            std::cout << "   mathematical algorithms as the original traj_opt_perching.cc\n";
        } else {
            std::cout << "⚠️  \033[1;33mREFACTORING NEEDS REVIEW\033[0m\n";
            std::cout << "   The trajectories differ, indicating potential issues in the refactoring.\n";
        }
    } else {
        std::cout << "❌ \033[1;31mTEST INCOMPLETE\033[0m\n";
        std::cout << "   One or both optimizers failed. Cannot validate refactoring.\n";
    }

    return 0;
}