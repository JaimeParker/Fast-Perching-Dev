#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "perching_optimizer.h"
#include "traj_opt.h"
#include "trajectory_optimizer.h"

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
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "        Trajectory Optimization Comparison Test\n";
    std::cout << "=================================================================\n";

    // Common problem setup
    Eigen::MatrixXd initial_state(3, 4);
    initial_state.setZero();
    initial_state.col(0) = Eigen::Vector3d(0.0, 0.0, 2.0);  // Initial position
    initial_state.col(1) = Eigen::Vector3d(5.0, 0.0, 0.0);  // Initial velocity
    initial_state.col(2) = Eigen::Vector3d(0.0, 0.0, 0.0);  // Initial acceleration
    initial_state.col(3) = Eigen::Vector3d(0.0, 0.0, 0.0);  // Initial jerk

    Eigen::Vector3d target_position(10.0, 0.0, 1.0);
    Eigen::Vector3d target_velocity(2.0, 0.0, 0.0);
    Eigen::Quaterniond landing_quaternion(1.0, 0.0, 0.0, 0.0);  // Identity quaternion

    int num_pieces = 8;

    std::cout << "\nProblem setup:\n";
    std::cout << "  Initial position: [" << initial_state.col(0).transpose() << "]\n";
    std::cout << "  Initial velocity: [" << initial_state.col(1).transpose() << "]\n";
    std::cout << "  Target position:  [" << target_position.transpose() << "]\n";
    std::cout << "  Target velocity:  [" << target_velocity.transpose() << "]\n";
    std::cout << "  Number of pieces: " << num_pieces << "\n";

    // Test results storage
    Trajectory traj_opt_result;
    Trajectory trajectory_optimizer_result;
    bool traj_opt_success = false;
    bool trajectory_optimizer_success = false;
    double traj_opt_time = 0.0;
    double trajectory_optimizer_time = 0.0;

    // Test 1: Original TrajOpt class
    {
        std::cout << "\n=== Testing Original TrajOpt Class ===\n";
        traj_opt::TrajOpt optimizer;

        // Configure identical parameters
        optimizer.setDynamicLimits(15.0, 12.0, 25.0, 3.0, 4.0, 2.5);
        optimizer.setRobotParameters(1.2, 0.35, 0.12, 0.6);
        optimizer.setOptimizationWeights(1.5, -1.0, 1.2, 1.1, 1.0, 1.3, 1.1, 1.4);
        optimizer.setIntegrationSteps(25);
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
            std::cout << "✓ TrajOpt optimization successful!\n";
            std::cout << "  Optimization time: " << std::fixed << std::setprecision(2) << traj_opt_time << " ms\n";
            printTrajectoryStats("TrajOpt", traj_opt_result);
        } else {
            std::cout << "✗ TrajOpt optimization failed!\n";
        }
    }

    // Test 2: New TrajectoryOptimizer class
    {
        std::cout << "\n=== Testing New TrajectoryOptimizer Class ===\n";
        traj_opt::TrajectoryOptimizer optimizer;

        // Configure identical parameters
        optimizer.setDynamicLimits(15.0, 12.0, 25.0, 3.0, 4.0, 2.5);
        optimizer.setRobotParameters(1.2, 0.35, 0.12, 0.6);
        optimizer.setOptimizationWeights(1.5, -1.0, 1.2, 1.1, 1.0, 1.3, 1.1, 1.4);
        optimizer.setIntegrationParameters(25);
        optimizer.setDebugMode(false);

        std::cout << "Configuration complete. Starting optimization...\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        trajectory_optimizer_success = optimizer.generateTrajectory(
            initial_state,
            target_position,
            target_velocity,
            landing_quaternion,
            num_pieces,
            trajectory_optimizer_result);

        auto end_time = std::chrono::high_resolution_clock::now();
        trajectory_optimizer_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        if (trajectory_optimizer_success) {
            std::cout << "✓ TrajectoryOptimizer optimization successful!\n";
            std::cout << "  Optimization time: " << std::fixed << std::setprecision(2) << trajectory_optimizer_time << " ms\n";
            printTrajectoryStats("TrajectoryOptimizer", trajectory_optimizer_result);
        } else {
            std::cout << "✗ TrajectoryOptimizer optimization failed!\n";
        }
    }

    // Compare results
    std::cout << "\n=== Comparison Results ===\n";

    if (traj_opt_success && trajectory_optimizer_success) {
        std::cout << "Both optimizers succeeded. Comparing trajectories...\n";

        bool trajectories_similar = compareTrajectories(traj_opt_result, trajectory_optimizer_result, 1e-2);

        if (trajectories_similar) {
            // FIXME: trajectory mismatch issue
            std::cout << "✓ Trajectories are very similar! Both optimizers produced consistent results.\n";
        } else {
            std::cout << "\033[1;33m⚠ Trajectories differ significantly. This may indicate implementation differences.\033[0m\n";
        }

        // Performance comparison
        double time_ratio = trajectory_optimizer_time / traj_opt_time;
        std::cout << "\nPerformance comparison:\n";
        std::cout << "  TrajOpt time:           " << std::fixed << std::setprecision(2) << traj_opt_time << " ms\n";
        std::cout << "  TrajectoryOptimizer time: " << std::fixed << std::setprecision(2) << trajectory_optimizer_time << " ms\n";
        std::cout << "  Speed ratio (New/Old):  " << std::fixed << std::setprecision(2) << time_ratio << "x\n";

        if (time_ratio < 0.9) {
            std::cout << "  → TrajectoryOptimizer is faster\n";
        } else if (time_ratio > 1.1) {
            std::cout << "  → TrajOpt is faster\n";
        } else {
            std::cout << "  → Performance is comparable\n";
        }

    } else if (traj_opt_success && !trajectory_optimizer_success) {
        std::cout << "Only TrajOpt succeeded. TrajectoryOptimizer needs debugging.\n";
    } else if (!traj_opt_success && trajectory_optimizer_success) {
        std::cout << "Only TrajectoryOptimizer succeeded. TrajOpt may have issues.\n";
    } else {
        std::cout << "Both optimizers failed. Check problem setup and dependencies.\n";
    }

    // Feasibility checks if trajectories exist
    if (traj_opt_success) {
        std::cout << "\nRunning feasibility check on TrajOpt result...\n";
        traj_opt::TrajOpt feasibility_checker;
        feasibility_checker.setDynamicLimits(15.0, 12.0, 25.0, 3.0, 4.0, 2.5);
        bool is_feasible = feasibility_checker.feasibleCheck(traj_opt_result);
        std::cout << "  TrajOpt result feasible: " << (is_feasible ? "✓ Yes" : "✗ No") << std::endl;
    }

    if (trajectory_optimizer_success) {
        std::cout << "\nRunning feasibility check on TrajectoryOptimizer result...\n";
        traj_opt::TrajectoryOptimizer feasibility_checker;
        feasibility_checker.setDynamicLimits(15.0, 12.0, 25.0, 3.0, 4.0, 2.5);
        bool is_feasible = feasibility_checker.checkFeasibility(trajectory_optimizer_result);
        std::cout << "  TrajectoryOptimizer result feasible: " << (is_feasible ? "✓ Yes" : "✗ No") << std::endl;
    }

    return 0;
}