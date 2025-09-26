#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iomanip>
#include <iostream>

#include "perching_optimizer.h"
#include "traj_opt.h"

int main() {
    std::cout << "=== Simple Perching Problem (Identical Seeds) ===\n";

    // Extremely simple problem setup
    Eigen::MatrixXd initial_state(3, 4);
    initial_state.setZero();
    initial_state.col(0) = Eigen::Vector3d(0.0, 0.0, 2.0);  // Start position
    initial_state.col(1) = Eigen::Vector3d(1.0, 0.0, 0.0);  // Start velocity

    Eigen::Vector3d target_position(5.0, 0.0, 1.0);             // Simple target
    Eigen::Vector3d target_velocity(0.5, 0.0, 0.0);             // Simple target velocity
    Eigen::Quaterniond landing_quaternion(1.0, 0.0, 0.0, 0.0);  // Identity (no tilt)

    int num_pieces = 4;  // Fewer pieces

    std::cout << "Simple problem setup:\n";
    std::cout << "  Initial pos: [" << initial_state.col(0).transpose() << "]\n";
    std::cout << "  Initial vel: [" << initial_state.col(1).transpose() << "]\n";
    std::cout << "  Target pos:  [" << target_position.transpose() << "]\n";
    std::cout << "  Target vel:  [" << target_velocity.transpose() << "]\n";
    std::cout << "  Pieces: " << num_pieces << "\n\n";

    // Test with identical simple parameters
    double vmax = 5.0, amax = 5.0, thrust_max = 15.0, thrust_min = 3.0;
    double omega_max = 2.0, omega_yaw_max = 1.0;
    double v_plus = 0.5, robot_l = 0.2, robot_r = 0.05, platform_r = 0.3;
    double rhoT = 1.0, rhoVt = -1.0, rhoP = 1.0, rhoV = 1.0, rhoA = 1.0;
    double rhoThrust = 1.0, rhoOmega = 1.0, rhoPerchingCollision = 1.0;
    int K = 10;  // Fewer integration steps

    Trajectory traj1, traj2;

    // Test 1: Original
    {
        traj_opt::TrajOpt optimizer;
        optimizer.setDynamicLimits(vmax, amax, thrust_max, thrust_min, omega_max, omega_yaw_max);
        optimizer.setRobotParameters(v_plus, robot_l, robot_r, platform_r);
        optimizer.setOptimizationWeights(rhoT, rhoVt, rhoP, rhoV, rhoA, rhoThrust, rhoOmega, rhoPerchingCollision);
        optimizer.setIntegrationSteps(K);
        optimizer.setDebugMode(false);

        std::cout << "=== Original TrajOpt ===\n";
        bool success1 = optimizer.generate_traj(initial_state, target_position, target_velocity, landing_quaternion, num_pieces, traj1);
        std::cout << "Success: " << (success1 ? "Yes" : "No") << "\n";
        if (success1) {
            std::cout << "Duration: " << std::fixed << std::setprecision(3) << traj1.getTotalDuration() << "s\n";
            std::cout << "End pos: [" << traj1.getPos(traj1.getTotalDuration()).transpose() << "]\n";
        }
    }

    // Test 2: Refactored
    {
        traj_opt::PerchingOptimizer optimizer;
        optimizer.setDynamicLimits(vmax, amax, thrust_max, thrust_min, omega_max, omega_yaw_max);
        optimizer.setRobotParameters(v_plus, robot_l, robot_r, platform_r);
        optimizer.setOptimizationWeights(rhoT, rhoVt, rhoP, rhoV, rhoA, rhoThrust, rhoOmega, rhoPerchingCollision);
        optimizer.setIntegrationSteps(K);
        optimizer.setDebugMode(false);

        std::cout << "\n=== PerchingOptimizer ===\n";
        bool success2 = optimizer.generateTrajectory(initial_state, target_position, target_velocity, landing_quaternion, num_pieces, traj2);
        std::cout << "Success: " << (success2 ? "Yes" : "No") << "\n";
        if (success2) {
            std::cout << "Duration: " << std::fixed << std::setprecision(3) << traj2.getTotalDuration() << "s\n";
            std::cout << "End pos: [" << traj2.getPos(traj2.getTotalDuration()).transpose() << "]\n";
        }
    }

    // Compare
    if (traj1.getPieceNum() > 0 && traj2.getPieceNum() > 0) {
        double duration_diff = std::abs(traj1.getTotalDuration() - traj2.getTotalDuration());
        Eigen::Vector3d end_pos_diff = traj1.getPos(traj1.getTotalDuration()) - traj2.getPos(traj2.getTotalDuration());

        std::cout << "\n=== Comparison ===\n";
        std::cout << "Duration difference: " << std::fixed << std::setprecision(6) << duration_diff << "s\n";
        std::cout << "End position difference: " << std::fixed << std::setprecision(6) << end_pos_diff.norm() << "m\n";

        if (duration_diff < 0.1 && end_pos_diff.norm() < 0.1) {
            std::cout << "✓ Results are very similar!\n";
        } else {
            std::cout << "⚠ Results differ significantly.\n";
        }
    }

    return 0;
}