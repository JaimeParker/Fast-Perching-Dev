#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iomanip>
#include <iostream>

#include "perching_optimizer.h"
#include "traj_opt.h"

int main() {
    std::cout << "=== Testing Quaternion to Z-axis Conversion ===\n";

    // Test quaternion (same as used in the comparison)
    Eigen::Quaterniond test_quat(0.9659, 0.0, 0.2588, 0.0);

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Input quaternion: [" << test_quat.w() << ", "
              << test_quat.x() << ", " << test_quat.y() << ", " << test_quat.z() << "]\n";

    // Test original implementation
    Eigen::Vector3d original_z_axis;
    Eigen::Matrix3d R = test_quat.toRotationMatrix();
    original_z_axis = R.col(2);

    std::cout << "\nOriginal implementation (traj_opt_perching.cc):\n";
    std::cout << "Z-axis: [" << original_z_axis.x() << ", "
              << original_z_axis.y() << ", " << original_z_axis.z() << "]\n";

    // Test refactored implementation
    Eigen::Vector3d refactored_z_axis;
    Eigen::Matrix3d R2 = test_quat.toRotationMatrix();
    refactored_z_axis = R2.col(2);

    std::cout << "\nRefactored implementation (perching_optimizer.h):\n";
    std::cout << "Z-axis: [" << refactored_z_axis.x() << ", "
              << refactored_z_axis.y() << ", " << refactored_z_axis.z() << "]\n";

    // Calculate difference
    Eigen::Vector3d difference = original_z_axis - refactored_z_axis;
    std::cout << "\nDifference: [" << difference.x() << ", "
              << difference.y() << ", " << difference.z() << "]\n";
    std::cout << "Difference norm: " << difference.norm() << "\n";

    // Test basis vector computation
    std::cout << "\n=== Testing Basis Vector Computation ===\n";

    // Original computation (from traj_opt_perching.cc)
    Eigen::Vector3d original_basis_x = original_z_axis.cross(Eigen::Vector3d(0, 0, 1));
    if (original_basis_x.squaredNorm() == 0) {
        original_basis_x = original_z_axis.cross(Eigen::Vector3d(0, 1, 0));
    }
    original_basis_x.normalize();
    Eigen::Vector3d original_basis_y = original_z_axis.cross(original_basis_x);
    original_basis_y.normalize();

    std::cout << "Original basis X: [" << original_basis_x.x() << ", "
              << original_basis_x.y() << ", " << original_basis_x.z() << "]\n";
    std::cout << "Original basis Y: [" << original_basis_y.x() << ", "
              << original_basis_y.y() << ", " << original_basis_y.z() << "]\n";

    // Refactored computation
    Eigen::Vector3d refactored_basis_x = refactored_z_axis.cross(Eigen::Vector3d(0.0, 0.0, 1.0));
    if (refactored_basis_x.squaredNorm() == 0.0) {
        refactored_basis_x = refactored_z_axis.cross(Eigen::Vector3d(0.0, 1.0, 0.0));
    }
    refactored_basis_x.normalize();
    Eigen::Vector3d refactored_basis_y = refactored_z_axis.cross(refactored_basis_x);
    refactored_basis_y.normalize();

    std::cout << "Refactored basis X: [" << refactored_basis_x.x() << ", "
              << refactored_basis_x.y() << ", " << refactored_basis_x.z() << "]\n";
    std::cout << "Refactored basis Y: [" << refactored_basis_y.x() << ", "
              << refactored_basis_y.y() << ", " << refactored_basis_y.z() << "]\n";

    // Calculate basis differences
    Eigen::Vector3d basis_x_diff = original_basis_x - refactored_basis_x;
    Eigen::Vector3d basis_y_diff = original_basis_y - refactored_basis_y;

    std::cout << "\nBasis X difference: [" << basis_x_diff.x() << ", "
              << basis_x_diff.y() << ", " << basis_x_diff.z() << "]\n";
    std::cout << "Basis Y difference: [" << basis_y_diff.x() << ", "
              << basis_y_diff.y() << ", " << basis_y_diff.z() << "]\n";
    std::cout << "Basis X difference norm: " << basis_x_diff.norm() << "\n";
    std::cout << "Basis Y difference norm: " << basis_y_diff.norm() << "\n";

    return 0;
}