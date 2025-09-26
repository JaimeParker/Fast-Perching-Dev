/**
 * @file perching_optimizer_api.cc
 * @brief Python bindings for PerchingOptimizer using pybind11
 * @author Zhaohong Liu and Claude Sonnet4 Agent
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "perching_optimizer.h"

namespace py = pybind11;

PYBIND11_MODULE(perching_optimizer_py, m) {
    m.doc() = "Python bindings for PerchingOptimizer trajectory optimization";

    // First, bind the Trajectory class which is a dependency of PerchingOptimizer
    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<>(), "Default constructor")
        .def(py::init<const std::vector<double>&, const std::vector<CoefficientMat>&>(),
             "Constructor with durations and coefficient matrices",
             py::arg("durations"), py::arg("coefficient_matrices"))

        // Core query methods
        .def("getPieceNum", &Trajectory::getPieceNum,
             "Get the number of trajectory pieces")
        .def("getTotalDuration", &Trajectory::getTotalDuration,
             "Get the total duration of the trajectory")

        // State evaluation methods
        .def("getPos", &Trajectory::getPos,
             "Get position at time t",
             py::arg("t"))
        .def("getVel", &Trajectory::getVel,
             "Get velocity at time t",
             py::arg("t"))
        .def("getAcc", &Trajectory::getAcc,
             "Get acceleration at time t",
             py::arg("t"))

        // Additional useful methods
        .def("getDurations", &Trajectory::getDurations,
             "Get durations of all trajectory pieces")
        .def("getPositions", &Trajectory::getPositions,
             "Get positions at all junctions")
        .def("getMaxVelRate", &Trajectory::getMaxVelRate,
             "Get maximum velocity magnitude")
        .def("getMaxAccRate", &Trajectory::getMaxAccRate,
             "Get maximum acceleration magnitude")
        .def("getMaxThrust", &Trajectory::getMaxThrust,
             "Get maximum thrust magnitude")

        // Junction state methods
        .def("getJuncPos", &Trajectory::getJuncPos,
             "Get position at junction index",
             py::arg("junction_index"))
        .def("getJuncVel", &Trajectory::getJuncVel,
             "Get velocity at junction index",
             py::arg("junction_index"))
        .def("getJuncAcc", &Trajectory::getJuncAcc,
             "Get acceleration at junction index",
             py::arg("junction_index"));

    // Bind the main PerchingOptimizer class
    py::class_<traj_opt::PerchingOptimizer>(m, "PerchingOptimizer")
        .def(py::init<>(), "Default constructor")

        // Configuration methods - designed for method chaining
        .def("setDynamicLimits", [](traj_opt::PerchingOptimizer& self, double max_velocity, double max_acceleration, double max_thrust, double min_thrust, double max_omega, double max_yaw_omega) -> traj_opt::PerchingOptimizer& {
                 self.setDynamicLimits(max_velocity, max_acceleration, max_thrust, min_thrust, max_omega, max_yaw_omega);
                 return self; }, "Set dynamic constraints for the vehicle", py::arg("max_velocity"), py::arg("max_acceleration"), py::arg("max_thrust"), py::arg("min_thrust"), py::arg("max_omega"), py::arg("max_yaw_omega"), py::return_value_policy::reference_internal)

        .def("setRobotParameters", [](traj_opt::PerchingOptimizer& self, double landing_speed_offset, double tail_length, double body_radius, double platform_radius) -> traj_opt::PerchingOptimizer& {
                 self.setRobotParameters(landing_speed_offset, tail_length, body_radius, platform_radius);
                 return self; }, "Set physical parameters of the robot", py::arg("landing_speed_offset"), py::arg("tail_length"), py::arg("body_radius"), py::arg("platform_radius"), py::return_value_policy::reference_internal)

        .def("setOptimizationWeights", [](traj_opt::PerchingOptimizer& self, double time_weight, double tail_velocity_weight, double position_weight, double velocity_weight, double acceleration_weight, double thrust_weight, double omega_weight, double perching_collision_weight) -> traj_opt::PerchingOptimizer& {
                 self.setOptimizationWeights(time_weight, tail_velocity_weight, position_weight, velocity_weight,
                                           acceleration_weight, thrust_weight, omega_weight, perching_collision_weight);
                 return self; }, "Set weights for different cost terms in optimization", py::arg("time_weight"), py::arg("tail_velocity_weight"), py::arg("position_weight"), py::arg("velocity_weight"), py::arg("acceleration_weight"), py::arg("thrust_weight"), py::arg("omega_weight"), py::arg("perching_collision_weight"), py::return_value_policy::reference_internal)

        .def("setIntegrationSteps", [](traj_opt::PerchingOptimizer& self, int integration_steps) -> traj_opt::PerchingOptimizer& {
                 self.setIntegrationSteps(integration_steps);
                 return self; }, "Set number of integration steps for trajectory evaluation", py::arg("integration_steps"), py::return_value_policy::reference_internal)

        .def("setDebugMode", [](traj_opt::PerchingOptimizer& self, bool debug_pause) -> traj_opt::PerchingOptimizer& {
                 self.setDebugMode(debug_pause);
                 return self; }, "Enable or disable debug pause mode", py::arg("debug_pause"),
             py::return_value_policy::reference_internal)  // Main trajectory generation method - wrapped to return tuple
        .def("generateTrajectory", [](traj_opt::PerchingOptimizer& self, const Eigen::MatrixXd& initial_state, const Eigen::Vector3d& target_pos, const Eigen::Vector3d& target_vel, const Eigen::Quaterniond& landing_quat, int num_pieces, const double& replanning_time = -1.0) -> py::tuple {
                 
                 Trajectory trajectory;
                 bool success = self.generateTrajectory(initial_state, target_pos, 
                                                       target_vel, landing_quat, 
                                                       num_pieces, trajectory, 
                                                       replanning_time);
                 return py::make_tuple(success, trajectory); }, "Generate perching trajectory. Returns (success, trajectory) tuple", py::arg("initial_state"), py::arg("target_pos"), py::arg("target_vel"), py::arg("landing_quat"), py::arg("num_pieces"), py::arg("replanning_time") = -1.0);

    // Bind additional classes/enums that might be useful

    // Bind Eigen types for convenience (if not already bound by pybind11/eigen.h)
    // These are typically handled automatically by pybind11/eigen.h but we can be explicit

    // Add module-level constants or utility functions if needed
    m.attr("__version__") = "1.0.0";

    // Add some utility functions for creating common matrix types
    m.def("createInitialState", [](const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, const Eigen::Vector3d& acc, const Eigen::Vector3d& jerk) -> Eigen::MatrixXd {
              Eigen::MatrixXd state(3, 4);
              state.col(0) = pos;
              state.col(1) = vel;
              state.col(2) = acc;
              state.col(3) = jerk;
              return state; }, "Helper function to create initial state matrix", py::arg("position"), py::arg("velocity"), py::arg("acceleration") = Eigen::Vector3d::Zero(), py::arg("jerk") = Eigen::Vector3d::Zero());

    m.def("createQuaternion", [](double w, double x, double y, double z) { return Eigen::Quaterniond(w, x, y, z).normalized(); }, "Helper function to create and normalize a quaternion", py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"));

    // Bind Eigen::Quaterniond explicitly if needed
    py::class_<Eigen::Quaterniond>(m, "Quaterniond")
        .def(py::init<double, double, double, double>(),
             "Constructor with w, x, y, z components",
             py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def("w", [](const Eigen::Quaterniond& q) { return q.w(); }, "Get w component")
        .def("x", [](const Eigen::Quaterniond& q) { return q.x(); }, "Get x component")
        .def("y", [](const Eigen::Quaterniond& q) { return q.y(); }, "Get y component")
        .def("z", [](const Eigen::Quaterniond& q) { return q.z(); }, "Get z component")
        .def("normalized", [](const Eigen::Quaterniond& q) { return q.normalized(); }, "Get normalized quaternion")
        .def("norm", [](const Eigen::Quaterniond& q) { return q.norm(); }, "Get quaternion norm");
}