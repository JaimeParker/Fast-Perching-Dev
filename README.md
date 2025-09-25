# Trajectory Optimizer

A ROS-independent trajectory optimization library for UAV perching maneuvers.

## Overview

This project provides trajectory optimization capabilities for UAV perching, extracted and refactored from the original ROS-based Fast-Perching implementation. The library includes two main interfaces:

1. **TrajOpt**: The original class interface, cleaned of ROS dependencies
2. **TrajectoryOptimizer**: A new, redesigned interface with improved code clarity

## Features

- **ROS-Independent**: Complete removal of ROS dependencies for use in any C++ project
- **MINCO Trajectory Generation**: Minimum control effort trajectory optimization
- **L-BFGS Optimization**: Efficient gradient-based optimization
- **Collision Avoidance**: Built-in collision detection and avoidance constraints
- **Dynamic Constraints**: Velocity, acceleration, thrust, and angular velocity limits
- **Perching Optimization**: Specialized for UAV perching maneuvers

## Dependencies

- **Eigen3**: Linear algebra library
- **C++14**: Minimum C++ standard required

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Basic Example

```cpp
#include "trajectory_optimizer.h"

// Create optimizer
traj_opt::TrajectoryOptimizer optimizer;

// Configure parameters
optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
optimizer.setRobotParameters(1.0, 0.3, 0.1, 0.5);
optimizer.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

// Set initial state
Eigen::MatrixXd initial_state(3, 4);
initial_state.col(0) = Eigen::Vector3d(0.0, 0.0, 2.0);  // position
initial_state.col(1) = Eigen::Vector3d(5.0, 0.0, 0.0);  // velocity

// Set target
Eigen::Vector3d target_position(10.0, 0.0, 1.0);
Eigen::Vector3d target_velocity(2.0, 0.0, 0.0);
Eigen::Quaterniond landing_quaternion(1.0, 0.0, 0.0, 0.0);

// Generate trajectory
Trajectory result_trajectory;
bool success = optimizer.generateTrajectory(
    initial_state, target_position, target_velocity, 
    landing_quaternion, 10, result_trajectory
);
```

### Configuration Methods

#### Dynamic Limits
```cpp
optimizer.setDynamicLimits(
    max_velocity,        // m/s
    max_acceleration,    // m/s²
    max_thrust,         // N
    min_thrust,         // N
    max_angular_velocity,    // rad/s
    max_yaw_angular_velocity // rad/s
);
```

#### Robot Parameters
```cpp
optimizer.setRobotParameters(
    velocity_plus,   // Landing velocity offset
    robot_length,    // Robot length (m)
    robot_radius,    // Robot radius (m)
    platform_radius  // Landing platform radius (m)
);
```

#### Optimization Weights
```cpp
optimizer.setOptimizationWeights(
    time_weight,              // Time penalty weight
    velocity_tail_weight,     // Terminal velocity weight
    position_weight,          // Position constraint weight
    velocity_weight,          // Velocity constraint weight
    acceleration_weight,      // Acceleration constraint weight
    thrust_weight,           // Thrust constraint weight
    angular_velocity_weight, // Angular velocity constraint weight
    perching_collision_weight // Collision avoidance weight
);
```

## Code Structure

```
├── include/
│   ├── trajectory_optimizer.h  # New redesigned interface
│   ├── traj_opt.h              # Original interface (ROS-free)
│   ├── minco.hpp               # MINCO trajectory generation
│   ├── lbfgs_raw.hpp          # L-BFGS optimization
│   ├── poly_traj_utils.hpp    # Polynomial trajectory utilities
│   └── root_finder.hpp        # Root finding utilities
├── src/
│   └── traj_opt_perching.cc   # Implementation
├── examples/
│   └── basic_example.cpp      # Usage example
└── CMakeLists.txt             # Build configuration
```

## Key Improvements

- **Removed ROS Dependencies**: Complete elimination of `ros::NodeHandle` and ROS-specific code
- **Improved Variable Naming**: Clear, descriptive variable names (e.g., `car_p_` → `current_position_`)
- **Consistent Formatting**: 4-space indentation and consistent code style
- **Better API Design**: Clean setter methods instead of parameter loading
- **Header-Only Option**: New TrajectoryOptimizer class is fully implemented in header
- **Thread Safety**: Proper handling of static variables in optimization callbacks

## Original Attribution

This code is derived from the Fast-Perching project:
- **Repository**: [ZJU-FAST-Lab/Fast-Perching](https://github.com/ZJU-FAST-Lab/Fast-Perching)
- **Original Authors**: ZJU FAST Lab

## License

Please refer to the original Fast-Perching repository for licensing information.