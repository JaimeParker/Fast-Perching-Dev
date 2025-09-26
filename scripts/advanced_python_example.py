#!/usr/bin/env python3
"""
Advanced example demonstrating PerchingOptimizer Python bindings
This example shows how to use the bindings for trajectory optimization,
visualization, and analysis in Python.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Import the perching optimizer module using our utility
from import_utils import import_perching_optimizer

po = import_perching_optimizer()
if po is None:
    sys.exit(1)

# Also get the project root for output paths
script_dir = Path(__file__).parent
project_root = script_dir.parent

def create_perching_scenario():
    """Create a challenging perching scenario"""
    
    # Initial state: hovering drone
    initial_pos = np.array([0.0, 0.0, 5.0])
    initial_vel = np.array([3.0, 1.0, -1.0])  
    initial_acc = np.array([0.0, 0.0, 0.0])
    initial_state = po.createInitialState(initial_pos, initial_vel, initial_acc)
    
    # Target: perching on a tilted platform
    target_pos = np.array([15.0, 8.0, 2.0])
    target_vel = np.array([2.0, 1.0, 0.0])
    
    # Landing orientation: 15-degree tilt
    angle_rad = np.radians(15)
    landing_quat = po.createQuaternion(np.cos(angle_rad/2), 0.0, np.sin(angle_rad/2), 0.0)
    
    return initial_state, target_pos, target_vel, landing_quat

def optimize_trajectory():
    """Optimize a perching trajectory using Python bindings"""
    
    print("=== Advanced PerchingOptimizer Example ===\n")
    
    # Create scenario
    initial_state, target_pos, target_vel, landing_quat = create_perching_scenario()
    
    print("Scenario setup:")
    print(f"  Initial position: {initial_state[:, 0]}")
    print(f"  Initial velocity: {initial_state[:, 1]}")
    print(f"  Target position:  {target_pos}")
    print(f"  Target velocity:  {target_vel}")
    print(f"  Landing orientation: [{landing_quat.w():.3f}, {landing_quat.x():.3f}, {landing_quat.y():.3f}, {landing_quat.z():.3f}]")
    
    # Create and configure optimizer
    optimizer = (po.PerchingOptimizer()
                   .setDynamicLimits(12.0, 8.0, 25.0, 3.0, 4.0, 2.5)
                   .setRobotParameters(1.2, 0.35, 0.12, 0.6)
                   .setOptimizationWeights(1.0, -0.8, 1.2, 1.1, 1.0, 0.9, 1.0, 1.5)
                   .setIntegrationSteps(25)
                   .setDebugMode(False))
    
    print(f"\n✓ Optimizer configured with enhanced parameters")
    
    # Generate trajectory
    print("Generating trajectory...")
    success, trajectory = optimizer.generateTrajectory(
        initial_state, target_pos, target_vel, landing_quat, 8
    )
    
    if not success:
        print("✗ Trajectory generation failed!")
        return None
        
    print("✓ Trajectory generated successfully!")
    print(f"  Duration: {trajectory.getTotalDuration():.2f}s")
    print(f"  Pieces: {trajectory.getPieceNum()}")
    print(f"  Max velocity: {trajectory.getMaxVelRate():.2f} m/s")
    print(f"  Max acceleration: {trajectory.getMaxAccRate():.2f} m/s²")
    print(f"  Max thrust: {trajectory.getMaxThrust():.2f} N")
    
    return trajectory

def analyze_trajectory(trajectory):
    """Analyze trajectory properties"""
    
    print("\n=== Trajectory Analysis ===")
    
    # Sample trajectory at regular intervals
    duration = trajectory.getTotalDuration()
    dt = 0.1
    times = np.arange(0, duration + dt, dt)
    
    positions = np.array([trajectory.getPos(t) for t in times])
    velocities = np.array([trajectory.getVel(t) for t in times])
    accelerations = np.array([trajectory.getAcc(t) for t in times])
    
    # Compute derived metrics
    speeds = np.linalg.norm(velocities, axis=1)
    acc_magnitudes = np.linalg.norm(accelerations, axis=1)
    
    print(f"Position range:")
    print(f"  X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
    print(f"  Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m") 
    print(f"  Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")
    
    print(f"Velocity statistics:")
    print(f"  Mean speed: {speeds.mean():.2f} m/s")
    print(f"  Peak speed: {speeds.max():.2f} m/s")
    print(f"  Final speed: {speeds[-1]:.2f} m/s")
    
    print(f"Acceleration statistics:")
    print(f"  Mean acceleration: {acc_magnitudes.mean():.2f} m/s²")
    print(f"  Peak acceleration: {acc_magnitudes.max():.2f} m/s²")
    
    # Check junction continuity
    print(f"Junction analysis:")
    for i in range(trajectory.getPieceNum() + 1):
        pos = trajectory.getJuncPos(i)
        vel = trajectory.getJuncVel(i)
        print(f"  Junction {i}: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
              f"vel=[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]")
    
    return times, positions, velocities, accelerations

def create_visualization(times, positions, velocities, accelerations):
    """Create comprehensive trajectory visualization"""
    
    print("\n=== Creating Visualization ===")
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                color='green', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                color='red', s=100, marker='s', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # Position vs time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(times, positions[:, 0], 'r-', label='X')
    ax2.plot(times, positions[:, 1], 'g-', label='Y')
    ax2.plot(times, positions[:, 2], 'b-', label='Z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # Velocity vs time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(times, velocities[:, 0], 'r-', label='Vx')
    ax3.plot(times, velocities[:, 1], 'g-', label='Vy')
    ax3.plot(times, velocities[:, 2], 'b-', label='Vz')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # Speed vs time
    ax4 = fig.add_subplot(2, 3, 4)
    speeds = np.linalg.norm(velocities, axis=1)
    ax4.plot(times, speeds, 'b-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Speed vs Time')
    ax4.grid(True)
    
    # Acceleration vs time
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(times, accelerations[:, 0], 'r-', label='Ax')
    ax5.plot(times, accelerations[:, 1], 'g-', label='Ay') 
    ax5.plot(times, accelerations[:, 2], 'b-', label='Az')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Acceleration (m/s²)')
    ax5.set_title('Acceleration vs Time')
    ax5.legend()
    ax5.grid(True)
    
    # Acceleration magnitude vs time
    ax6 = fig.add_subplot(2, 3, 6)
    acc_mags = np.linalg.norm(accelerations, axis=1)
    ax6.plot(times, acc_mags, 'purple', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('|Acceleration| (m/s²)')
    ax6.set_title('Acceleration Magnitude vs Time')
    ax6.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    output_path = project_root / 'assets' / 'python_trajectory_analysis.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    
    return fig

def main():
    """Main function demonstrating advanced usage"""
    
    # Optimize trajectory
    trajectory = optimize_trajectory()
    if trajectory is None:
        return 1
    
    # Analyze trajectory
    times, positions, velocities, accelerations = analyze_trajectory(trajectory)
    
    # Create visualization
    fig = create_visualization(times, positions, velocities, accelerations)
    
    print(f"\n=== Summary ===")
    print(f"✓ Successfully demonstrated PerchingOptimizer Python bindings")
    print(f"✓ Generated {trajectory.getPieceNum()}-piece trajectory")
    print(f"✓ Duration: {trajectory.getTotalDuration():.2f}s")
    print(f"✓ Analysis and visualization completed")
    print(f"✓ Results saved to assets/python_trajectory_analysis.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())