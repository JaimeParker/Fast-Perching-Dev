#!/usr/bin/env python3
"""
Test script for PerchingOptimizer Python bindings
"""

import sys
import os
import numpy as np
from pathlib import Path

# Import the perching optimizer module using our utility
from import_utils import import_perching_optimizer

po = import_perching_optimizer()
if po is None:
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of the Python bindings"""
    
    print("\n=== Testing PerchingOptimizer Python Bindings ===")
    
    # Test 1: Create optimizer instance
    print("1. Creating PerchingOptimizer instance...")
    optimizer = po.PerchingOptimizer()
    print("   ✓ PerchingOptimizer created successfully")
    
    # Test 2: Configure the optimizer using method chaining
    print("2. Configuring optimizer parameters...")
    optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0) \
             .setRobotParameters(1.0, 0.3, 0.1, 0.5) \
             .setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0) \
             .setIntegrationSteps(20) \
             .setDebugMode(False)
    print("   ✓ Configuration completed using method chaining")
    
    # Test 3: Create initial state using helper function
    print("3. Creating initial state matrix...")
    initial_pos = np.array([0.0, 0.0, 3.0])
    initial_vel = np.array([2.0, 0.0, -0.5])
    initial_acc = np.array([0.0, 0.0, 0.0])
    initial_state = po.createInitialState(initial_pos, initial_vel, initial_acc)
    print(f"   ✓ Initial state shape: {initial_state.shape}")
    print(f"   ✓ Initial state:\n{initial_state}")
    
    # Test 4: Create target parameters
    print("4. Setting up target parameters...")
    target_pos = np.array([8.0, 2.0, 1.0])
    target_vel = np.array([1.5, 0.5, 0.0])
    landing_quat = po.createQuaternion(0.9659, 0.0, 0.2588, 0.0)
    num_pieces = 6
    print(f"   ✓ Target position: {target_pos}")
    print(f"   ✓ Target velocity: {target_vel}")
    print(f"   ✓ Landing quaternion: [{landing_quat.w():.4f}, {landing_quat.x():.4f}, {landing_quat.y():.4f}, {landing_quat.z():.4f}]")
    
    # Test 5: Generate trajectory
    print("5. Generating trajectory...")
    success, trajectory = optimizer.generateTrajectory(
        initial_state, target_pos, target_vel, landing_quat, num_pieces
    )
    
    if success:
        print("   ✓ Trajectory generation successful!")
        print(f"   ✓ Number of pieces: {trajectory.getPieceNum()}")
        print(f"   ✓ Total duration: {trajectory.getTotalDuration():.3f}s")
        print(f"   ✓ Max velocity: {trajectory.getMaxVelRate():.3f}")
        print(f"   ✓ Max acceleration: {trajectory.getMaxAccRate():.3f}")
        print(f"   ✓ Max thrust: {trajectory.getMaxThrust():.3f}")
        
        # Test trajectory evaluation
        print("6. Testing trajectory evaluation...")
        t_eval = trajectory.getTotalDuration() / 2.0  # Evaluate at midpoint
        pos_mid = trajectory.getPos(t_eval)
        vel_mid = trajectory.getVel(t_eval)
        acc_mid = trajectory.getAcc(t_eval)
        
        print(f"   ✓ Position at t={t_eval:.3f}s: {pos_mid}")
        print(f"   ✓ Velocity at t={t_eval:.3f}s: {vel_mid}")
        print(f"   ✓ Acceleration at t={t_eval:.3f}s: {acc_mid}")
        
        # Test junction queries
        print("7. Testing junction queries...")
        final_pos = trajectory.getJuncPos(trajectory.getPieceNum())
        final_vel = trajectory.getJuncVel(trajectory.getPieceNum())
        print(f"   ✓ Final position: {final_pos}")
        print(f"   ✓ Final velocity: {final_vel}")
        
        return True
    else:
        print("   ✗ Trajectory generation failed!")
        return False

def test_trajectory_class():
    """Test standalone Trajectory class functionality"""
    print("\n=== Testing Trajectory Class ===")
    
    # Create empty trajectory
    traj = po.Trajectory()
    print(f"✓ Empty trajectory pieces: {traj.getPieceNum()}")
    print(f"✓ Empty trajectory duration: {traj.getTotalDuration()}")
    
    return True

def main():
    """Main test function"""
    print("Starting PerchingOptimizer Python binding tests...\n")
    
    # Test module attributes
    print(f"Module version: {po.__version__}")
    
    # Run basic functionality tests
    success1 = test_basic_functionality()
    success2 = test_trajectory_class()
    
    print("\n=== Test Results ===")
    if success1 and success2:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())