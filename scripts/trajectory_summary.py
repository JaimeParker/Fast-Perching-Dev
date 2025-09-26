#!/usr/bin/env python3
"""
Simple Trajectory Summary Generator

Creates a text-based summary of the trajectory comparison results.
"""

import numpy as np

def load_and_compare_trajectories():
    """Load trajectories and generate comparison summary."""
    
    print("🚁 Perching Trajectory Analysis Summary")
    print("=" * 50)
    
    try:
        # Load original trajectory
        data1 = np.loadtxt('../assets/original_trajectory.csv', delimiter=',', skiprows=1)
        traj1 = {
            'time': data1[:, 0],
            'pos': data1[:, 1:4],
            'vel': data1[:, 4:7], 
            'acc': data1[:, 7:10]
        }
        
        # Load refactored trajectory  
        data2 = np.loadtxt('../assets/perching_optimizer_trajectory.csv', delimiter=',', skiprows=1)
        traj2 = {
            'time': data2[:, 0],
            'pos': data2[:, 1:4],
            'vel': data2[:, 4:7],
            'acc': data2[:, 7:10] 
        }
        
        print(f"✅ Successfully loaded both trajectories")
        
    except Exception as e:
        print(f"❌ Error loading trajectories: {e}")
        return
    
    # Basic comparison
    print(f"\n📊 BASIC COMPARISON:")
    print(f"{'Metric':<25} {'Original':<15} {'Refactored':<15} {'Difference':<15}")
    print("-" * 70)
    
    duration1, duration2 = traj1['time'][-1], traj2['time'][-1]
    print(f"{'Duration (s)':<25} {duration1:<15.3f} {duration2:<15.3f} {abs(duration1-duration2):<15.3f}")
    
    points1, points2 = len(traj1['time']), len(traj2['time'])  
    print(f"{'Data Points':<25} {points1:<15d} {points2:<15d} {abs(points1-points2):<15d}")
    
    # Final positions
    final_pos1, final_pos2 = traj1['pos'][-1], traj2['pos'][-1]
    final_diff = np.linalg.norm(final_pos1 - final_pos2)
    
    print(f"\n🎯 FINAL POSITIONS:")
    print(f"  Original:   [{final_pos1[0]:.3f}, {final_pos1[1]:.3f}, {final_pos1[2]:.3f}]")
    print(f"  Refactored: [{final_pos2[0]:.3f}, {final_pos2[1]:.3f}, {final_pos2[2]:.3f}]")
    print(f"  Difference: {final_diff:.6f}m")
    
    # Dynamic limits check
    vel1_max = np.linalg.norm(traj1['vel'], axis=1).max()
    vel2_max = np.linalg.norm(traj2['vel'], axis=1).max()
    acc1_max = np.linalg.norm(traj1['acc'], axis=1).max()  
    acc2_max = np.linalg.norm(traj2['acc'], axis=1).max()
    
    print(f"\n🚀 DYNAMIC CHARACTERISTICS:")
    print(f"  Max Velocity:")
    print(f"    Original:   {vel1_max:.3f}m/s")
    print(f"    Refactored: {vel2_max:.3f}m/s")
    print(f"  Max Acceleration:")
    print(f"    Original:   {acc1_max:.3f}m/s²") 
    print(f"    Refactored: {acc2_max:.3f}m/s²")
    
    # Check constraint violations (assuming limits from C++ code)
    vel_limit, acc_limit = 10.0, 10.0
    
    print(f"\n⚖️  CONSTRAINT COMPLIANCE:")
    print(f"  Velocity Limit ({vel_limit}m/s):")
    print(f"    Original:   {'✅ OK' if vel1_max <= vel_limit else '❌ VIOLATED'}")
    print(f"    Refactored: {'✅ OK' if vel2_max <= vel_limit else '❌ VIOLATED'}")
    print(f"  Acceleration Limit ({acc_limit}m/s²):")
    print(f"    Original:   {'✅ OK' if acc1_max <= acc_limit else '❌ VIOLATED'}")
    print(f"    Refactored: {'✅ OK' if acc2_max <= acc_limit else '❌ VIOLATED'}")
    
    # Trajectory similarity analysis
    print(f"\n📏 TRAJECTORY SIMILARITY ANALYSIS:")
    
    # Interpolate to common time base
    min_duration = min(duration1, duration2)
    common_time = np.linspace(0, min_duration, 1000)
    
    pos1_interp = np.column_stack([
        np.interp(common_time, traj1['time'], traj1['pos'][:, i]) for i in range(3)
    ])
    pos2_interp = np.column_stack([
        np.interp(common_time, traj2['time'], traj2['pos'][:, i]) for i in range(3)
    ])
    
    position_differences = np.linalg.norm(pos1_interp - pos2_interp, axis=1)
    
    print(f"  Max Position Difference:  {position_differences.max():.6f}m")
    print(f"  Mean Position Difference: {position_differences.mean():.6f}m")
    print(f"  RMS Position Difference:  {np.sqrt(np.mean(position_differences**2)):.6f}m")
    
    # Similarity assessment
    max_diff = position_differences.max()
    if max_diff < 0.1:
        similarity = "🎯 EXCELLENT - Trajectories are nearly identical"
    elif max_diff < 1.0:
        similarity = "✅ GOOD - Minor differences, likely due to numerical precision"
    elif max_diff < 5.0:
        similarity = "⚠️  MODERATE - Noticeable differences, may indicate algorithmic changes"
    else:
        similarity = "❌ POOR - Significant differences, refactoring may have issues"
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"  Similarity Rating: {similarity}")
    
    # Performance implications
    duration_ratio = duration2 / duration1
    if duration_ratio < 0.95:
        performance = "⚡ Refactored version completes faster"
    elif duration_ratio > 1.05:
        performance = "🐌 Refactored version takes longer"
    else:
        performance = "⚖️  Performance is comparable"
        
    print(f"  Performance: {performance}")
    
    # Summary conclusion
    print(f"\n📋 SUMMARY CONCLUSION:")
    both_feasible = vel1_max <= vel_limit and vel2_max <= vel_limit and acc1_max <= acc_limit and acc2_max <= acc_limit
    
    if both_feasible and max_diff < 1.0:
        print(f"  🎉 SUCCESS: Both trajectories are feasible and very similar.")
        print(f"      The refactoring preserved the optimization algorithm correctly.")
    elif both_feasible:
        print(f"  ✅ ACCEPTABLE: Both trajectories are feasible but show some differences.")
        print(f"      This could be due to different local minima in the optimization.")
    else:
        print(f"  ⚠️  WARNING: One or both trajectories may violate constraints.")
        print(f"      Review the refactored implementation for potential issues.")

if __name__ == "__main__":
    load_and_compare_trajectories()