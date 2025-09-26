#!/usr/bin/env python3
"""
Import utilities for PerchingOptimizer Python bindings
Provides robust module importing regardless of execution location
"""

import sys
import os
from pathlib import Path


def setup_perching_optimizer_import():
    """
    Set up the Python path to import perching_optimizer_py module.
    
    This function intelligently locates the compiled Python module
    by searching common build directory locations relative to the
    project structure.
    
    Returns:
        bool: True if module path was found and added, False otherwise
    """
    
    # Get the script's location and work backwards to find project root
    current_file = Path(__file__).resolve()
    
    # Try different ways to find the project root
    search_paths = []
    
    # If we're in a scripts/ subdirectory
    if current_file.parent.name == 'scripts':
        project_root = current_file.parent.parent
        search_paths.append(project_root)
    
    # If we're directly in the project root
    elif (current_file.parent / 'CMakeLists.txt').exists():
        project_root = current_file.parent
        search_paths.append(project_root)
    
    # Search upwards for CMakeLists.txt (project root indicator)
    else:
        current_dir = current_file.parent
        for _ in range(5):  # Limit search depth
            if (current_dir / 'CMakeLists.txt').exists():
                search_paths.append(current_dir)
                break
            current_dir = current_dir.parent
    
    # Try current working directory as well
    cwd = Path.cwd()
    if (cwd / 'CMakeLists.txt').exists():
        search_paths.append(cwd)
    
    # For each potential project root, check common build directories
    build_dir_names = [
        'build',
        'cmake-build-debug',
        'cmake-build-release',
        'build-debug',
        'build-release'
    ]
    
    module_found = False
    
    for project_root in search_paths:
        for build_name in build_dir_names:
            build_dir = project_root / build_name
            if build_dir.exists():
                # Check if the Python module exists in this build directory
                module_files = list(build_dir.glob('perching_optimizer_py*.so'))
                if module_files:
                    sys.path.insert(0, str(build_dir))
                    print(f"Found PerchingOptimizer module in: {build_dir}")
                    module_found = True
                    return True
    
    if not module_found:
        print("Error: Could not locate perching_optimizer_py module.")
        print("\nSearched the following locations:")
        for project_root in search_paths:
            for build_name in build_dir_names:
                build_dir = project_root / build_name
                print(f"  - {build_dir}")
        
        print("\nTo resolve this issue:")
        print("1. Ensure the Python module is built: cd build && make perching_optimizer_py")
        print("2. Run this script from the project root or scripts/ directory")
        print("3. Or set PYTHONPATH manually: export PYTHONPATH=/path/to/build:$PYTHONPATH")
        
        return False
    
    return True


def import_perching_optimizer():
    """
    Import the perching_optimizer_py module with error handling.
    
    Returns:
        module: The imported perching_optimizer_py module, or None if failed
    """
    
    if not setup_perching_optimizer_import():
        return None
    
    try:
        import perching_optimizer_py as po
        print("✓ Successfully imported perching_optimizer_py module")
        return po
    except ImportError as e:
        print(f"✗ Failed to import perching_optimizer_py: {e}")
        print("\nPossible solutions:")
        print("1. Rebuild the Python module: cd build && make perching_optimizer_py")
        print("2. Check that all dependencies are installed (pybind11, Eigen3)")
        print("3. Verify the module was compiled for the correct Python version")
        return None


if __name__ == "__main__":
    # Test the import functionality
    print("Testing PerchingOptimizer module import...")
    po = import_perching_optimizer()
    
    if po:
        print(f"Module version: {po.__version__}")
        print("Import test successful!")
    else:
        print("Import test failed!")
        sys.exit(1)