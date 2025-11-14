#!/usr/bin/env python
"""
Verify that CausalBench setup is correct.
"""

import sys
from pathlib import Path

def check_imports():
    """Check that all required packages can be imported."""
    print("Checking imports...")
    
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("dowhy", "dowhy"),
        ("econml", "econml"),
        ("causallearn", "causal-learn"),
        ("graphviz", "graphviz"),
    ]
    
    missing = []
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} (missing)")
            missing.append(package_name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def check_directories():
    """Check that required directories exist."""
    print("\nChecking directories...")
    
    required_dirs = [
        "src",
        "notebooks",
        "scripts",
        "tests",
        "data",
        "results",
        "output"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    return all_exist


def check_modules():
    """Check that source modules exist."""
    print("\nChecking modules...")
    
    required_modules = [
        "src/data_loader.py",
        "src/dag_builder.py",
        "src/dowhy_pipeline.py",
        "src/econml_estimators.py",
        "src/discovery.py",
        "src/metrics.py",
        "src/viz.py",
    ]
    
    all_exist = True
    for module_path in required_modules:
        if Path(module_path).exists():
            print(f"  ✓ {module_path}")
        else:
            print(f"  ✗ {module_path} (missing)")
            all_exist = False
    
    return all_exist


def main():
    """Run all checks."""
    print("=" * 50)
    print("CausalBench Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Imports", check_imports),
        ("Directories", check_directories),
        ("Modules", check_modules),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All checks passed! CausalBench is ready to use.")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
