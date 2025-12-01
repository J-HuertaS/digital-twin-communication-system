"""
Script to run all Hamming code tests.
Executes unit tests, performance analysis, and edge cases.
"""

import subprocess
import sys


def print_header(title):
    """Print a formatted header."""
    width = 70
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")


def run_command(description, command):
    """Run a command and display results."""
    print(f"\n>>> {description}")
    print(f"Command: {' '.join(command)}\n")

    try:
        result = subprocess.run(command, capture_output=False, text=True, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Run all test suites."""
    print_header("HAMMING CODE - COMPLETE TEST SUITE")

    all_passed = True

    # 1. Unit Tests with pytest
    print_header("UNIT TESTS (pytest)")
    success = run_command(
        "Running unit tests for all configurations...",
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_hamming_configurations.py",
            "-v",
            "--tb=short",
        ],
    )
    all_passed = all_passed and success

    # 2. Edge Cases
    print_header("EDGE CASE TESTS")
    success = run_command(
        "Running edge case tests...",
        [sys.executable, "-m", "pytest", "tests/test_edge_cases.py", "-v", "--tb=short"],
    )
    all_passed = all_passed and success

    # Summary
    print_header("TEST SUMMARY")
    if all_passed:
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!\n")
        return 0
    else:
        print("✗ Some tests failed. Please review the output above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
