import subprocess
import sys

# List the required packages
required_packages = [
    "numpy",
    "opencv-python",
    "scikit-learn",
    "matplotlib",
    "pandas"
]

def install(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed: {package}")
    except subprocess.CalledProcessError:
        print(f"Error installing package: {package}")

def main():
    for package in required_packages:
        install(package)

if __name__ == "__main__":
    main()