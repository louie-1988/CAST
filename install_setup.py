#!/usr/bin/env python3
"""
Setup script for CAST pipeline installation and validation
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, critical=True):
    """Run a shell command with error handling"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Command: {cmd}")
        print(f"  Error: {e.stderr}")
        if critical:
            print("  This is a critical dependency. Installation cannot continue.")
            sys.exit(1)
        return False

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} detected")
        print("  Python 3.8+ is required")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")

def install_dependencies():
    """Install Python dependencies"""
    print("\n" + "="*50)
    print("INSTALLING DEPENDENCIES")
    print("="*50)
    
    # Install basic requirements
    run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing basic dependencies"
    )
    
    # Install MoGe
    run_command(
        f"{sys.executable} -m pip install git+https://github.com/microsoft/MoGe.git",
        "Installing MoGe depth estimation model"
    )
    
    # Install package in development mode
    run_command(
        f"{sys.executable} -m pip install -e .",
        "Installing CAST package in development mode"
    )

def setup_environment():
    """Setup environment files"""
    print("\n" + "="*50)
    print("SETTING UP ENVIRONMENT")
    print("="*50)
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy example env file
        env_file.write_text(env_example.read_text())
        print("✓ Created .env file from template")
        print("  Please edit .env file and add your API keys")
    elif env_file.exists():
        print("✓ .env file already exists")
    else:
        print("✗ No .env.example file found")


def validate_installation():
    """Validate the installation"""
    print("\n" + "="*50)
    print("VALIDATING INSTALLATION")
    print("="*50)
    
    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("open3d", "Open3D"),
        ("trimesh", "Trimesh"),
        ("replicate", "Replicate"),
        ("openai", "OpenAI"),
    ]
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✓ {name} import successful")
        except ImportError:
            print(f"✗ {name} import failed")
    
    # Test CAST imports
    try:
        from cast.core.pipeline import create_pipeline
        print("✓ CAST pipeline import successful")
    except ImportError as e:
        print(f"✗ CAST pipeline import failed: {e}")
    
    # Test MoGe import
    try:
        from moge.model.v2 import MoGeModel
        print("✓ MoGe model import successful")
    except ImportError:
        print("✗ MoGe model import failed")
        print("  Try: pip install git+https://github.com/microsoft/MoGe.git")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    
    print("1. Edit the .env file and add your API keys:")
    print("   - REPLICATE_API_TOKEN")
    print("   - TRIPO3D_API_KEY") 
    print("   - DASHSCOPE_API_KEY")
    print()
    print("2. Test the installation:")
    print("   python -m cast --validate-only")
    print()
    print("3. Run on a test image:")
    print("   python -m cast --image path/to/your/image.jpg")
    print()
    print("4. See README.md for detailed usage instructions")

def main():
    """Main setup function"""
    print("CAST Pipeline Setup")
    print("="*50)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment()
    
    
    # Validate installation
    validate_installation()
    
    # Print next steps
    print_next_steps()
    
    print("\n✓ Setup complete!")

if __name__ == "__main__":
    main()