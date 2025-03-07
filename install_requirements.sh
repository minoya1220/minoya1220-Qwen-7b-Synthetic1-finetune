#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Starting installation of Qwen-7b-Synthetic1-SFT dependencies ===${NC}"
echo -e "${YELLOW}=== This may take some time depending on your internet connection ===${NC}"

# Function to install Python on Linux
install_python() {
    echo -e "${YELLOW}Installing Python 3.9+...${NC}"
    
    # For Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        echo -e "${GREEN}Detected Debian/Ubuntu-based distribution${NC}"
        sudo apt-get update
        sudo apt-get install -y python3 python3-dev python3-venv
        sudo ln -sf /usr/bin/python3 /usr/bin/python
    # For CentOS/RHEL/Fedora
    elif command -v dnf &> /dev/null; then
        echo -e "${GREEN}Detected Fedora/RHEL-based distribution${NC}"
        sudo dnf install -y python39 python39-devel
        sudo ln -sf /usr/bin/python3.9 /usr/bin/python
    # For older CentOS/RHEL
    elif command -v yum &> /dev/null; then
        echo -e "${GREEN}Detected older RHEL/CentOS-based distribution${NC}"
        sudo yum install -y python39 python39-devel
        sudo ln -sf /usr/bin/python3.9 /usr/bin/python
    else
        echo -e "${RED}Unsupported Linux distribution. Please install Python 3.9+ manually.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Python installation completed.${NC}"
}

# Function to install pip
install_pip() {
    echo -e "${YELLOW}Installing pip...${NC}"
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    rm get-pip.py
    echo -e "${GREEN}pip installation completed.${NC}"
}

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}Python is not installed. Attempting to install Python 3.9+...${NC}"
    install_python
    
    # Check again if Python is installed
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Python installation failed. Please install Python 3.9+ manually and try again.${NC}"
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo -e "${GREEN}Detected Python version: $PYTHON_VERSION${NC}"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${YELLOW}Python 3.9 or higher is required. Your version is $PYTHON_VERSION. Attempting to install newer version...${NC}"
    install_python
    
    # Check Python version again
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        echo -e "${RED}Failed to install Python 3.9+. Please install it manually and try again.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Successfully installed Python $PYTHON_VERSION${NC}"
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo -e "${YELLOW}pip is not installed. Attempting to install pip...${NC}"
    install_pip
    
    # Check again if pip is installed
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}pip installation failed. Please install pip manually and try again.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Successfully installed pip${NC}"
fi

# Check if CUDA is available (optional)
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    echo -e "${GREEN}CUDA detected: Driver version $CUDA_VERSION${NC}"
else
    echo -e "${YELLOW}CUDA not detected. Installation will proceed but GPU acceleration won't be available.${NC}"
fi

# Create and activate virtual environment (optional)
echo -e "${YELLOW}Do you want to create a virtual environment? (y/n)${NC}"
read -r create_venv

if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python -m venv .venv
    
    # Activate virtual environment
    source .venv/bin/activate
    
    echo -e "${GREEN}Virtual environment created and activated.${NC}"
    
    # Upgrade pip in the virtual environment
    echo -e "${GREEN}Upgrading pip...${NC}"
    pip install --upgrade pip
fi

# Install requirements
echo -e "${GREEN}Installing requirements from requirements.txt...${NC}"
pip install -r requirements.txt

# Check installation status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}=== All dependencies installed successfully! ===${NC}"
    
    # Print PyTorch CUDA availability
    echo -e "${YELLOW}Checking PyTorch CUDA availability...${NC}"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
    
    echo -e "${GREEN}=== Setup complete! ===${NC}"
    
    if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
        echo -e "${YELLOW}To activate the virtual environment in the future, run: ${GREEN}source .venv/bin/activate${NC}"
    fi
else
    echo -e "${RED}=== Installation failed. Please check the error messages above. ===${NC}"
    exit 1
fi 