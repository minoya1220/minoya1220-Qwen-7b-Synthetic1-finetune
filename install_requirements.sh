#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Starting installation of Qwen-7b-Synthetic1-SFT dependencies ===${NC}"
echo -e "${YELLOW}=== This may take some time depending on your internet connection ===${NC}"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Python is not installed. Please install Python 3.9+ and try again.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo -e "${GREEN}Detected Python version: $PYTHON_VERSION${NC}"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}Python 3.9 or higher is required. Your version is $PYTHON_VERSION${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo -e "${RED}pip is not installed. Please install pip and try again.${NC}"
    exit 1
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
    
    # Activate virtual environment based on OS
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi
    
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
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
            echo -e "${YELLOW}To activate the virtual environment in the future, run: ${GREEN}source .venv/Scripts/activate${NC}"
        else
            echo -e "${YELLOW}To activate the virtual environment in the future, run: ${GREEN}source .venv/bin/activate${NC}"
        fi
    fi
else
    echo -e "${RED}=== Installation failed. Please check the error messages above. ===${NC}"
    exit 1
fi 