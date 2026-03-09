#!/bin/bash

# Capture the absolute path of the project root
PROJECT_ROOT=$(pwd)
echo "📂 Project directory detected as: $PROJECT_ROOT"

ask_permission() {
    echo ""
    read -r -p "$1 (y/n): " confirm
    if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
        echo "Skipping this section..."
        return 1
    fi
    return 0
}

if ask_permission "Create virtual environment"; then
    echo "Upgrading pip, virtualenv..."
    python -m pip install --upgrade pip pip virtualenv
    echo "Creating Python virtual environment..."
    python -m venv venv
    source venv/Scripts/activate
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
fi


if ask_permission "Add dev_utils to your ~/.bashrc using the current path?"; then
    # Define the line using the dynamic path
    LINE_TO_ADD=". \"$PROJECT_ROOT/dev_utils/.bashrc\""

    # Check if this specific path or the file name is already in .bashrc
    if grep -Fq "$PROJECT_ROOT/dev_utils/.bashrc" ~/.bashrc; then
        echo "This project path is already in ~/.bashrc. Skipping..."
    else
        {
            echo ""
            echo "$LINE_TO_ADD"
        } >> ~/.bashrc
        echo "Added to ~/.bashrc."
    fi
fi

if ask_permission "Check and setup pre-commit?"; then
    if ! pre-commit --version &> /dev/null; then
        echo "pre-commit not found."
        if ask_permission "Would you like to install pre-commit now?"; then
            python -m pip install -U pre-commit
        fi
    else
        echo "pre-commit is already installed ($(pre-commit --version))."
    fi

    # Initialize and run
    if pre-commit --version &> /dev/null; then
        echo "Initializing pre-commit hooks..."
        pre-commit install
        echo "Running pre-commit against all files to ensure baseline quality..."
        pre-commit run --all-files
    fi
fi
