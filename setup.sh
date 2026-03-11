#!/bin/bash
set -e

PROJECT_ROOT="$(pwd)"
echo "📂 Project directory detected as: $PROJECT_ROOT"

ask_permission() {
    echo ""
    read -r -p "$1 (y/n): " confirm
    if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
        echo "Skipping..."
        return 1
    fi
    return 0
}

if ask_permission "Update Ubuntu packages and install essentials (Python3, pip, venv, OpenJDK 17)"; then
    sudo apt update
    sudo apt upgrade -y
    sudo apt install -y python3 python3-pip python3-venv openjdk-17-jdk wget unzip build-essential
fi

if ask_permission "Install Apache Spark 4.1.1 for WSL2"; then
    SPARK_DIR="/opt/spark"
    SPARK_ARCHIVE="spark-4.1.1-bin-hadoop3.tgz"
    SPARK_URL="https://dlcdn.apache.org/spark/spark-4.1.1/${SPARK_ARCHIVE}"

    echo "Preparing Spark installation directory..."
    sudo mkdir -p /opt
    cd /opt || exit 1

    if [ ! -d "$SPARK_DIR" ]; then
        echo "Downloading Spark..."
        sudo wget "$SPARK_URL"

        echo "Extracting Spark..."
        sudo tar -xzf "$SPARK_ARCHIVE"

        sudo mv spark-4.1.1-bin-hadoop3 spark
        sudo chown -R "$USER":"$USER" spark

        echo "Spark installed in $SPARK_DIR"
    else
        echo "Spark already installed at $SPARK_DIR"
    fi

    if ! grep -q "SPARK_HOME=/opt/spark" "$HOME/.bashrc"; then
        {
            echo ""
            echo "# Spark environment"
            echo "export SPARK_HOME=/opt/spark"
            echo "export PATH=\$SPARK_HOME/bin:\$PATH"
            echo "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64"
        } >> "$HOME/.bashrc"
        echo "Spark variables added to ~/.bashrc"
    else
        echo "Spark environment already configured"
    fi
fi

cd "$PROJECT_ROOT" || exit 1

if ask_permission "Create Python virtual environment"; then
    sudo apt install -y python3 python3-venv python3-pip
    python3 -m venv venv

    source venv/bin/activate

    PYSPARK_PYTHON="$(realpath venv/bin/python)"
    export PYSPARK_PYTHON

    PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
    export PYSPARK_DRIVER_PYTHON

    echo "Using venv Python for PySpark: $PYSPARK_PYTHON"

    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies..."
        pip install -r requirements.txt
    fi
fi

if ask_permission "Add dev_utils to your ~/.bashrc using the current path"; then
    LINE_TO_ADD=". \"$PROJECT_ROOT/dev_utils/.bashrc\""

    if grep -Fq "$PROJECT_ROOT/dev_utils/.bashrc" "$HOME/.bashrc"; then
        echo "Entry already exists in ~/.bashrc"
    else
        {
            echo ""
            echo "$LINE_TO_ADD"
        } >> "$HOME/.bashrc"
        echo "Added dev_utils to ~/.bashrc"
    fi
fi

if ask_permission "Check and setup pre-commit"; then
    if ! command -v pre-commit &> /dev/null; then
        echo "pre-commit not found."
        if ask_permission "Install pre-commit now"; then
            python3 -m pip install -U pre-commit
        fi
    else
        echo "pre-commit already installed ($(pre-commit --version))"
    fi

    if command -v pre-commit &> /dev/null; then
        echo "Initializing pre-commit hooks..."
        pre-commit install

        echo "Running pre-commit on all files..."
        pre-commit run --all-files
    fi
fi

if ask_permission "Run minimal PySpark test"; then
    source venv/bin/activate

    python3 <<EOF
from pyspark.sql import SparkSession
import os, sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
df.show()

print("Spark version:", spark.version)
spark.stop()
EOF
fi

echo ""
echo "Setup completed successfully."
echo "Restart your terminal or run: source ~/.bashrc"
