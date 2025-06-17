#!/bin/bash

# This script sets up Docker proxy settings, installs necessary Docker containers,
# and downloads required models for vLLM benchmarking.
# It is designed to be run on a server with Docker installed.
# The script assumes the following:
# - Docker is installed and running on the server.
# Usage:
#   ./build_containers.sh [gpu] # Optional argument to specify GPU setup.

# CPU server container: vllm:ipex-cpu-ww09
# GPU server container: vllm/vllm-openai:0.9.1
# Benchmark container: vllm:0.8.0
set -e

function setup_docker_proxy() {
    # Ensure the script runs with root privileges
    if [ "$EUID" -ne 0 ]; then
      echo "Please run as root"
      exit
    fi

    # Define proxy settings
    HTTP_PROXY="http://proxy-dmz.intel.com:912"
    HTTPS_PROXY="http://proxy-dmz.intel.com:912"
    NO_PROXY="intel.com,.intel.com,localhost,127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12"

    # Docker daemon configuration file
    DOCKER_CONFIG_FILE="/etc/docker/daemon.json"

    # Create or update the daemon.json file
    if [ ! -f "$DOCKER_CONFIG_FILE" ]; then
      echo "{}" > "$DOCKER_CONFIG_FILE"
    fi

    sudo apt install jq -y
    # Use jq to add proxy settings to the daemon.json file
    jq ". + {\"proxies\": {\"default\": {\"httpProxy\": \"$HTTP_PROXY\", \"httpsProxy\": \"$HTTPS_PROXY\", \"noProxy\": \"$NO_PROXY\"}}}" \
      "$DOCKER_CONFIG_FILE" > /tmp/daemon_temp.json && mv /tmp/daemon_temp.json "$DOCKER_CONFIG_FILE"

    # Restart the Docker service to apply changes
    echo "Restarting Docker service..."
    systemctl restart docker

    # Verify the changes
    echo "Docker daemon proxy settings:"
    cat "$DOCKER_CONFIG_FILE"

    # Create an empty config.json if it doesn't exist
    if [ ! -f ~/.docker/config.json ]; then
      echo '{}' > ~/.docker/config.json
    fi

    jq '. + {"proxies": {"default": {"httpProxy": "http://proxy-dmz.intel.com:912", "httpsProxy": "http://proxy-dmz.intel.com:912", "noProxy": "intel.com,.intel.com,localhost,127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12"}}}' ~/.docker/config.json > ~/.docker/config_temp.json && \
    mv ~/.docker/config_temp.json ~/.docker/config.json
}

function install_docker_containers() {

    mkdir -p test
    pushd test

    git clone -b ipex-cpu-ww09 https://github.com/intel-sandbox/vllm-xpu.git
    pushd vllm-xpu
    docker build -f Dockerfile.cpu -t vllm:ipex-cpu-ww09 .

    git checkout v0.8.0
    docker build -f Dockerfile.cpu -t vllm:v0.8.0 .
    popd

    # get tester scripts
    git clone -b quick_aws_embedding https://github.com/weilinwa/tuner.git
    cd tuner

    # Build ngix
    pushd nginx
    docker build -t nginx-lb:latest -f Dockerfile.nginx .
    bash launch_nginx.sh
    popd

    # Create results directory
    mkdir -p results

}

function install_docker_containers_gpu() {

  mkdir -p test

  pushd test

  git clone -b v0.8.0 https://github.com/vllm-project/vllm.git
  pushd vllm
  # Build docker image for CPU (for benchmark container)
  docker build \
    --build-arg http_proxy=http://proxy-dmz.intel.com:912 \
    --build-arg http_proxy=http://proxy-dmz.intel.com:912 \
    --build-arg no_proxy=localhost,127.0.0.0/8,intel.com \
    -t vllm:0.8.0 -f Dockerfile.cpu .

  popd

  # Pull vllm-gpu docker image
  docker pull vllm/vllm-openai:0.9.1

  # get tester scripts
  git clone -b quick_aws_embedding https://github.com/weilinwa/tuner.git
  pushd tuner

  # Build ngix
  pushd nginx
  docker build -t nginx-lb:latest -f Dockerfile.nginx .
  bash launch_nginx.sh
  popd

  # Create results directory
  mkdir -p results
}

function download_models() {

    python3 -m venv vllm_venv
    source vllm_venv/bin/activate
    # Install huggingface-cli
    python -m pip install huggingface_hub
    # Download models
    echo "Downloading models..."
    # Download the model
    # Note: Replace `hf_XXXXXXXX` with your actual Hugging Face token.
    # huggingface-cli login --token hf_XXXXXXXX
    huggingface-cli download ibm-granite/granite-embedding-278m-multilingual --cache-dir ./models/.cache/huggingface/hub/
    # Download large model from cmdline. Rest of the models are downloaded in the tester script.
    huggingface-cli download Salesforce/SFR-Embedding-Mistral --cache-dir ./models/.cache/huggingface/hub/
    #huggingface-cli download intfloat/multilingual-e5-large-instruct --cache-dir ./models/.cache/huggingface/hub/
    #huggingface-cli download NovaSearch/stella_en_1.5B_v5 --cache-dir ./models/.cache/huggingface/hub/
    #huggingface-cli download intfloat/multilingual-e5-small --cache-dir ./models/.cache/huggingface/hub/
}

# Install docker containers and download models
setup_docker_proxy
if [ "$1" == "gpu" ]; then
    install_docker_containers_gpu
else
    install_docker_containers
fi
download_models


## Run the tester script
#platform=$2
#python tester.py -p $platform -e -b