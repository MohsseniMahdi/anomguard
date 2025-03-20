# Credit Card Fraud Detection Project

## Prerequisites
Ensure you have the following installed:
  1. Python
  2. pip
  3. make (for Unix-based systems)


## Installation

  1. make install_requirements
     Installs required dependencies from requirements.txt.
  2. make reinstall_package
     Uninstalls the package anomguard if installed, then reinstalls it in editable mode.
  3. make install
     Installs or updates the package.


## Cleaning

  1. make clean
    Removes temporary and cache files, including __pycache__, .coverage, and build artifacts.

## Model Training and Cloud Storage

  1. make run_preprocces
     Trains the machine learning model. In our case, model defined on .env file with MODEL_VERSION variable.
     Saves the trained model to a specified cloud storage location or local location.

## Containerization and Deployment
  1. make build_docker
     Builds the Docker image for the application. This uses docker file.
  2. make run_docker
     Runs the application inside a Docker container.
  3. make stop_docker
     Stops the running Docker container.
  4. make docker_allow
     Sets the GCP project to current (as described in the .env file) project
  4. make docker_create_repo
     Creates repository on the Google Artifact Registry, to store the docker image.
  5. make docker_push
     Pushes the docker image to the registry
  6. make docker_deploy
     deploys the image on Cloud Run, which hosts your api.




- # [**Go to main page**](../README.md)
