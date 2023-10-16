# Predict Fuel Efficiency Using Tensorflow in Python - Dockerized

This project demonstrates how to build a fuel efficiency predicting model using TensorFlow in Python. The model uses a dataset containing features like the distance the engine has traveled, the number of cylinders in the car, and other relevant features. The goal is to predict the miles per gallon (MPG) of a car based on these features.

## Project Structure

`train.py`: This script contains the code for data preprocessing, model training, and saving the trained model.
`inference.py`: This script loads the trained model and performs a sample prediction.
`Dockerfile`: Docker configuration file for building the project inside a Docker container.
`.dockerignore`: List of files and directories to be ignored during Docker build process.
`auto-mpg.csv`: The dataset used for training and testing the model.
`fuel_efficiency_model.h5`: The saved trained model file.
`README.md`: Project documentation file.

## Prerequisites

Make sure you have Docker installed on your system.

## Usage with Docker

### 1. Build the Docker Image:
`docker build -t fuel-efficiency-prediction .`

### 2. Run the Docker Container:
`docker run -it --rm fuel-efficiency-prediction`

This will run the training script (`train.py`) inside the Docker container, train the model, and save the trained model as `fuel_efficiency_model.h5`.

### 3. Perform Prediction inside the Docker Container:
`docker exec -it <container_id_or_name> python inference.py`

Replace `<container_id_or_name>` with the actual ID or name of the running Docker container. This command performs a sample prediction using the trained model inside the Docker container.
Make sure to modify the `sample_input` variable in `inference.py` with the desired input features for prediction.

## Dataset

The dataset (`auto-mpg.csv`) contains information about various car models, including features like cylinders, displacement, horsepower, weight, acceleration, model year, and origin. The target variable is miles per gallon (MPG).

## Model Architecture

The model architecture consists of two fully connected layers with batch normalization and dropout to avoid overfitting. The output layer predicts the MPG value.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
