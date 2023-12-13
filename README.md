# NFL Prediction Project

## Description
This project is designed to predict the outcomes of NFL games using various statistical inputs, including team ELO ratings, season information, and game outcomes since 1920. It consists of a Python-based machine learning model built using PyTorch and a user-friendly script for inputting game data to generate predictions. It's nearly impossible to predict outcomes of NFL games, this model uses past data to give a prediction that is accurate as possible. I am not responsible for any losses associated with the use of this model. Bet at your own risk.

## Requirements
The project requires the following libraries:

- Python 3.6 or higher
- PyTorch
- Pandas
- Numpy
- Scikit-learn
- Joblib
- Scipy

## Installation
**Step 1:** Clone the repository to your local machine.

```bash
git clone https://github.com/BigAirDaddy/nflgamepredictor
cd nflgamepredictor
```

**Step 2:** Install the requirements

```bash
pip3 install -r requirements.txt
```
**Step 3:** Run the model

```bash
python3 nfl.py
```
## Project Structure

- predict.py: Contains the neural network model for prediction, loads the trained model, and makes predictions based on user input.
- nfl.py: A user-friendly script for inputting game data, interfacing with predict.py to output game predictions.

## Usage
Run the nfl.py script and follow the prompts to input the game data. Up to date ELO number for each team is required. An easy way to find this information is at https://www.nfeloapp.com/nfl-power-ratings/

## Predictions 
The model will output a number between 0 and 1. The number is the percentage chance of the home team winning. For example, if I set CLE as the home team and the model outputs 0.67, it predicts Cleveland has a 67 percent chance of winning. The away team would have a 37 percent chance of winning. 

