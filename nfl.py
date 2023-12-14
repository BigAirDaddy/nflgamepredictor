import pandas as pd
import predict  
from colorama import Fore, Style


print(Fore.GREEN,"####################################")
print(Style.RESET_ALL,Fore.RED,Style.BRIGHT,"Welcome to the NFL Game Predictor",Style.RESET_ALL)
print(Fore.GREEN,"####################################")
print(Style.RESET_ALL,"I am not responsible for any losses associated with these predictions. Use at your own risk.\n")



# calculate ELO probability
def calculate_elo_prob1(elo1, elo2):
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))


def get_user_input():
    team1 = input("Enter the home team using three letters (e.g., CLE): ").upper()
    team2 = input("Enter the away team using three letters (e.g., HOU): ").upper()
    playoff = int(input('Playoff game? Enter 1 for yes 0 for No: ' ) )
    neutral = int(input("Nuetral site? Enter 1 for yes 0 for No: "))
    elo1 = float(input("Enter the ELO for the home team: "))
    elo2 = float(input("Enter the ELO for the away team: "))
    season = int(input("Enter the season (e.g., 2023): "))
    date = input("Enter the date in the correct format (e.g., 9/7/2023): ")
    elo_prob1 = calculate_elo_prob1(elo1, elo2)

    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        'date': [date],
        'season': [season],
        'neutral':[neutral],
        'playoff':[playoff],
        'team1': [team1],
        'team2': [team2],
        'elo1': [elo1],
        'elo2': [elo2],
        'elo_prob1': [elo_prob1]
    })
    
    
    # input_data.to_csv('user_input.csv', index=False)
    
    return input_data


user_input_df = get_user_input()


prediction = predict.make_prediction(user_input_df)

# Display the prediction
print("Outcome > 0.5 Home team is favored. Outcome < 0.5 Away team is favored")
print(f"The predicted outcome is: {prediction}")
