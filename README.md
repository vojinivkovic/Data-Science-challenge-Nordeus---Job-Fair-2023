# Data-Science-challenge-Nordeus---Job-Fair-2023
This project was done as a part Data Science Nordeus challenge. First I addressed processing NaN values in columns and encoding categorical with handmade encoder. 
I replaces column avg_stars_top_14_players with avg_stars_top_3_bench_player to have feature that bring more information. 
Data preprocessing was done using MinMaxScaler regarding the league in which team is competing. All the features were scaled based on values that teams have in the league.
To gather more data I used SMOTE on data made with artificial unbalanced classes. I tried many ML models such as Multinomial Logistic Regression, Decision Tree, Random Forest etc.
Random Forest gave the best results. With GridSearchCV I get best values for hyperparameters. 
