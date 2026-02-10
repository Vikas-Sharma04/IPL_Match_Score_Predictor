# IPL Score Predictor

This **ML Model** allows users to predict the total runs in an IPL match based on current runs and wickets.

## **Algorithms Used**

The following machine learning models were implemented to predict the total score:

- **Linear Regression**: A basic regression model for understanding the linear relationship between features.  
- **Decision Tree Regressor**: A tree-based model for splitting data based on conditions to predict scores.  
- **Random Forest Regressor**: An ensemble of decision trees to improve accuracy and reduce overfitting.  
- **K-Nearest Neighbor Regressor**: A non-parametric model that predicts based on the average score of nearby data points.

## **Dataset**

The dataset includes over-by-over details of IPL matches from 2008 to 2020, with the following features:

- **mid**: Match ID  
- **date**: Date of the match  
- **venue**: Venue of the match  
- **bat_team**: Batting team  
- **bowl_team**: Bowling team  
- **batsman**: Batsman  
- **bowler**: Bowler  
- **runs**: Runs scored  
- **wickets**: Wickets taken  
- **overs**: Overs bowled  
- **runs_last_5**: Runs scored in the last 5 overs  
- **wickets_last_5**: Wickets lost in the last 5 overs  
- **striker**: Batsman on strike  
- **non-striker**: Batsman at the non-striker end  
- **total**: Total score (target variable)