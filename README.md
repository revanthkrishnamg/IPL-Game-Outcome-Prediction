# Predictive Modeling of IPL Match Outcomes Using Machine Learning

# Abstract
This study uses machine learning to predict outcomes of IPL cricket matches by analyzing detailed statistics on batting, bowling, and fielding, along with player form and consistency. We evaluated algorithms like Decision Tree, Random Forest, LightGBM, and XGBoost using data from Kaggle and ESPN Cricinfo. Results showed that XGBoost and LightGBM provided the best accuracy, with Random Forest also performing well. Using a combination of these models, we achieved an accuracy of 71.58%. Our research emphasizes the importance of including detailed fielding data and thorough feature selection to improve prediction accuracy, enhancing strategic decision-making in IPL cricket.

# Introduction
Machine learning has revolutionized the prediction of cricket match outcomes, providing more precise insights that aid strategic decision-making. The increasing popularity of short cricket formats like ODIs and T20s has driven the demand for accurate predictions. Unlike traditional methods that rely on historical data and expert opinions, our approach uses comprehensive player statistics and machine learning to analyze various performance metrics. Our research includes detailed fielding statistics, which have often been overlooked but can significantly impact match outcomes, especially in shorter formats. We aim to develop a robust predictive model that integrates batting, bowling, and fielding data along with player form and consistency, to offer a comprehensive tool for improving strategies in the competitive IPL environment.

# Methodology Overview

## Data Collection
We collected datasets from Kaggle and ESPN Cricinfo, which include detailed ball-by-ball IPL data, match details, and player statistics for batsmen, bowlers, and fielders. These datasets were organized into pandas DataFrames for further processing.

## Data Preprocessing
### Initial Cleanup
The datasets were already clean, requiring minimal initial preprocessing.

### Calculating Player Attributes
We computed various traditional and derived player attributes:
- **Batting**: Includes basic attributes like runs, average, and strike rate. We also derived additional metrics to assess batting performance more comprehensively.
- **Bowling**: Includes basic attributes like wickets and economy rate, and derived metrics to evaluate effectiveness.
- **Fielding**: We calculated traditional fielding attributes such as catches and stumpings.

### Advanced Metrics Using AHP
Using the Analytic Hierarchy Process (AHP), we derived weights for more nuanced attributes like player form and consistency, based on traditional metrics. This process included pairwise comparisons to determine the relative importance of each attribute.

![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/0aff2c84-4fc7-4bec-8760-c0d044ebc09d)
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/94bacc57-46f2-4a90-bb33-3c584dfc8225)
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/06a16b46-4ab3-422f-a410-b083db3ad60d)
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/b8501cdb-bb95-4f1b-b216-61a833e97e74)

### Data structure with the Form and Consistency Attributes for Batting and Bowling:
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/3c40e2d6-c8a8-4c54-955c-67eb30b0affb)
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/66258031-fb62-4df1-8f11-16b248572e74)


### Calculating the Weights for Fielding Metrics using AHP
For fielding metrics, the weighting process reflects the relative importance of each type of fielding action catches, stumpings, and run outs alongside the number of innings a player has fielded in. The Analytic Hierarchy Process (AHP) is used to determine these weights. AHP helps in dealing with complex decision-making and enables the evaluation of elements by breaking them down into a series of pairwise comparisons (Saaty et al. 2012). Here’s how the weights for fielding metrics were derived using AHP as per the below table.
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/38ea5b05-eefe-4ac9-b882-b7552e0b12a8)
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/86a19fd6-b4f2-4d84-86ea-a65e1777deeb)
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/aa499b97-0daf-4bde-a505-f257e89f1743)

### Data structure with the Form and Consistency Attributes for Fielding:
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/0e36d7f3-719a-4d39-abb8-ea37210f360e)

### Integrating IPL and International Data
We combined IPL statistics with international performance data to provide a comprehensive view of each player's capabilities, adjusting the weights to favor more relevant IPL performances.

![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/b4113e69-c94f-4f18-b21b-1f0bc647e275)

### Feature Engineering
We developed new features that capture the impact of top performers in each match, identifying and representing top batsmen, bowlers, and fielders based on their form and consistency scores.
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/8bc7201c-aae3-4a2b-8e06-5cb63606db01)

### Preparing Final Dataset
The dataset was finalized by encoding categorical variables like team names and venues, removing irrelevant columns, and optimizing for machine learning models. The final dataset was saved in a structured format, ready for model building.

### Balancing the Dataset
We checked the balance of our target variable (wins by Team 1) and found a slight imbalance (480 wins vs 470 losses). This small difference is unlikely to significantly affect the accuracy of our predictive models.
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/4610459d-20c7-4086-b1ce-7f80cbb28a5a)

## Model Building and Evaluation

### Model Selection
We used various statistical models, each chosen for its ability to handle the complexities of cricket data. The models included Decision Trees, Random Forest, LightGBM, and XGBoost, among others.
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/133e1b67-40b8-4110-9e32-43dfc485480b)

### Initial Model Results
Our initial testing showed that LightGBM provided the highest test accuracy of about 71%. Other models like Decision Trees, Random Forest, and XGBoost also demonstrated strong fits to the training data but varied in their test performance due to issues such as overfitting.
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/1503bd32-93cd-41fc-855d-eec2137892b8)

### Hyperparameter Tuning
We performed extensive hyperparameter tuning using GridSearchCV, which improved the accuracy, precision, recall, and F1 score of our models, particularly enhancing the performance of Random Forest and XGBoost.
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/79d2b48b-27ac-4151-a295-1322ec73765d)

### Final Model Evaluation
After optimizing the models, we conducted further testing to confirm their practical applicability. XGBoost emerged as the best-performing model, with robust accuracy and F1 scores, indicating its effectiveness in handling diverse and complex datasets.
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/2bad1e97-315b-4265-bfee-9d678cc6443b)
![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/36d278d8-87df-4225-9e7b-32adbeeec5f2)

### Majority Voting Classifier
We implemented a majority voting classifier combining predictions from LightGBM, Random Forest, and XGBoost, achieving an accuracy of 71.58%. This ensemble approach leveraged the strengths of individual models to enhance overall prediction accuracy.

![image](https://github.com/revanthkrishnamg/IPL-Game-Outcome-Prediction/assets/149286080/2aab39ae-56ab-4deb-b06a-989c3f553a6f)

## Implications and Future Directions

### Study Implications
Our study underscores the importance of incorporating fielding statistics into predictive models for IPL matches, demonstrating that fielding plays a crucial role in match outcomes. The use of ensemble methods and advanced feature engineering techniques also proved effective in refining our predictions.

### Future Work
Future research will explore integrating real-time data, applying models to other cricket formats, and updating player form data annually. We also plan to investigate the impact of external factors like weather and pitch conditions on match outcomes.

## Team
Revanth Krishna
Shashanka JJ
Manoj
