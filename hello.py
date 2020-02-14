from dffml import Features, DefFeature
from dffml.noasync import train, accuracy, predict
from dffml_model_scikit import LinearRegression
from dffml_model_scratch.slr import SLR
from ridge import ridge
model = ridge(
    
    features=Features(
        DefFeature("Years", int, 1),
        #DefFeature("Expertise", int, 1),
        #DefFeature("Trust", float, 1),
    ),
    predict=DefFeature("Salary", int, 1),
)

# Train the model
train(
    model,
    {"Years": 0, "Salary": 10}, #"Expertise": 1, "Trust": 0.1, 
    {"Years": 1, "Salary": 20}, #"Expertise": 3, "Trust": 0.2, 
    {"Years": 2, "Salary": 30}, #"Expertise": 5, "Trust": 0.3, 
    {"Years": 3, "Salary": 40}, #"Expertise": 7, "Trust": 0.4, 
)

# Assess accuracy
print(
    "Accuracy:",
    accuracy(
        model,
        {"Years": 4, "Salary": 50},#"Expertise": 9, "Trust": 0.5,  
        {"Years": 5,  "Salary": 60},#"Expertise": 11, "Trust": 0.6,
),
)

# Make prediction
for i, features, prediction in predict(
    model,
    {"Years": 6}, #"Expertise": 13, "Trust": 0.7
    {"Years": 7}, #"Expertise": 15, "Trust": 0.8
):
    features["Salary"] = prediction["Salary"]["value"]
    print(features)