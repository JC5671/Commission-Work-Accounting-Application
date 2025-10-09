# Research: Finding the best ML model that predicts the same data it was 
# 			trained on. This program will show the statistics and histogram.
#
# Data Preprocessing: 
# 	1. Preprocess the data by taking the log (np.log()) of the output to 
# 		normalize skewness.
# 	2. Remove outliers in the log output data.
# During training:
# 	3. One hot encode the job type.
# 	4. Apply Standard Scaler for the hours worked.
# Finally:
# 	5. Train different models
#   6. Predict with 5-fold cv
#   7. Predict its own data for percent difference
# 	6. Take the exponent (np.exp()) of the predictions to convert output 
# 		back to normal
# 
# Current Winner: Decicison Tree because it has the highest accuracy of 
#                 33.8% (73/216) on the interval -5.0% to 5.0%. Additionally, it 
#                 has with low percent difference average and stdev, and 
#                 reasonable rmse from 5-fold cv.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# For data preprocessing and cross validation
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import root_mean_squared_error
# ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# --------------------------------- Functions ---------------------------------
def train(model, train_x, train_y):
    # Preprocess the features columns (job type, hours worked)
    pipeline_job_type = Pipeline([
        ("Encoder", OneHotEncoder(handle_unknown="ignore")) 
    ])
    pipeline_hours_worked = Pipeline([
        ("Scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("Formatted Job Type", pipeline_job_type, ["Job Type"]),
        ("Formatted Hours Worked", pipeline_hours_worked, ["Hours Worked"])
    ])

    # Create ML model with preprocessor
    model_pipeline = Pipeline([
        ("Preprocessing", preprocessor),
        ("Regressor", model)
    ])
    
    # Train the model
    model_pipeline.fit(train_x, train_y)

    return model_pipeline
    
# --------------------------------- Main Code ---------------------------------
# Read data from csv
df = pd.read_csv(
     "../Commission Log - obfuscated.csv",
     parse_dates = ["Date"],
     converters = {
         "Job Type": lambda x: x.lower().strip(),
         "Hours Worked": lambda x: float(x),
         "Pay": lambda x: float(x.replace("$", "").replace(" ", "").replace(",", ""))
     })

# Format the data by taking the log of pay and remove the outliers
'''
# plot before
plt.figure(figsize = (10, 6))
plt.hist(df["Pay"], bins = 20)
plt.xlabel("Pay")
plt.ylabel("Frequency")
plt.title("Histogram of Pay Before Formatting")
'''

# use log pay to normalize
df["Log Pay"] = np.log(df["Pay"])

# drop outliers
Q1 = df["Log Pay"].quantile(0.25)
Q3 = df["Log Pay"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df["Log Pay"] >= lower_bound) & (df["Log Pay"] <= upper_bound)].copy()
df = df.reset_index()

'''
# plot after
plt.figure(figsize = (10, 6))
plt.hist(df["Log Pay"], bins = 20)
plt.xlabel("Log Pay")
plt.ylabel("Frequency")
plt.title("Histogram of Pay After Formatting")
plt.show()
'''

# Train and test different models
model_list = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Poisson Regression": PoissonRegressor(),
    "Support Vector Regression": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "K Neighbors": KNeighborsRegressor(n_neighbors=5),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
}

for model_name, model in model_list.items():
    # Seperate x and y:
    # The train data is the same as the test data to see the best fit 
    data_x = pd.DataFrame({
        "Job Type": df["Job Type"], 
        "Hours Worked": df["Hours Worked"]
    })
    data_y = df["Log Pay"]
    train_x = data_x
    train_y = data_y
    test_x = train_x
    test_y = train_y

    # train model
    trained_model = train(model, train_x, train_y)

    # perform cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_logged_pred = cross_val_predict(trained_model, data_x, data_y, cv=kf)
    true_preds = np.exp(y_logged_pred)
    true_test_y = np.exp(data_y)
    rmse = root_mean_squared_error(true_test_y, true_preds)

    # calculate percent difference
    preds = trained_model.predict(test_x)
    true_test_y = np.exp(test_y) # taking the exponent since it was logged
    true_preds = np.exp(preds) # taking the exponent since it was logged
    percent_diff = (true_test_y - true_preds) / true_test_y * 100

    print(f"{model_name}:")
    print("------------------------------------------------")
    print(f"5-fold RMSE:                           {np.mean(rmse):.2f}")
    print(f"Percent Difference Mean:               {np.mean(percent_diff):.2f}%")
    print(f"Percent Difference Standard Deviation: {np.std(percent_diff):.2f}%")
    print(f"Percent Difference Min:                {np.min(percent_diff):.2f}%")
    print(f"Percent Difference 25th Percentile:    {np.percentile(percent_diff, 25):.2f}%")
    print(f"Percent Difference Median:             {np.median(percent_diff):.2f}%")
    print(f"Percent Difference 75th Percentile:    {np.percentile(percent_diff, 75):.2f}%")
    print(f"Percent Difference Max:                {np.max(percent_diff):.2f}%\n")

    # create a percent difference plot
    plt.figure(figsize=(10,6))
    plt.hist(percent_diff, bins=[i for i in range(-205, 206, 10)], color='skyblue', edgecolor='black')
    plt.title(f"Histogram of Percent Difference ({model_name})")
    plt.xlim(-200, 200)
    plt.ylim(0, 100)
    plt.xlabel("Percent Difference (%)")
    plt.ylabel("Frequency")
    plt.grid(True)

plt.show()

