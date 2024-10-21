import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score, confusion_matrix
import joblib
from matplotlib import cm

# Step 1: Data Processing

#Converts into dataframe
csvfilepath = 'Project_1_Data.csv'
df = pd.read_csv(csvfilepath)
print(df.head())
print("End of Step 1")

# Step 2: Data Visualization

column_class = 'Step'
feat1 = 'X'
feat2 = 'Y'
feat3 = 'Z'

norm = plt.Normalize(df[column_class].min(), df[column_class].max())  
cmap = cm.viridis  
colors = cmap(norm(df[column_class]))

# Creates a 3D scatter plot with a color bar
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(df[feat1], df[feat2], df[feat3], c=df[column_class], cmap=cmap, alpha=0.6)

# Axis labels and title
ax.set_xlabel(feat1)
ax.set_ylabel(feat2)
ax.set_zlabel(feat3)
ax.set_title(f'3D Scatter Plot of {feat1}, {feat2}, and {feat3} by {column_class}')

#  Color bar 
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=5)
cbar.set_label(f'{column_class} (Step)')
print("End of Step 2")


#Step 3: Correlation Analysis
target = 'Step'
correlation_matrix = df. corr(method='pearson')

#Heatmap Graph
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap= 'coolwarm', square=True, cbar_kws={'shrink': .8})
plt.title('Correlation Heatmap')
print("Correlation with the target variable:")
print(correlation_matrix[target])
#loop function to interpret the correlations
for feature in correlation_matrix[target].index:
    correlation = correlation_matrix[target][feature]
    if feature != target:
        if correlation > 0:
            print(f"The correlation between {feature} and {target} is {correlation: .2f}, indicating a positive correlation.")
        elif correlation < 0:
            print(f"The correlation between {feature} and {target} is {correlation:2f} indicating a negative correlation.")
        else:
            print(f"There is no correlation between {feature} and {target}. ")

print("End of Step 3")

#Step 4: Classification Model
X_var = df.drop(columns=['Step'])
y_var = df['Step']
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_var, y_var, test_size=0.2, random_state=42)
print("End of Step 4")


# Logistic Regression 
param_grid_logreg = [
    {'penalty': ['l1'], 'C': [0.1, 1, 10, 100], 'solver': ['saga'], 'max_iter': [5000]},
    {'penalty': ['l2'], 'C': [0.1, 1, 10, 100], 'solver': ['saga'], 'max_iter': [5000]},
    {'penalty': ['elasticnet'], 'C': [0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5, 0.9], 'solver': ['saga'], 'max_iter': [5000]}
]
#Performs grid search to find best parameter
logreg_model = LogisticRegression(solver='saga', max_iter=5000)
logreg_grid_search = GridSearchCV(logreg_model, param_grid_logreg, cv=5, verbose=1, n_jobs=-1)
logreg_grid_search.fit(X_train_data, y_train_data)
print("Best Logistic Regression Parameters:", logreg_grid_search.best_params_)
best_logreg_model = logreg_grid_search.best_estimator_
y_pred_logreg = best_logreg_model.predict(X_test_data)
print("Logistic Regression Accuracy:", accuracy_score(y_test_data, y_pred_logreg))
print(classification_report(y_test_data, y_pred_logreg))

#Random Forest

#Defines trees, depth, and samples
param_grid_rfmodel = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_model = RandomForestClassifier()
rf_grid_search = GridSearchCV(rf_model, param_grid_rfmodel, cv=5, verbose=1, n_jobs=-1)
rf_grid_search.fit(X_train_data, y_train_data)
print("Best Random Forest Parameters:", rf_grid_search.best_params_)
best_rf_model = rf_grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_data)
print("Random Forest Accuracy:", accuracy_score(y_test_data, y_pred_rf))
print(classification_report(y_test_data, y_pred_rf))

#Support Vector

#Define Paramters such as regularization, type of kernal and coefficient
param_grid_svcmodel = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
svc_model = SVC()
svc_grid_search = GridSearchCV(svc_model, param_grid_svcmodel, cv=5, verbose=1, n_jobs=-1)
svc_grid_search.fit(X_train_data, y_train_data)
print("Best SVM Parameters:", svc_grid_search.best_params_)
best_svc_model = svc_grid_search.best_estimator_
y_pred_svc = best_svc_model.predict(X_test_data)
print("SVM Accuracy:", accuracy_score(y_test_data, y_pred_svc))
print(classification_report(y_test_data, y_pred_svc))

# Gradient Boosting
# Defines stages, contribution, depth, and samples 
param_dist_gradboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9]
}
gradboost_model = GradientBoostingClassifier()
gradboost_random_search = RandomizedSearchCV(gradboost_model, param_dist_gradboost, n_iter=10, cv=5, verbose=1, n_jobs=-1, random_state=42)
gradboost_random_search.fit(X_train_data, y_train_data)
print("Best Gradient Boosting Parameters:", gradboost_random_search.best_params_)
best_gradboost_model = gradboost_random_search.best_estimator_
y_pred_gradboost = best_gradboost_model.predict(X_test_data)
print("Gradient Boosting Accuracy:", accuracy_score(y_test_data, y_pred_gradboost))
print(classification_report(y_test_data, y_pred_gradboost))
print("End of Step 4")

#Step 5: Model Performance Analysis
#Dictionary
models = {
    'Logistic Regression': y_pred_logreg,
    'Random Forest': y_pred_rf,
    'SVM': y_pred_svc,
    'Gradient Boosting': y_pred_gradboost
}
#Loops through each model
for model_name, y_pred in models.items():
    print(f"\n--- {model_name} Confusion Matrix ---")
    cm = confusion_matrix(y_test_data, y_pred)
    plt.figure(figsize=(8, 6))
    #Displays heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_logreg_model.classes_, yticklabels=best_logreg_model.classes_)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Step')
    plt.ylabel('True Step')
    plt.show()
plt.show()
print("End of Step 5")

#Step 6: Stacked Model Performance Analysis
#Define Estimators
estimators = [
    ('logreg', best_logreg_model),
    ('rf', best_rf_model)
]

#Create classifier and fit onto the data
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_clf.fit(X_train_data, y_train_data)

# Predict and evaluate the stacked model
y_pred_stack = stacking_clf.predict(X_test_data)

# Performance Metrics
accuracy_stack = accuracy_score(y_test_data, y_pred_stack)
precision_stack = precision_score(y_test_data, y_pred_stack, average='weighted')
f1_stack = f1_score(y_test_data, y_pred_stack, average='weighted')

#Prints the model performance
print("\n--- Stacking Model Performance ---")
print(f"Stacking Accuracy: {accuracy_stack:.4f}")
print(f"Stacking Precision: {precision_stack:.4f}")
print(f"Stacking F1 Score: {f1_stack:.4f}")
print(classification_report(y_test_data, y_pred_stack))

# Confusion Matrix for Stacking Model
cm_stack = confusion_matrix(y_test_data, y_pred_stack)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Blues', xticklabels=best_logreg_model.classes_, yticklabels=best_logreg_model.classes_)
plt.title('Stacking Classifier Confusion Matrix')
plt.xlabel('Predicted Step')
plt.ylabel('True Step')
plt.show()

# Discussion on Stacking Model Effectiveness
if accuracy_stack > max(accuracy_score(y_test_data, y_pred) for y_pred in models.values()):
    print("The stacking model shows a significant increase in accuracy, indicating that combining the strengths of the Logistic Regression and Random Forest models leads to improved performance.")
else:
    print("The stacking model did not show a significant increase in accuracy, which may suggest that the models do not provide complementary strengths in this particular dataset.")

print("End of Step 6")
# Step 7: Model Evaluation
stacking_model_filename = 'stacking_model.joblib'
joblib.dump(stacking_clf, stacking_model_filename)
print(f"Stacked model saved to {stacking_model_filename}")
loaded_stacking_model = joblib.load(stacking_model_filename)
maintenance_coordinates = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]
#Prints the predicted maintenance steps
maintenance_coordinates_df = pd.DataFrame(maintenance_coordinates, columns=X_train_data.columns)
predicted_steps = loaded_stacking_model.predict(maintenance_coordinates_df)
print("Predicted Maintenance Steps for the given coordinates:", predicted_steps)
print("End of Step 7")