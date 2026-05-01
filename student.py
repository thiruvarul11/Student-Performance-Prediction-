# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("student_data.csv")

print("Dataset Preview:")
print(df.head())

# -------------------------------
# Step 2: Data Visualization
# -------------------------------

# Scatter Plot: Study Hours vs Final Score
plt.figure()
plt.scatter(df['study_hours'], df['final_score'])
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")
plt.show()

# Scatter Plot: Attendance vs Final Score
plt.figure()
plt.scatter(df['attendance'], df['final_score'])
plt.xlabel("Attendance")
plt.ylabel("Final Score")
plt.title("Attendance vs Final Score")
plt.show()

# Scatter Plot: Previous Score vs Final Score
plt.figure()
plt.scatter(df['previous_score'], df['final_score'])
plt.xlabel("Previous Score")
plt.ylabel("Final Score")
plt.title("Previous Score vs Final Score")
plt.show()

# Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Step 3: Prepare Data
# -------------------------------
X = df[['study_hours', 'attendance', 'previous_score']]
y = df['final_score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: Train Model
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Prediction & Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nModel Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# Step 6: Predict New Student
# -------------------------------
print("\nEnter Student Details:")
study_hours = float(input("Study Hours: "))
attendance = float(input("Attendance (%): "))
previous_score = float(input("Previous Score: "))

new_data = [[study_hours, attendance, previous_score]]
prediction = model.predict(new_data)

print("\nPredicted Final Score:", prediction[0])