import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 1. Load Loan Approval Dataset
data = pd.read_csv("loan_approval.csv")

print("\nDataset Loaded Successfully")
print("Dataset Shape:", data.shape)


# 2. Separate features and target
target_column = "Loan_Status"

data[target_column] = (
    data[target_column]
    .astype(str)
    .str.strip()
    .map({"Y": 1, "N": 0})
)

data = data.dropna(subset=[target_column])

X = data.drop(columns=[target_column])
y = data[target_column]


# Remove Loan_ID
if "Loan_ID" in X.columns:
    X = X.drop(columns=["Loan_ID"])


# 3. Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


# 4. Numerical preprocessing
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# 5. Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# 6. Combine preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# 7. Create Pipeline with preprocessing and Decision Tree
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ]
)


# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 9. Train the Pipeline
pipeline.fit(X_train, y_train)


# 10. Predict test data
y_pred = pipeline.predict(X_test)


# 11. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


# 12. Display result
print("\n================================")
print("Scikit-learn Pipeline Result")
print("================================")
print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))
print("Accuracy        :", round(accuracy, 4))
print("Accuracy (%)    :", round(accuracy * 100, 2), "%")

print("\nPipeline Steps:")
print(pipeline)