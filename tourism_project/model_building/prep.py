# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi


# =========================================
# 1. Load Dataset from Hugging Face
#
# Define constants for the dataset and output paths
# =========================================
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/geniusut/tourism-package-prediction-project/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
#print(df.head(5))

# =========================================
# 2. Remove Unnecessary Columns
# =========================================
df.drop(columns=["CustomerID"], inplace=True, errors="ignore")
df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

#print(df.head(5))
#print(df.columns)

# =========================================
# 3. Check Missing Values
# =========================================
print("Missing Values:\n", df.isnull().sum())


# =========================================
# 4. Encode Categorical Variables
# =========================================

# Import LabelEncoder for converting categorical values to numbers
from sklearn.preprocessing import LabelEncoder

# Initialize encoder
le = LabelEncoder()

# Loop through all columns with object (categorical) data type
for col in df.select_dtypes(include="object"):
    # Apply label encoding to convert text categories into numbers
    df[col] = le.fit_transform(df[col])



# Check data types of all columns after encoding
print(df.dtypes)


# =========================================
# 5. Define Target and Features
# =========================================

target_col = "ProdTaken"

X = df.drop(target_col, axis=1)
y = df[target_col]


# =========================================
# 6. Train-Test Split
# =========================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Combine features and target for saving
train_df = X_train.copy()
train_df[target_col] = y_train

test_df = X_test.copy()
test_df[target_col] = y_test


# =========================================
# 7. Save Locally
# =========================================

os.makedirs("data/processed", exist_ok=True)
train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Files saved locally!")


# =========================================
# 8. Upload to Hugging Face (Correct Way)
# =========================================

# Define files to upload (ONLY train & test as per rubric)
files = [
    "data/processed/train.csv",
    "data/processed/test.csv"
]

# Replace with your actual Hugging Face dataset repo
repo_id = "geniusut/tourism-package-prediction-project"
repo_type = "dataset"

# Upload files
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # keeps only filename
        repo_id=repo_id,
        repo_type=repo_type,
    )

print("Train & Test datasets uploaded successfully!")
