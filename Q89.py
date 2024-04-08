import pandas as pd
from sklearn.model_selection import train_test_split


# 8. Categorizing the rating into three classes
def ratingClasses(rating):
    if rating > 3:
        return 'Good'
    elif rating < 3:
        return 'Bad'
    else:
        return 'Average'

# Apply the categorization function to the 'overall' column to create a new 'rating_class' column
procRev.DataFrameRev['rating_class'] = procRev.DataFrameRev['overall'].apply(ratingClasses)

# 9. Taking Review Text as input feature and Rating Class as the target variable
X = procRev.DataFrameRev['reviewText']  # Features from preprocessed review texts
y = procRev.DataFrameRev['rating_class']  # Target variable from categorized ratings

# Splitting the dataset into training and test sets with a ratio of 75:25
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=42)

# Output the sizes of the training and test sets to verify
print(f"Training set size: {len(xTrain)}")
print(f"Test set size: {len(xTest)}")