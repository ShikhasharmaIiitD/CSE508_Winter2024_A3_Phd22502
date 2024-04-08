# a) Number of Reviews
numRev = len(procRev.DataFrameRev)
print(f"Number of Reviews: {numRev}")

# b) Average Rating Score
rateAvg = procRev.DataFrameRev['overall'].mean()
print(f"Average Rating Score: {rateAvg:.2f}")

# c) Number of Unique Products
uniqProd = procRev.DataFrameRev['asin'].nunique()
print(f"Number of Unique Products: {uniqProd}")

# d) Number of Good Ratings (>= 3)
goodRating = procRev.DataFrameRev[procRev.DataFrameRev['overall'] >= 3].shape[0]
print(f"Number of Good Ratings: {goodRating}")

# e) Number of Bad Ratings (< 3)
badRating = procRev.DataFrameRev[procRev.DataFrameRev['overall'] < 3].shape[0]
print(f"Number of Bad Ratings: {badRating}")

# f) Number of Reviews corresponding to each Rating
revPerRate = procRev.DataFrameRev['overall'].value_counts().sort_index()
print("Number of Reviews corresponding to each Rating:")
print(revPerRate)