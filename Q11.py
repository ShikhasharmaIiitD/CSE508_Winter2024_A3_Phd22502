import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class CollaborativeFiltering:
    def __init__(self, df):
        # Initialize with a DataFrame and prepare scaled matrices
        self.df = df
        self.matScUI = None  # Scaled User-Item matrix
        self.matScIU = None  # Scaled Item-User matrix
        self.dataNorm()
        
    def dataNorm(self):
        # Normalize the data: Fill missing values with 0, scale between 0 and 1
        matUI = self.df.pivot_table(index='reviewerID', columns='asin', values='overall').fillna(0)
        scaler = MinMaxScaler()
        self.matScUI = pd.DataFrame(scaler.fit_transform(matUI), index=matUI.index, columns=matUI.columns)
        self.matScIU = self.matScUI.T  # Transpose for item-user matrix
    
    @staticmethod
    def calcCosSim(matrix):
        # Calculate cosine similarity matrix
        return cosine_similarity(matrix)
    
    @staticmethod
    def nearestN(matSim, top_n):
        # Find indices of top N nearest neighbors based on similarity matrix
        indcN = np.argsort(-matSim, axis=1)[:, :top_n]
        return indcN

    @staticmethod
    def calcMAE(actual, predicted):
        # Calculate Mean Absolute Error, considering only non-zero entries (i.e., actual ratings)
        indcNZero = np.nonzero(actual)
        actNZero = actual[indcNZero]
        predNZero = predicted[indcNZero]
        return mean_absolute_error(actNZero, predNZero)

    def predRating(self, scMat, indcN, matSim):
        # Predict ratings for every user-item combination
        predictions = np.zeros(scMat.shape)
        nUsers, nItems = scMat.shape

        for nIindxUsr in range(nUsers):
            for indxNbr in indcN[nIindxUsr]:
                # Accumulate weighted ratings from similar users/items
                predictions[nIindxUsr, :] += matSim[nIindxUsr, indxNbr] * scMat.iloc[indxNbr, :]

            sumSim = np.sum(np.abs(matSim[nIindxUsr, indcN[nIindxUsr]]))
            if sumSim > 0:
                # Normalize by sum of similarities to prevent inflation
                predictions[nIindxUsr] /= sumSim

        return predictions

    def colabFiltering(self, scMat, nNList=[10, 20, 30, 40, 50]):
        # Perform collaborative filtering and evaluate using K-Fold Cross Validation
        avgMVal = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for nNbr in nNList:
            valMAE = []

            for indxTrain, indxTest in kf.split(scMat):
                # Split the data into train and test sets for cross-validation
                matTrain, matTest = scMat.iloc[indxTrain], scMat.iloc[indxTest]
                matSim = self.calcCosSim(matTrain.to_numpy())
                indcN = self.nearestN(matSim, nNbr)

                # Predict and calculate MAE
                predMat = self.predRating(matTrain, indcN, matSim)
                actMat = matTest.to_numpy()
                scMAE = self.calcMAE(actMat, predMat)

                valMAE.append(scMAE)

            avgMVal[nNbr] = np.mean(valMAE)  # Average MAE for current neighbor count

        return avgMVal


# Generate a sample DataFrame with random data
np.random.seed(42)
revDF = pd.DataFrame({
    'reviewerID': np.random.randint(1, 100, 1000),
    'asin': np.random.randint(1, 20, 1000),
    'overall': np.random.randint(1, 6, 1000)
})

cf = CollaborativeFiltering(revDF)  # Initialize collaborative filtering with the DataFrame

# Perform and print MAE values for User-User and Item-Item Collaborative Filtering
uuMval = cf.colabFiltering(cf.matScUI)
print("User-User Collaborative Filtering MAE values:", uuMval)

print("\n")

iiMVal = cf.colabFiltering(cf.matScIU)
print("Item-Item Collaborative Filtering MAE values:", iiMVal)
print("\n")

# Plot the MAE values for a visual comparison
plt.figure(figsize=(10, 5))
plt.plot(list(uuMval.keys()), list(uuMval.values()), label='User-User CF', marker='o')
plt.plot(list(iiMVal.keys()), list(iiMVal.values()), label='Item-Item CF', marker='x')
plt.xlabel('Number of Nearest Neighbors')
plt.ylabel('Mean Absolute Error')
plt.title('Collaborative Filtering MAE Comparison')
plt.legend()
plt.grid(True)
plt.show()