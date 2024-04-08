# Import necessary libraries
import pandas as pd
import gzip
import json

# Define a class to read gzipped JSON files in chunks
class ReadJsonGz:
    def __init__(self, path, sizeChunk=10000):
        # Initialize path to the file and size of the chunk to read
        self.path = path
        self.sizeChunk = sizeChunk

    def parsing(self):
        # Open the gzipped JSON file for reading
        with gzip.open(self.path, 'rb') as f:
            chunk = []  # Initialize an empty list to hold data chunks
            for line in f:  # Iterate over each line in the file
                chunk.append(json.loads(line))  # Load the JSON data and append to the chunk list
                if len(chunk) >= self.sizeChunk:  # If the chunk reaches the specified size...
                    yield pd.DataFrame(chunk)  # Convert the chunk to a DataFrame and yield
                    chunk = []  # Reset chunk list for next iteration
            if chunk:  # If there are any remaining data after file is fully read...
                yield pd.DataFrame(chunk)  # Convert the remaining data to a DataFrame and yield

# Define a class to extract metadata for a specific product
class ProdMeta:
    def __init__(self, locMetaData, prodName='Headphones'):
        # Initialize location of metadata file and product name to filter
        self.locMetaData = locMetaData
        self.prodName = prodName
        self.DataFrameProdMet = None

    def metExtraction(self):
        handler = ReadJsonGz(self.locMetaData, sizeChunk=5000)  # Create a ReadJsonGz object with specified chunk size
        metadata = []  # Initialize an empty list to hold metadata
        for chunk in handler.parsing():  # Iterate over each chunk produced by the ReadJsonGz object
            # Filter the chunk for rows where the title contains the product name (case insensitive)
            filterChunk = chunk[chunk['title'].str.contains(self.prodName, case=False, na=False)]
            metadata.append(filterChunk)  # Append the filtered chunk to the metadata list
        # Concatenate all chunks into a single DataFrame and reset the index
        self.DataFrameProdMet = pd.concat(metadata, ignore_index=True)
        self.columnsMet()  # Filter the DataFrame columns based on a predefined list

    def columnsMet(self):
        # Define a list of columns to retain
        colPrint = ['asin', 'title', 'brand', 'feature', 'description',
                    'price', 'imageURL', 'imageURLHighRes', 'also_buy',
                    'also_viewed', 'salesRank', 'categories']
        if self.DataFrameProdMet is not None:  # If the DataFrame is not empty...
            # Filter the columns of the DataFrame based on the predefined list
            sCol = [col for col in colPrint if col in self.DataFrameProdMet.columns]
            self.DataFrameProdMet = self.DataFrameProdMet[sCol]

# Define a class to filter reviews for a specific product and merge with its metadata
class RevProd:
    def __init__(self, locReview, DataFrameProdMet):
        # Initialize location of review file and DataFrame containing product metadata
        self.locReview = locReview
        self.DataFrameProdMet = DataFrameProdMet
        self.DataFrameRev = None

    def filterRevMergeMet(self):
        # Extract the set of product IDs (asin) from the metadata DataFrame
        headphoneIds = set(self.DataFrameProdMet['asin'])
        handler = ReadJsonGz(self.locReview, sizeChunk=5000)  # Create a ReadJsonGz object for the review file
        filteredRev = []  # Initialize an empty list to hold filtered reviews
        for chunk in handler.parsing():  # Iterate over each chunk produced by the ReadJsonGz object
            # Filter the chunk for reviews corresponding to the product IDs
            filteredChunk = chunk[chunk['asin'].isin(headphoneIds)]
            filteredRev.append(filteredChunk)  # Append the filtered chunk to the list
        # Concatenate all chunks into a single DataFrame and reset the index
        self.DataFrameRev = pd.concat(filteredRev, ignore_index=True)
        # Merge the filtered reviews DataFrame with the product metadata DataFrame on product ID (asin)
        self.DataFrameRev = pd.merge(self.DataFrameRev, self.DataFrameProdMet, on='asin', how='left')

    def preprocess(self):
        if self.DataFrameRev is not None:  # If the DataFrame is not empty...
            # Remove duplicate rows based on reviewer ID, product ID, and review time
            self.DataFrameRev.drop_duplicates(subset=['reviewerID', 'asin', 'reviewTime'], inplace=True)
            # Remove rows where the review text is missing
            self.DataFrameRev.dropna(subset=['reviewText'], inplace=True)

# Paths to dataset files
locMetaData = '/content/drive/MyDrive/CSE508_Winter2024_A3_2021046/meta_Electronics.json.gz'
locReview = '/content/drive/MyDrive/CSE508_Winter2024_A3_2021046/Electronics_5.json.gz'

# Execute the process using the refactored classes
preProcMet = ProdMeta(locMetaData)  # Create a ProdMeta object for product metadata extraction
preProcMet.metExtraction()  # Extract product metadata

procRev = RevProd(locReview, preProcMet.DataFrameProdMet)  # Create a RevProd object for reviews processing
procRev.filterRevMergeMet()  # Filter reviews and merge with product metadata
procRev.preprocess()  # Preprocess the merged DataFrame

# Display the structure of the final DataFrame
print(procRev.DataFrameRev.info())
print(procRev.DataFrameRev.head())
print("\n")
# Corrected reference to the total number of reviews
print(f"Total number of rows/reviews for the product: {len(procRev.DataFrameRev)}")