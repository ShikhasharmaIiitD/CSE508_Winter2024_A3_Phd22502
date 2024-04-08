pip install beautifulsoup4 unidecode nltk

# Import necessary libraries
import pandas as pd
import re
from bs4 import BeautifulSoup
import unidecode
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings

# Download necessary NLTK datasets for text processing
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Suppress BeautifulSoup warnings that can occur with malformed HTML inputs
warnings.filterwarnings("ignore", category=UserWarning)

# Define a class to process text data
class ProcText:
    def __init__(self, acrDict):
        # Initialize with an acronym dictionary to expand acronyms in text
        self.acrDict = acrDict
        # Lemmatizer to reduce words to their base form
        self.lemt = WordNetLemmatizer()
        # Set of English stopwords to be removed from text
        self.stpWords = set(stopwords.words('english'))

    # a. Removing HTML Tags
    def htmlremove(self, text):
        # Use BeautifulSoup to remove HTML tags from text
        return BeautifulSoup(text, "html.parser").get_text()

    # b. Removing accented characters
    def acCharRemove(self, text):
        # Remove accented characters from text, converting them to their closest ASCII representation
        return unidecode.unidecode(text)

    # c. Expanding acronyms based on provided acronym dictionary
    def expAcr(self, text):
        # Loop over the acronym dictionary to replace acronyms with their full form in text
        for ac, exp in self.acrDict.items():
            text = re.sub(r'\b' + re.escape(ac) + r'\b', exp, text, flags=re.IGNORECASE)
        return text

    # d. Removing Special Characters
    def specialCharRemove(self, text):
        # Remove special characters from text, leaving only alphabets and spaces
        return re.sub(r'[^a-zA-Z\s]', '', text)

    # e. Lemmatization and f. Text Normalizer
    def lemNorm(self, text):
        # Convert text to lowercase, split into words, lemmatize, and remove stopwords
        words = text.lower().split()
        lemWords = [self.lemt.lemmatize(word) for word in words if word not in self.stpWords]
        return ' '.join(lemWords)

    # Overall text preprocessing function
    def preprocessing(self, text):
        # Apply the defined preprocessing steps in order
        text = self.htmlremove(text)
        text = self.acCharRemove(text)
        text = self.expAcr(text)
        text = self.specialCharRemove(text)
        text = self.lemNorm(text)
        return text

# Define a simple acronym dictionary for demonstration
acrDict = {
    'nlp': 'natural language processing',
    'ai': 'artificial intelligence',
}

# Function to apply text preprocessing to a given text using the defined acronym dictionary
def textPreProc(text, acrDict):
    processor = ProcText(acrDict)  # Initialize the text processing class with the acronym dictionary
    return processor.preprocessing(text)  # Apply preprocessing to the given text

# Apply the preprocessing function to each review in the DataFrame and store the results in a new column
procRev.DataFrameRev['processed_reviewText'] = procRev.DataFrameRev['reviewText'].apply(lambda x: textPreProc(x, acrDict))

# Print the original and processed review texts for comparison
print(procRev.DataFrameRev[['reviewText', 'processed_reviewText']].head())