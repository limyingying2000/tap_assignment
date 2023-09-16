import html
import re
import nltk
from nltk.corpus import stopwords
import gensim
from nltk.stem import WordNetLemmatizer

def moderate_text_cleaning(text):
    # Convert text to lowercase
    text = text.lower()
    # decode html
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    
    # # Remove punctuation except question marks and numbers
    text = re.sub(r'[^\w\s?]', ' ', text)
    text = re.sub(r'\d', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords and lemmatize
    stop_words = list(stopwords.words("english"))
    other_stopwords = ['singapore', 'mr', 'government', 'also', 'need', 'even', 'may', 'still']
    stop_words.extend(other_stopwords)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)



def minimal_text_cleaning(text):

    # decode html
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)


def sent_to_words(sentences):
    """
    Convert sentences to lists of words using Gensim's simple_preprocess.
    :param sentences: List of sentences/strings.
    :return: List of lists containing words.
    """
    for sentence in sentences:
        # deacc=True removes punctuations
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)