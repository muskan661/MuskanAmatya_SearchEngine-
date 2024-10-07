from flask import Flask, render_template, request
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Folder where .txt documents are stored
DATA_DIR = 'data'

# Load and preprocess each document from the data folder
nltk.download('stopwords') # removing words like 'and', 'the'...
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

def load_documents(data_dir):
    documents = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='latin1') as f:
                documents[filename] = preprocess(f.read())
    return documents

# Load all documents from the data folder
documents = load_documents(DATA_DIR)

# Prepare a list of document texts and filenames for TF-IDF
doc_names = list(documents.keys())
doc_texts = list(documents.values())

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(doc_texts)

# Query processing function
def query_processing(query):
    query_processed = preprocess(query)
    query_vec = vectorizer.transform([query_processed])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_docs = cosine_similarities.argsort()[-5:][::-1]  # Top 5 documents
    return [doc_names[i] for i in top_docs]  # Return document names

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        results = query_processing(query)
        result_docs = [(doc, open(os.path.join(DATA_DIR, doc), 'r').read()[:200]) for doc in results]  # Short snippet
        return render_template('results.html', query=query, results=result_docs)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
