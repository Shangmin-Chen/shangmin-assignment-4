from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Flask app
app = Flask(__name__)

# Fetch the entire 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')

# Step 1: Preprocessing and Creating Term-Document Matrix
# Initialize TfidfVectorizer with stopwords and max features for better performance
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)

# Convert documents to a TF-IDF term-document matrix
X_tfidf = vectorizer.fit_transform(newsgroups.data)

# Step 2: Applying SVD for Dimensionality Reduction (LSA)
n_components = 100  # Number of components for dimensionality reduction
svd = TruncatedSVD(n_components=n_components)
X_reduced = svd.fit_transform(X_tfidf)

# Step 3: Query handling and Cosine Similarity Calculation
def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Preprocess the query using the same TF-IDF vectorizer
    query_tfidf = vectorizer.transform([query])
    
    # Reduce query to LSA space using the same SVD transformation
    query_reduced = svd.transform(query_tfidf)
    
    # Compute cosine similarities between the query and all documents
    cosine_similarities = cosine_similarity(query_reduced, X_reduced).flatten()
    
    # Sort the similarities and get the indices of the top 5 most similar documents
    top_indices = np.argsort(cosine_similarities)[-5:][::-1]
    
    # Retrieve the top 5 documents and their similarities
    top_documents = [newsgroups.data[i] for i in top_indices]
    top_similarities = cosine_similarities[top_indices]
    
    return top_documents, top_similarities, top_indices.tolist()

# Step 4: Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    
    # Convert similarities to a list (if it's still a NumPy array)
    similarities = similarities.tolist() if isinstance(similarities, np.ndarray) else similarities
    
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices})

# Step 5: Run the app
if __name__ == '__main__':
    app.run(debug=True)
