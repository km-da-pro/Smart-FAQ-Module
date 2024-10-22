from flask import Flask, request, jsonify, render_template
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FAQ data
with open('faqs.json', 'r') as file:
    faq_data = json.load(file)

# Flatten FAQ data
faq_entries = [(item['question'], item['answer']) for category in faq_data.values() for item in category]
questions = [question for question, _ in faq_entries]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

def find_most_relevant_faq(user_query):
    user_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vector, faq_vectors).flatten()
    best_match_index = similarities.argmax()
    return faq_entries[best_match_index]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_faq():
    user_query = request.json.get('query')
    question, answer = find_most_relevant_faq(user_query)
    
    response = {
        'question': question,
        'answer': answer
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
