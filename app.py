import requests
from flask import Flask, render_template, request, g
import spacy
from sentence_transformers import SentenceTransformer, util
import boto3
import os
from dotenv import load_dotenv
import time

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set up API keys and URLs
# GOOGLE_TRANSLATE_API_URL = ""
# GOOGLE_API_KEY = 
# AWS_ACCESS_KEY = 
# AWS_SECRET_KEY = 
# AWS_REGION = 

# Initialize models for similarity and embeddings
nlp = spacy.load("en_core_web_md")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize metrics variables
api_call_count = 0
total_similarity_score = 0
similarity_score_count = 0

# Tourist attractions descriptions in Vellore, Tamil Nadu
descriptions = {
    "vellore_fort": "The Vellore Fort is a 16th-century granite fort, one of the most prominent historical landmarks of Vellore. Built by the Vijayanagar kings, this fort is renowned for its robust architecture, massive walls, and intricate design. The fort houses a temple dedicated to Lord Shiva, a mosque, and a Christian church within its premises, symbolizing the region's religious diversity.",
    "golden_temple": "The Sripuram Golden Temple, located at the foothills of the Vedagiri hills in Vellore, is one of the most famous spiritual destinations in South India. This temple is renowned for its golden-covered sanctum, which is made of pure gold and is said to be one of the largest in the world. The temple is dedicated to Goddess Lakshmi, the goddess of wealth, and is surrounded by beautiful gardens, walkways, and peaceful landscapes.",
    "amirthi_zoological_park": "Amirthi Zoological Park, situated at the base of the Amirthi Hills, is a popular destination for nature lovers and wildlife enthusiasts. The park is known for its diverse collection of flora and fauna, set amidst lush green surroundings and picturesque waterfalls. It is home to several species of animals such as tigers, leopards, monkeys, and various birds. The park offers a variety of activities like nature walks, boating, and trekking to its visitors.",
    "yelagiri_hills": "Yelagiri Hills, a serene hill station nestled in the Eastern Ghats, is a hidden gem offering spectacular views and a perfect getaway for nature lovers. The hill station is known for its lush greenery, pleasant climate, and trekking opportunities. Visitors can enjoy a variety of outdoor activities such as trekking, paragliding, and rock climbing, or simply relax and soak in the tranquility of the surroundings.",
}

# Theater listing data
theaters = [
    {
        'name': 'Inox Selvam',
        'distance': '~2.0 km',
        'rating': '4.1',
        'image': '/static/images/theater1.jpg',
        'hours': '9AM to 2AM',
        'facilities': ['Snacks', 'Restaurant', 'Washroom'],
        'languages': ['English', 'Hindi', 'Telugu', 'Tamil'],
        'pricing': [
            {'type': 'Standard', 'price': '200 ₹'},
            {'type': 'Premium', 'price': '350 ₹'}
        ]
    },
    {
        'name': 'Galaxy Cinemas',
        'distance': '~6.0 km',
        'rating': '4.1',
        'image': '/static/images/theater2.jpg',
        'hours': '9AM to 2AM',
        'facilities': ['Snacks', 'Restaurant', 'Washroom'],
        'languages': ['English', 'Hindi', 'Telugu', 'Tamil'],
        'pricing': [
            {'type': 'Standard', 'price': '200 ₹'},
            {'type': 'Premium', 'price': '350 ₹'}
        ]
    }
]

# Emergency data for hospitals and police contacts
emergency_data = {
    "hospitals": [
        {
            "name": "Christian Medical College (CMC) Hospital",
            "location": "IDA Scudder Rd, Vellore, Tamil Nadu 632004",
            "phone": "+91 416 228 1000",
            "emergency": "+91 416 228 2111"
        },
        {
            "name": "Government Vellore Medical College Hospital",
            "location": "Adukkamparai, Vellore, Tamil Nadu 632011",
            "phone": "+91 416 226 1900",
            "emergency": "+91 416 226 1000"
        },
        {
            "name": "Sri Narayani Hospital & Research Centre",
            "location": "Sripuram, Thirumalaikodi, Vellore, Tamil Nadu 632055",
            "phone": "+91 416 220 6300"
        },
        {
            "name": "Apollo KH Hospital",
            "location": "25 Arcot Rd, Thottapalayam, Vellore, Tamil Nadu 632004",
            "phone": "+91 416 222 2100"
        },
        {
            "name": "Vasan Eye Care Hospital",
            "location": "No. 1, Anna Salai, Opposite Collector Office, Vellore, Tamil Nadu 632001",
            "phone": "+91 416 222 3333"
        }
    ],
    "police": [
        {
            "name": "Vellore District Police Headquarters",
            "phone": "+91 416 225 2242",
            "location": "Katpadi Rd, Gandhi Nagar, Vellore, Tamil Nadu 632006"
        },
        {
            "name": "Vellore Town Police Station",
            "phone": "+91 416 222 1000",
            "location": "Long Bazaar Rd, Thottapalayam, Vellore, Tamil Nadu 632004"
        },
        {
            "name": "Katpadi Police Station",
            "phone": "+91 416 229 1000",
            "location": "NH234, Katpadi, Vellore, Tamil Nadu 632007"
        }
    ],
    "additional_contacts": {
        "ambulance": "108",
        "fire": "101"
    }
}

# Function to translate text using AWS Translate
def translate_with_aws(texts, target_language):
    global api_call_count
    client = boto3.client('translate',
                          aws_access_key_id=AWS_ACCESS_KEY,
                          aws_secret_access_key=AWS_SECRET_KEY,
                          region_name=AWS_REGION)
    translated_texts = []
    api_call_count += 1
    for text in texts:
        response = client.translate_text(
            Text=text,
            SourceLanguageCode='en',
            TargetLanguageCode=target_language
        )
        translated_texts.append(response['TranslatedText'])
    return translated_texts

# Function to calculate similarity scores
def calculate_similarity_scores(original_texts, aws_translations):
    global total_similarity_score, similarity_score_count
    scores = []
    for i, original_text in enumerate(original_texts):
        doc_original = nlp(original_text)
        doc_aws = nlp(aws_translations[i])

        # Spacy similarity
        spacy_aws_score = doc_original.similarity(doc_aws)

        # Cosine similarity using embeddings
        emb_original = embedder.encode(original_text, convert_to_tensor=True)
        emb_aws = embedder.encode(aws_translations[i], convert_to_tensor=True)
        cosine_aws_score = util.pytorch_cos_sim(emb_original, emb_aws).item()

        aws_total = (spacy_aws_score + cosine_aws_score) / 2
        scores.append(aws_total)

        # Update total similarity score and count
        total_similarity_score += aws_total
        similarity_score_count += 1

    return scores

# Function to calculate average similarity score
def get_average_similarity_score():
    if similarity_score_count == 0:
        return 0
    return total_similarity_score / similarity_score_count

# Before each request, record the start time
@app.before_request
def before_request():
    g.start_time = time.time()

# After each request, calculate the response time
@app.after_request
def after_request(response):
    response_time = time.time() - g.start_time
    print(f"Response time: {response_time:.4f} seconds")
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_language = 'en'
    n_descriptions = list(descriptions.values())

    if request.method == 'POST':
        selected_language = request.form['language']
        aws_translations = translate_with_aws(n_descriptions, selected_language)
        similarity_scores = calculate_similarity_scores(n_descriptions, aws_translations)

        translations = {place: aws_translations[i] for i, place in enumerate(descriptions)}
    else:
        translations = descriptions

    print(f"API calls made: {api_call_count}")
    print(f"Average similarity score: {get_average_similarity_score():.4f}")

    return render_template('index.html', descriptions=translations, selected_language=selected_language)

@app.route('/entertainment', methods=['GET', 'POST'])
def entertainment():
    selected_language = request.form.get('language', 'en')
    translated_theaters = theaters.copy()

    n_theater_fields = []
    for theater in translated_theaters:
        n_theater_fields.extend([
            theater['name'],
            theater['distance'],
            theater['rating'],
            theater['hours'],
            ', '.join(theater['facilities']),
            ', '.join(theater['languages']),
            ' | '.join([f"{price['type']} - {price['price']}" for price in theater['pricing']])
        ])

    if request.method == 'POST' and selected_language != 'en':
        aws_translations = translate_with_aws(n_theater_fields, selected_language)

        index = 0
        for theater in translated_theaters:
            theater['name'] = aws_translations[index]
            theater['distance'] = aws_translations[index + 1]
            theater['rating'] = aws_translations[index + 2]
            theater['hours'] = aws_translations[index + 3]
            theater['facilities'] = aws_translations[index + 4].split(', ')
            theater['languages'] = aws_translations[index + 5].split(', ')
            pricing_translated = aws_translations[index + 6].split(' | ')
            theater['pricing'] = [{'type': p.split(' - ')[0], 'price': p.split(' - ')[1]} for p in pricing_translated]
            index += 7
    elif selected_language == 'en':
        translated_theaters = theaters

    print(f"API calls made: {api_call_count}")
    print(f"Average similarity score: {get_average_similarity_score():.4f}")

    return render_template('entertainment.html', theaters=translated_theaters, selected_language=selected_language)

@app.route('/emergency')
def emergency():
    return render_template('emergency.html', emergency_data=emergency_data)

if __name__ == '__main__':
    app.run(debug=True)