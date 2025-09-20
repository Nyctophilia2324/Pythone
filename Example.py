import numpy as np
import re
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import urllib.parse
import csv
import os
from datetime import datetime
import pandas as pd

# Create a synthetic dataset for demonstration
def generate_sample_data(num_samples=100000):
    np.random.seed(42)

    # Legitimate URL patterns
    legitimate_patterns = [
        "https://www.{}.com/{}",
        "https://{}.com/{}",
        "https://www.{}.org/{}",
        "https://{}.edu/{}",
        "https://{}.gov/{}"
    ]

    # Phishing URL patterns
    phishing_patterns = [
        "http://{}-login.{}.com/{}",
        "https://{}.{}.secure-login.com/{}",
        "http://{}.{}.account-verification.com/{}",
        "https://update-{}.{}.com/{}",
        "http://{}.{}.webmail-service.com/{}"
    ]

    # Common domains and paths
    domains = ["google", "facebook", "apple", "microsoft", "amazon", "netflix", "bankofamerica", "wellsfargo", "paypal"]
    paths = ["login", "signin", "account", "verify", "security", "update", "profile", "settings", "auth"]
    random_words = ["secure", "official", "verification", 'account', 'signin', 'login', 'update', 'service', 'webmail']

    urls = []
    labels = []

    # Generate legitimate URLs
    for _ in range(num_samples // 2):
        pattern = np.random.choice(legitimate_patterns)
        domain = np.random.choice(domains)
        path = np.random.choice(paths)
        random_str = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), 5))
        url = pattern.format(domain, path, random_str)
        urls.append(url)
        labels.append(0)  # legitimate

    # Generate phishing URLs
    for _ in range(num_samples // 2):
        pattern = np.random.choice(phishing_patterns)
        domain = np.random.choice(domains)
        random_word1 = np.random.choice(random_words)
        random_word2 = np.random.choice(random_words)
        path = np.random.choice(paths)
        url = pattern.format(random_word1, domain, path)
        urls.append(url)
        labels.append(1)  # phishing

    return urls, labels

# Preprocess URLs
def preprocess_url(url):
    # Decode URL-encoded characters
    try:
        url = urllib.parse.unquote(url)
    except:
        pass

    # Remove protocol and www
    url = re.sub(r'https?://(www\.)?', '', url)

    # Split on special characters and keep tokens
    tokens = re.split(r'[./?=_-]', url)

    # Remove empty tokens and join with spaces
    tokens = [token for token in tokens if token != '']

    return ' '.join(tokens)

# Save URLs to a single CSV file
def save_url_to_csv(url, prediction, confidence):
    filename = "url_classifications.csv"
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    # Get the next ID number
    if file_exists:
        # Read the last row to get the last ID
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) > 1:  # If there are rows besides the header
                last_id = int(rows[-1][0])  # First column is the ID
                next_id = last_id + 1
            else:
                next_id = 1
    else:
        next_id = 1
    
    # Open the file in append mode
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(['ID', 'URL', 'Classification', 'Confidence', 'Date_Added'])
        
        # Check if URL already exists in the file
        if file_exists:
            with open(filename, 'r') as read_file:
                reader = csv.reader(read_file)
                # Skip the header row
                next(reader, None)
                existing_urls = [row[1] for row in reader if row]  # URL is in the second column
        else:
            existing_urls = []
        
        # Add URL if it's not already in the file
        if url not in existing_urls:
            writer.writerow([next_id, url, prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            print(f"URL automatically saved to {filename} with ID {next_id}")
        else:
            print(f"URL already exists in {filename}")

# Generate sample data
print("Generating sample data...")
urls, labels = generate_sample_data(2000)

# Preprocess all URLs
print("Preprocessing URLs...")
processed_urls = [preprocess_url(url) for url in urls]

# Initialize MiniLM model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(processed_urls, show_progress_bar=True)

# Convert embeddings to DataFrame with feature names
feature_names = [f'feature_{i}' for i in range(embeddings.shape[1])]
X = pd.DataFrame(embeddings, columns=feature_names)
y = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train LightGBM classifier
print("Training LightGBM model...")
lgb_classifier = lgb.LGBMClassifier(
    num_leaves=31,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
lgb_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = lgb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# Create a function to predict new URLs
def predict_url(url):
    processed = preprocess_url(url)
    embedding = model.encode([processed])
    
    # Convert to DataFrame with the same feature names
    embedding_df = pd.DataFrame(embedding, columns=feature_names)
    
    prediction = lgb_classifier.predict(embedding_df)
    probability = lgb_classifier.predict_proba(embedding_df)

    # Get prediction details
    pred_label = 'Phishing' if prediction[0] == 1 else 'Legitimate'
    confidence = probability[0][prediction[0]]
    
    # Automatically save the URL to the CSV file
    save_url_to_csv(url, pred_label, confidence)

    return {
        'url': url,
        'processed': processed,
        'prediction': pred_label,
        'confidence': confidence
    }

# Interactive URL testing
def interactive_testing():
    print("\n" + "="*60)
    print("PHISHING URL DETECTOR")
    print("="*60)
    print("Enter URLs to check if they're phishing or legitimate")
    print("URLs will be automatically saved to url_classifications.csv")
    print("Type 'quit' to exit the program")
    print("="*60)

    while True:
        url = input("\nEnter a URL to check: ").strip()

        if url.lower() == 'quit':
            print("Exiting program. Goodbye!")
            break

        if not url:
            print("Please enter a valid URL.")
            continue

        # Add http:// if no protocol is specified
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        try:
            result = predict_url(url)
            print(f"\nURL: {result['url']}")
            print(f"Processed: {result['processed']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")

            if result['prediction'] == 'Phishing':
                print("WARNING: This URL might be phishing!")
            else:
                print("This URL appears to be legitimate.")

        except Exception as e:
            print(f"Error processing URL: {e}")

# Run the interactive testing
interactive_testing()
