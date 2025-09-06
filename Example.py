import numpy as np
import re
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import urllib.parse
import warnings
warnings.filterwarnings('ignore')

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
    random_words = ["secure", "official", "verification", "account", "signin", "login", "update", "service", "webmail"]

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

# Save URLs to files
def save_url_to_file(url, prediction):
    filename = "phishing_urls.txt" if prediction == 'Phishing' else "legitimate_urls.txt"

    # Check if URL already exists in the file
    try:
        with open(filename, 'r') as file:
            existing_urls = file.read().splitlines()
    except FileNotFoundError:
        existing_urls = []

    # Add URL if it's not already in the file
    if url not in existing_urls:
        with open(filename, 'a') as file:
            file.write(url + '\n')
        print(f"URL automatically saved to {filename}")
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

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
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
    prediction = lgb_classifier.predict(embedding)
    probability = lgb_classifier.predict_proba(embedding)

    # Automatically save the URL to the appropriate file
    pred_label = 'Phishing' if prediction[0] == 1 else 'Legitimate'
    save_url_to_file(url, pred_label)

    return {
        'url': url,
        'processed': processed,
        'prediction': pred_label,
        'confidence': probability[0][prediction[0]]
    }

# Interactive URL testing
def interactive_testing():
    print("\n" + "="*60)
    print("PHISHING URL DETECTOR")
    print("="*60)
    print("Enter URLs to check if they're phishing or legitimate")
    print("URLs will be automatically saved to appropriate files")
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