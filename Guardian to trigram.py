
# ______________________________________________________________________________________________________________________________________________________________________
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
'''Pre process the Data'''
# ________________________________________________________________________________________________________________________________________________________
#---------------------------------------------------------------------------------------------------------------------------------------
# Pre-Processing (MAIN FUCNTION)
#---------------------------------------------------------------------------------------------------------------------------------------

import json
import re
import time
from nltk import ngrams, pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Compile the regex once
word_re = re.compile(r'\W+')

# Initialize the stopwords once
stop_words = set(stopwords.words('english'))

# Cache POS tags that are frequently used
keep_tags = {'NN', 'CD', 'VBN', 'VB', 'JJ', 'NNS', 'VBG', 'IN', 'RB', 'JJS'}

def pre_main(json_file_path):
    start_time = time.time()  # Start timer for pre_main

    loaded_data = load_from_json(json_file_path)
    trigrams = {}

    # Flatten all articles from all days into a single list
    all_articles = [article for daily_data in loaded_data for item in daily_data if isinstance(daily_data, list)
                    for article in item.get('response', {}).get('results', [])]

    # Initialize the stemmer only once
    stemmer = PorterStemmer()

    # Process each article sequentially
    for article in all_articles:
        processed_article = process_article(article, stemmer)
        if processed_article:
            publication_date, ngrams_list = processed_article
            if publication_date in trigrams:
                trigrams[publication_date].update(ngrams_list)
            else:
                trigrams[publication_date] = set(ngrams_list)

            # Print statement to indicate completion of processing for the particular date
            print(f"Finished processing articles for date: {publication_date}")

    # Convert sets back to lists and sort trigrams by date
    trigrams = {k: sorted(list(v)) for k, v in trigrams.items()}

    # Save the processed data to a minified JSON file
    output_file_path = json_file_path.replace('.json', '_processed.json')
    with open(output_file_path, 'w') as outfile:
        json.dump(trigrams, outfile, separators=(',', ':'))

    print(f"Processed data saved to {output_file_path}")

    end_time = time.time()  # End timer for pre_main
    print(f"Time taken to run pre_main: {end_time - start_time:.2f} seconds")

def process_article(article, stemmer):
    start_time = time.time()  # Start timer for process_article

    if 'fields' in article and 'bodyText' in article['fields']:
        raw_text = article['fields']['bodyText']

        # Preprocessing the tokens
        preprocessed_tokens = pre_processing(raw_text)

        # Perform stemming on the tokens
        stemmed_tokens = [stemmer.stem(token) for token in preprocessed_tokens]

        # Keeping Adjectives, Nouns & Verbs
        filtered_tokens = filter_pos_tokens(stemmed_tokens)

        # Generating unigrams, bigrams, and trigrams
        ngrams_list = set(generate_ngrams(filtered_tokens, 1))
        ngrams_list.update(generate_ngrams(filtered_tokens, 2))
        ngrams_list.update(generate_ngrams(filtered_tokens, 3))

        # Extracting publication date
        publication_date = article["webPublicationDate"].split('T')[0]

        end_time = time.time()  # End timer for process_article
        print(f"Time taken to process one article: {end_time - start_time:.2f} seconds")

        return publication_date, ngrams_list
    else:
        end_time = time.time()  # End timer for process_article
        print(f"Time taken to process one article: {end_time - start_time:.2f} seconds")
        return None

def load_from_json(json_file_path):
    start_time = time.time()  # Start timer for load_from_json

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    end_time = time.time()  # End timer for load_from_json
    print(f"Time taken to load JSON data: {end_time - start_time:.2f} seconds")
    
    return data

# Function to preprocess text
def pre_processing(text):
    start_time = time.time()  # Start timer for pre_processing

    text = word_re.sub(' ', text)
    tokens = [token.lower() for token in word_tokenize(text) if token.lower() not in stop_words]

    end_time = time.time()  # End timer for pre_processing
    print(f"Time taken to preprocess text: {end_time - start_time:.2f} seconds")

    return tokens

# Function to generate n-grams
def generate_ngrams(tokens, n):
    return [' '.join(grams) for grams in ngrams(tokens, n)]

# Function to filter POS tokens
def filter_pos_tokens(tokens):
    start_time = time.time()  # Start timer for filter_pos_tokens

    tagged_tokens = pos_tag(tokens)
    filtered_tokens = [word for word, tag in tagged_tokens if tag in keep_tags]

    end_time = time.time()  # End timer for filter_pos_tokens
    print(f"Time taken to filter POS tokens: {end_time - start_time:.2f} seconds")

    return filtered_tokens

# _______________________________________________________________________________________________________________________________________________________________
if __name__ == "__main__":
    pre_main('Guardian_full\Guardian_Annual\guardian_articles_00_23_tradedays.json')


# if __name__ == "__main__":
#     pre_main('Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_02_tradedays.json')

# if __name__ == "__main__":
#     pre_main('Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_03_tradedays.json')

# if __name__ == "__main__":
#     pre_main('Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_04_tradedays.json')

# if __name__ == "__main__":
#     pre_main('Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_05_tradedays.json')

# if __name__ == "__main__":
#     pre_main('Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_06_tradedays.json')

# if __name__ == "__main__":
#     pre_main('Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_07_tradedays.json')


# ______________________________________________________________________________________________________________________________________________________________________
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
import json
import os
from wordcloud import WordCloud
from collections import defaultdict

# Load the JSON file with a raw string or double backslashes for the path
with open('Guardian_full\\Guardian_Annual\\guardian_articles_00_10.json', 'r') as file:
    data = json.load(file)

# Create a directory to save word clouds
output_dir = 'Raw clouds'
os.makedirs(output_dir, exist_ok=True)

# Since data is a list, iterate through it
sections = defaultdict(list)

# Efficiently accumulate body texts for each section
for outer_list in data:
    for item in outer_list:
        if isinstance(item, dict) and 'response' in item and 'results' in item['response']:
            for article in item['response']['results']:
                section_id = article.get('sectionId')
                body_text = article.get('fields', {}).get('bodyText', '')
                sections[section_id].append(body_text)

# Join the accumulated lists into single strings for each section
sections = {k: ' '.join(v) for k, v in sections.items()}

# List of specific section IDs to process
section_ids = ['world']#,'money','business', 'uk-news', 'politics']

# Generate and save word clouds for each section without multiprocessing
# for section_id in section_ids:
#     if section_id in sections and sections[section_id]:  # Ensure section exists and has content
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sections[section_id])
        
#         # Save the word cloud to the output directory
#         file_path = os.path.join(output_dir, f'{section_id}_wordcloud.png')
#         wordcloud.to_file(file_path)
#         print(f'Saved word cloud for {section_id} to {file_path}')



'''def fetch_guardian_sections():
    url = 'https://content.guardianapis.com/sections'
    params = {
    'api-key': '53d0470c-7acd-4496-b853-f746153893b4',
    'section': 'technology',
    'page': 1,
    'page-size': 10
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        sections_data = response.json()
        sections = sections_data.get('response', {}).get('results', [])
        return [section['id'] for section in sections]
    else:
        print(f"Error: {response.status_code}")
        return None

# Example usage
api_key = 'your_api_key_here'
sections = fetch_guardian_sections()
if sections:
    print("Available sections:")
    for section_id in sections:
        print(section_id)'''


# Preprocessing text
#%%
# Preprocess for category
import json
import re
import time
from nltk import ngrams, pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from multiprocessing import Pool, cpu_count
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Compile the regex once
word_re = re.compile(r'\W')

# Initialize the stopwords once
stop_words = set(stopwords.words('english'))

def pre_main(json_file_path):
    start_time = time.time()

    loaded_data = load_from_json(json_file_path)
    trigrams = {}

    # Flatten all articles from all days into a single list for multiprocessing
    all_articles = [article for daily_data in loaded_data for item in daily_data if isinstance(daily_data, list)
                    for article in item.get('response', {}).get('results', [])]

    if not all_articles:
        print("No articles found in the dataset.")
        return

    # Ensure at least one process is used
    num_processes = max(1, min(cpu_count(), len(all_articles)))

    # Use multiprocessing to process the articles in parallel
    with Pool(num_processes) as pool:
        batch_size = max(10, len(all_articles) // num_processes)
        processed_articles = pool.map(process_article_batch, [all_articles[i:i + batch_size] for i in range(0, len(all_articles), batch_size)])

    # Organize processed articles back into the trigrams dictionary by section ID
    for batch in processed_articles:
        for processed_article in batch:
            if processed_article:
                section_id, ngrams = processed_article
                if section_id in trigrams:
                    trigrams[section_id] += ngrams
                    trigrams[section_id] = list(set(trigrams[section_id]))
                else:
                    trigrams[section_id] = ngrams

    # Sorting trigrams by section ID
    trigrams = dict(sorted(trigrams.items(), key=lambda x: x[0]))

    # Save the processed data to a minified JSON file
    output_file_path = 'Guardian_full\Guardian_Annual\guardian_articles_00_23_tradedays_categorywise_processed.json'

    with open(output_file_path, 'w') as outfile:
        json.dump(trigrams, outfile, separators=(',', ':'))

    print(f"Processed data saved to {output_file_path}")

def process_article_batch(articles):
    stemmer = PorterStemmer()
    processed_batch = []

    for article in articles:
        if 'fields' in article and 'bodyText' in article['fields']:
            raw_text = article['fields']['bodyText']

            # Preprocessing the tokens
            preprocessed_tokens = pre_processing(raw_text)

            # Perform stemming on the tokens
            stemmed_tokens = [stemmer.stem(token) for token in preprocessed_tokens]

            # Keeping Adjectives, Nouns & Verbs
            filtered_tokens = filter_pos_tokens(stemmed_tokens)

            # Generating unigrams, bigrams, and trigrams
            ngrams_list = generate_ngrams(filtered_tokens, 1)
            ngrams_list += generate_ngrams(filtered_tokens, 2)
            ngrams_list += generate_ngrams(filtered_tokens, 3)

            # Extracting section ID
            section_id = article["sectionId"]
            processed_batch.append((section_id, ngrams_list))
        else:
            processed_batch.append(None)

    return processed_batch

def load_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to preprocess text
def pre_processing(text):
    text = word_re.sub(' ', text)
    text = text.encode("utf-8").decode("unicode-escape")
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    return tokens

# Function to generate n-grams
def generate_ngrams(tokens, n):
    return [' '.join(grams) for grams in ngrams(tokens, n)]

# Function to filter POS tokens
def filter_pos_tokens(tokens):
    tagged_tokens = pos_tag(tokens)
    keep_tags = {'NN', 'CD', 'VBN', 'VB', 'JJ', 'NNS', 'VBG', 'IN', 'RB', 'JJS'}
    return [word for word, tag in tagged_tokens if tag in keep_tags]

# File path to the JSON data
file_path = 'Guardian_full\Guardian_Annual\guardian_articles_00_23_tradedays.json'

# Run your main function with the correct file path
if __name__ == "__main__":
    print('Function started')
    pre_main(file_path)
