import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from multiprocessing import Pool, cpu_count
import time

def load_json(file_path):
    """
    Load data from a JSON file.
    """
    start_time = time.time()
    with open(file_path, 'r') as file:
        data = json.load(file)
    end_time = time.time()
    print(f"Time taken to load JSON file: {end_time - start_time:.2f} seconds")
    return data

def combine_terms_by_category(emotion_dict):
    """
    Combine terms into separate strings for each category within each emotion.
    """
    start_time = time.time()
    
    categories = ['general_terms']  # Add other categories as needed
    combined_terms = {category: {} for category in categories}
    
    for emotion, terms in emotion_dict.items():
        for category in categories:
            combined_terms[category][emotion] = ' '.join(terms[category])
    
    end_time = time.time()
    print(f"Time taken to combine terms by category: {end_time - start_time:.2f} seconds")
    return combined_terms

def compute_jaccard_similarity_for_emotion(args):
    """
    Computes the Jaccard similarity for a specific emotion and date.
    """
    start_time = time.time()
    
    emotion, emotion_vector, document_vectors, document_dates = args
    similarity_scores = {
        document_dates[idx]: jaccard_score(emotion_vector, document_vectors[idx], average='macro') 
        for idx in range(document_vectors.shape[0])
    }
    
    end_time = time.time()
    print(f"Time taken to compute Jaccard similarity for emotion {emotion}: {end_time - start_time:.2f} seconds")
    return emotion, similarity_scores

def compute_jaccard_similarity(emotion_dict, documents):
    """
    Computes the Jaccard similarity between emotion terms and documents on a daily basis.
    """
    start_time = time.time()
    
    if isinstance(documents, list):
        raise TypeError("Expected documents to be a dictionary, but got a list. Please check the structure of your data.")

    categories = ['general_terms']  # Add other categories as needed
    document_dates = list(documents.keys())
    document_strings = [' '.join(doc) for doc in documents.values()]

    similarity_scores_by_category = {}

    for category in categories:
        combined_terms = {emotion: ' '.join(terms[category]) for emotion, terms in emotion_dict.items()}
        
        # Vectorize the terms and documents using CountVectorizer
        vectorizer = CountVectorizer().fit(list(combined_terms.values()) + document_strings)
        emotion_vectors = vectorizer.transform(list(combined_terms.values())).toarray()
        document_vectors = vectorizer.transform(document_strings).toarray()

        # Prepare the data for multiprocessing
        tasks = [(emotion, emotion_vectors[emotion_idx], document_vectors, document_dates) 
                 for emotion_idx, emotion in enumerate(combined_terms.keys())]

        # Use all available CPU cores for multiprocessing
        num_processes = cpu_count()  # Use all available cores
        with Pool(num_processes) as pool:
            results = pool.map(compute_jaccard_similarity_for_emotion, tasks)

        similarity_scores = {emotion: scores for emotion, scores in results}
        
        similarity_scores_by_category[category] = similarity_scores
    
    end_time = time.time()
    print(f"Time taken to compute Jaccard similarity: {end_time - start_time:.2f} seconds")
    return similarity_scores_by_category

def combine_jaccard_scores(daily_jaccard_scores):
    """
    Combine daily Jaccard scores into a DataFrame.
    """
    start_time = time.time()
    
    data = {}
    for category, emotions in daily_jaccard_scores.items():
        for emotion, dates in emotions.items():
            for date, score in dates.items():
                if date not in data:
                    data[date] = {}
                data[date][f'jaccard_{category}_{emotion}'] = score  # Add prefix 'jaccard_'
    
    end_time = time.time()
    print(f"Time taken to combine Jaccard scores: {end_time - start_time:.2f} seconds")
    return pd.DataFrame.from_dict(data, orient='index')

def save_jaccard_similarity_to_json(df, filename='jaccard_similarity_results.json'):
    """
    Save the DataFrame to a JSON file with dates as the index.
    """
    start_time = time.time()
    
    df.to_json(filename, orient='index', date_format='iso')
    
    end_time = time.time()
    print(f"Time taken to save Jaccard similarity to JSON: {end_time - start_time:.2f} seconds")

def jaccard_similarity_daily(trigrams_file):
    """
    Computes daily Jaccard similarity, saves the results to a JSON file, and returns the DataFrame.
    """
    start_time = time.time()
    
    # Load pre-processed trigrams from the JSON file
    trigrams_dict = load_json(trigrams_file)

    # Ensure that trigrams_dict is a dictionary
    if isinstance(trigrams_dict, list):
        print("Warning: Expected a dictionary for trigrams_dict. Please check your data format.")
        trigrams_dict = {str(i): doc for i, doc in enumerate(trigrams_dict)}

    # Load emotion dictionaries from a JSON file
    emotion_dictionaries = load_json('emotion_dictionaries.json')

    # Calculate Jaccard similarity scores
    similarity_scores_by_category = compute_jaccard_similarity(emotion_dictionaries, trigrams_dict)

    # Combine the similarity scores into a DataFrame
    jaccard_df = combine_jaccard_scores(similarity_scores_by_category)

    # Save the DataFrame to a JSON file
    save_jaccard_similarity_to_json(jaccard_df)

    end_time = time.time()
    print(f"Total time taken for Jaccard similarity daily computation: {end_time - start_time:.2f} seconds")
    
    return jaccard_df

if __name__ == '__main__':
    print('Jaccard similarity computation started')
    jaccard_similarity_daily(trigrams_file = '.json')
    print('Jaccard similarity computation complete')
