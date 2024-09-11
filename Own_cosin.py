import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
from functools import partial

def load_json(file_path):
    """
    Load data from a JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def combine_terms_by_category(emotion_dict):
    """
    Combine terms into separate strings for each category within each emotion.
    """
    categories = ['general_terms']  # Add other categories if needed
    combined_terms = {category: {} for category in categories}
    
    for emotion, terms in emotion_dict.items():
        for category in categories:
            combined_terms[category][emotion] = ' '.join(terms[category])
    
    return combined_terms

def compute_cosine_similarity_for_chunk(chunk, combined_terms_by_category):
    """
    Compute cosine similarity for a chunk of documents.
    """
    document_dates = list(chunk.keys())
    document_strings = [' '.join(doc) for doc in chunk.values()]

    similarity_scores_by_category = {}

    for category, combined_terms in combined_terms_by_category.items():
        # Vectorize the combined terms and documents using TF-IDF
        vectorizer = TfidfVectorizer().fit(list(combined_terms.values()) + document_strings)
        emotion_vectors = vectorizer.transform(list(combined_terms.values()))
        document_vectors = vectorizer.transform(document_strings)

        # Compute cosine similarity between the combined emotion vectors and document vectors
        similarity_matrix = cosine_similarity(emotion_vectors, document_vectors)

        # Convert cosine similarity values to angles in degrees
        angle_matrix = np.degrees(np.arccos(np.clip(similarity_matrix, -1.0, 1.0)))

        # Create a dictionary to store the results for this category
        similarity_scores = {
            emotion: {
                document_dates[idx]: angle_matrix[emotion_idx, idx]
                for idx in range(len(document_dates))
            }
            for emotion_idx, emotion in enumerate(combined_terms.keys())
        }
        
        similarity_scores_by_category[category] = similarity_scores

    return similarity_scores_by_category

def parallel_cosine_similarity_computation(documents, combined_terms_by_category):
    """
    Parallelize the computation of cosine similarity.
    """
    # Determine the number of processes to use (avoid overloading the CPU)
    num_processes = min(cpu_count() - 1, 4)

    # Split the documents dictionary into chunks for parallel processing
    chunk_size = len(documents) // num_processes + (len(documents) % num_processes > 0)
    chunks = [
        dict(list(documents.items())[i:i + chunk_size])
        for i in range(0, len(documents), chunk_size)
    ]

    # Use multiprocessing to compute cosine similarity in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(
            partial(compute_cosine_similarity_for_chunk, combined_terms_by_category=combined_terms_by_category),
            chunks
        )

    # Merge the results from all chunks
    merged_results = {}
    for result in results:
        for category, emotions in result.items():
            if category not in merged_results:
                merged_results[category] = {}
            for emotion, dates in emotions.items():
                if emotion not in merged_results[category]:
                    merged_results[category][emotion] = {}
                merged_results[category][emotion].update(dates)

    return merged_results

def add_prefix_to_columns(df, prefix):
    """
    Adds a prefix to all column names in a DataFrame except for the 'Date' column.
    """
    return df.rename(columns=lambda x: f"{prefix}{x}" if x != 'Date' else x)

def cosine_theta_results_daily(trigram_file,save_file):
    """
    Compute cosine theta angle similarity for daily data, store the results in a JSON file with prefixed columns.
    
    Returns:
    pd.DataFrame: DataFrame containing daily cosine theta angle similarities indexed by date and with columns for each emotion and term.
    """
    # Load the emotion dictionaries from the JSON file
    emotion_dictionaries = load_json('emotion_dictionaries.json')

    # Load the pre-processed trigrams from a JSON file
    daily_data = load_json(trigram_file)

    # Combine terms by category
    combined_terms_by_category = combine_terms_by_category(emotion_dictionaries)

    # Compute cosine theta angle similarity in parallel
    daily_similarities = parallel_cosine_similarity_computation(daily_data, combined_terms_by_category)

    # Transform the results into a DataFrame
    data = {}
    
    for category, emotions in daily_similarities.items():
        for emotion, date_values in emotions.items():
            for date, value in date_values.items():
                if date not in data:
                    data[date] = {}
                data[date][f"{category}_{emotion}"] = value

    df = pd.DataFrame.from_dict(data, orient='index')

    # Sorting the DataFrame by date index
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Add the 'cosine_' prefix to all columns
    df = add_prefix_to_columns(df, 'cosine_')

    # Save the DataFrame to a JSON file
    df.to_json(save_file, orient='index', date_format='iso')

    print("Cosine Theta Daily Similarities DataFrame saved to 'cosine_theta_results.json'.")
    print(df)  # Debugging statement

    return df

if __name__ == '__main__':
    print('Cosine theta computation started')
    cosine_theta_results_daily(trigram_file='Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_01_tradedays_processed.json',save_file='Similarity Results\Cosin\cosine_theta_results_01.json')
    print('Cosine theta computation complete')
