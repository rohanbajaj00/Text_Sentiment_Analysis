import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
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

def compute_euclidean_distance_for_chunk(chunk, combined_terms_by_category):
    """
    Computes the Euclidean distance for a chunk of documents.
    """
    document_dates = list(chunk.keys())
    document_strings = [' '.join(doc) for doc in chunk.values()]

    distance_scores_by_category = {}

    for category, combined_terms in combined_terms_by_category.items():
        # Vectorize the combined terms and documents using TF-IDF
        vectorizer = TfidfVectorizer().fit(list(combined_terms.values()) + document_strings)
        emotion_vectors = vectorizer.transform(list(combined_terms.values()))
        document_vectors = vectorizer.transform(document_strings)

        # Compute Euclidean distance between the combined emotion vectors and document vectors
        distance_matrix = euclidean_distances(emotion_vectors, document_vectors)

        # Invert the distances (so higher values represent more similarity)
        inverse_distance_matrix = 1 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero

        # Create a dictionary to store the results for this category
        distance_scores = {
            emotion: {
                document_dates[idx]: inverse_distance_matrix[emotion_idx, idx]
                for idx in range(len(document_dates))
            }
            for emotion_idx, emotion in enumerate(combined_terms.keys())
        }
        
        distance_scores_by_category[category] = distance_scores

    return distance_scores_by_category

def parallel_euclidean_distance_calculation(documents, combined_terms_by_category):
    """
    Parallelize the calculation of Euclidean distances.
    """
    # Limit the number of processes to a fraction of the CPU cores
    num_processes = max(1, cpu_count() // 2)

    # Split the documents into chunks for parallel processing
    chunk_size = len(documents) // num_processes + (len(documents) % num_processes > 0)
    chunks = [
        dict(list(documents.items())[i:i + chunk_size])
        for i in range(0, len(documents), chunk_size)
    ]

    with Pool(processes=num_processes) as pool:
        results = pool.map(
            partial(compute_euclidean_distance_for_chunk, combined_terms_by_category=combined_terms_by_category),
            chunks
        )

    # Merge the results
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

def format_euclidean_distance_results(distance_scores_by_category):
    """
    Format the Euclidean distance results into a DataFrame with dates as index and each emotion/term pair as columns.
    
    Parameters:
    distance_scores_by_category (dict): Dictionary with categories as keys, emotions as sub-keys, and another dictionary as value which contains dates and Euclidean distance scores.
    
    Returns:
    pd.DataFrame: DataFrame with dates as index and emotion/term pairs as columns.
    """
    # Flatten the data structure into a list of records for DataFrame creation
    records = []
    for category, emotions in distance_scores_by_category.items():
        for emotion, dates in emotions.items():
            for date, score in dates.items():
                records.append({"Date": date, "Category_Emotion": f"euclid_{category}_{emotion}", "Score": score})
    
    # Create DataFrame from the records
    df = pd.DataFrame(records)
    
    # Pivot the DataFrame to get dates as index and emotion/term pairs as columns
    df_pivot = df.pivot_table(index='Date', columns='Category_Emotion', values='Score', aggfunc='first')

    return df_pivot

def save_df_to_json(df, save_filename):
    """
    Save the DataFrame to a JSON file with dates as index.
    
    Parameters:
    df (pd.DataFrame): DataFrame to be saved.
    filename (str): Name of the file where the DataFrame will be saved.
    """
    df.to_json(save_filename, orient='index', date_format='iso')
    print(f"Results saved to {save_filename}")

def euclidean_distance_daily(trigram_file,saving_file):
    """
    Function to initiate the calculation of daily Euclidean distance and save the results to a JSON file.
    """
    # Load emotion dictionaries from JSON
    emotion_dictionaries = load_json('emotion_dictionaries.json')

    # Load pre-processed trigrams from a JSON file
    trigrams_dict = load_json(trigram_file)

    # Combine emotion terms by category
    combined_terms_by_category = combine_terms_by_category(emotion_dictionaries)

    # Compute Euclidean distance for daily data using multiprocessing
    daily_distances = parallel_euclidean_distance_calculation(trigrams_dict, combined_terms_by_category)

    # Format the results into a DataFrame
    daily_distances_df = format_euclidean_distance_results(daily_distances)

    # Save the DataFrame to a JSON file
    save_df_to_json(daily_distances_df,saving_file)

    return daily_distances_df

if __name__ == '__main__':
    print('Euclidean distance calculation started')
    euclidean_distance_daily(trigram_file = 'Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_01_tradedays_processed.json',saving_file='Similarity Results\Euclid\euclidean_distance_01.json')
    print('Euclidean distance calculation complete')

if __name__ == '__main__':
    print('Euclidean distance calculation started')
    euclidean_distance_daily(trigram_file = 'Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_02_tradedays_processed.json',saving_file='Similarity Results\Euclid\euclidean_distance_02.json')
    print('Euclidean distance calculation complete')

if __name__ == '__main__':
    print('Euclidean distance calculation started')
    euclidean_distance_daily(trigram_file = 'Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_03_tradedays_processed.json',saving_file='Similarity Results\Euclid\euclidean_distance_03.json')
    print('Euclidean distance calculation complete')

if __name__ == '__main__':
    print('Euclidean distance calculation started')
    euclidean_distance_daily(trigram_file = 'Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_04_tradedays_processed.json',saving_file='Similarity Results\Euclid\euclidean_distance_04.json')
    print('Euclidean distance calculation complete')

if __name__ == '__main__':
    print('Euclidean distance calculation started')
    euclidean_distance_daily(trigram_file = 'Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_05_tradedays_processed.json',saving_file='Similarity Results\Euclid\euclidean_distance_05.json')
    print('Euclidean distance calculation complete')
if __name__ == '__main__':
    print('Euclidean distance calculation started')
    euclidean_distance_daily(trigram_file = 'Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_06_tradedays_processed.json',saving_file='Similarity Results\Euclid\euclidean_distance_06.json')
    print('Euclidean distance calculation complete')
if __name__ == '__main__':
    print('Euclidean distance calculation started')
    euclidean_distance_daily(trigram_file = 'Guardian_full\Guardian_Annual\Guardian_for_pre\guardian_articles_07_tradedays_processed.json',saving_file='Similarity Results\Euclid\euclidean_distance_07.json')
    print('Euclidean distance calculation complete')
