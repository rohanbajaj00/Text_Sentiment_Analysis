import json
import pandas as pd
import numpy as np
import Levenshtein
from rapidfuzz import process, fuzz
import numpy as np
import time
from rapidfuzz import process

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# ________________________________________________________________________________________

def calculate_rapidfuzz_ratio_matrix_vectorized(terms, phrases):
    """
    Vectorized calculation of fuzz.ratio using RapidFuzz's process.cdist which is optimized for pairwise comparisons.
    """
    # Use RapidFuzz's process.cdist directly with fuzz.ratio as the scorer
    ratio_matrix = np.array(process.cdist(terms, phrases, scorer=fuzz.ratio))
    return ratio_matrix

def process_date(date, phrases, emotion_dictionaries):
    start_time = time.time()  # Start the timer
    
    results = {}
    results[date] = {}
    print(f"Processing date: {date} with {len(phrases)} phrases")  # Debugging line

    for emotion, terms in emotion_dictionaries.items():
        print(f"Calculating ratios for {emotion} with {len(terms)} terms")  # Debugging line

        # Calculate the RapidFuzz ratio matrix for the terms and phrases using vectorized approach
        ratio_matrix = calculate_rapidfuzz_ratio_matrix_vectorized(terms, phrases)
        
        # Calculate the average ratio instead of the maximum ratio
        average_ratio = np.mean(ratio_matrix)
        
        # Ensure that the result is stored in the correct nested structure
        results[date][emotion] = {date: average_ratio}  # Store the result as a dictionary with the date key
        print(f"Average ratio for {emotion}: {average_ratio:.2f}")
    
    end_time = time.time()  # End the timer
    
    elapsed_time = end_time - start_time
    print(f"Time taken to process date {date}: {elapsed_time:.2f} seconds")  # Print the elapsed time
    
    return results


# ____________________________________________________________________________________________



def levenshtein_calculation(date_phrases_dict, emotion_dictionaries):
    results = {}

    for date, phrases in date_phrases_dict.items():
        date_results = process_date(date, phrases, emotion_dictionaries)
        results.update(date_results)

    return results

def compute_daily_ratios_df(ratios_by_category):
    records = [
        {"Date": date, "Emotion_Term": f"{emotion}", "Ratio": score}
        for category, emotions in ratios_by_category.items()
        for emotion, dates in emotions.items()
        for date, score in dates.items()
    ]

    df = pd.DataFrame(records)

    print("DataFrame before pivoting:")
    print(df.head())  # Print the first few rows of the DataFrame

    df_pivot = df.pivot_table(index='Date', columns='Emotion_Term', values='Ratio', aggfunc='mean')

    return df_pivot

def add_prefix_to_columns(df, prefix):
    return df.rename(columns=lambda x: f"{prefix}{x}" if x != 'Date' else x)

def save_to_json(df, filename):
    df.reset_index(inplace=True)
    df.set_index('Date', inplace=True)
    df.to_json(filename, orient='index', date_format='iso')

def lev_results_daily(trigrams_file_path, emotions_file_path, output_file_path):
    trigrams_dict = load_json(trigrams_file_path)
    print('Trigrams loaded')

    emotion_dictionaries = load_json(emotions_file_path)
    print('Emotion dictionaries loaded')

    

    ratios_by_category = levenshtein_calculation(trigrams_dict, emotion_dictionaries)
    print('Levenshtein ratios calculated')

    daily_ratios_df = compute_daily_ratios_df(ratios_by_category)
    print('Daily ratios computed')

    daily_ratios_df = add_prefix_to_columns(daily_ratios_df, 'lev_')

    save_to_json(daily_ratios_df, output_file_path)
    print(f'Results saved to {output_file_path}')

    return daily_ratios_df

# Example usage:
if __name__ == "__main__":
    TRIGRAMS_FILE_PATH = 'Guardian_full\Guardian_Annual\guardian_articles_00_20_tradedays_processed.json'
    EMOTIONS_FILE_PATH = 'emotion_dictionaries_lev.json'
    OUTPUT_FILE_PATH = 'Similarity Results\Lev\levenshtein_results_(test).json'

    print('Levenshtein calculation started')
    lev_results_daily(TRIGRAMS_FILE_PATH, EMOTIONS_FILE_PATH, OUTPUT_FILE_PATH)
    print('Levenshtein calculation complete')



'''
import Levenshtein
import json

def load_json(file_path):
    """
    Load data from a JSON file.
    
    Parameters:
    file_path (str): The path to the JSON file.
    
    Returns:
    dict: The loaded JSON data as a Python dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_levenshtein_ratio(str1, str2):
    """
    Calculate the Levenshtein ratio between two strings.
    
    Parameters:
    str1 (str): The first string.
    str2 (str): The second string.
    
    Returns:
    float: The Levenshtein ratio, a value between 0 and 1.
    """
    return Levenshtein.ratio(str1, str2)

def calculate_ratios(emotion_dict, date_phrases_dict):
    """
    Calculate the Levenshtein ratios for all emotion terms against phrases in the date dictionary.
    
    Parameters:
    emotion_dict (dict): Dictionary where keys are emotions and values are lists of terms.
    date_phrases_dict (dict): Dictionary where keys are dates and values are lists of phrases.
    
    Returns:
    dict: Dictionary with dates as keys and sub-dictionaries as values, where each sub-dictionary
          contains the emotion and its associated Levenshtein ratio.
    """
    results = {}

    for date, phrases in date_phrases_dict.items():
        results[date] = {}
        for emotion, terms in emotion_dict.items():
            max_ratio = 0  # Initialize the maximum ratio found for this emotion and date
            for term in terms:
                for phrase in phrases:
                    ratio = calculate_levenshtein_ratio(term, phrase)
                    if ratio > max_ratio:
                        max_ratio = ratio
            results[date][emotion] = max_ratio

    return results

def lev_results_daily(emotion_file_path, date_phrases_file_path, output_file_path):
    """
    Calculates Levenshtein ratios and saves the result to a JSON file.
    
    Parameters:
    emotion_file_path (str): Path to the JSON file containing the emotion dictionary.
    date_phrases_file_path (str): Path to the JSON file containing the date phrases dictionary.
    output_file_path (str): Path to save the output JSON file with the results.
    """
    # Load the emotion dictionary from a JSON file
    emotion_dict = load_json(emotion_file_path)
    print('Emotion dictionary loaded')

    # Load the date phrases dictionary from a JSON file
    date_phrases_dict = load_json(date_phrases_file_path)
    print('Date phrases dictionary loaded')

    # Calculate Levenshtein ratios
    ratios_by_date = calculate_ratios(emotion_dict, date_phrases_dict)
    print('Levenshtein ratios calculated')

    # Save the results to a JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(ratios_by_date, output_file, indent=4)
    print(f'Results saved to {output_file_path}')

# Example usage:
if __name__ == "__main__":
    EMOTION_FILE_PATH = 'emotion_dictionaries_lev.json'
    DATE_PHRASES_FILE_PATH = 'guardian_articles_(test)_tradedays_processed.json'
    OUTPUT_FILE_PATH = 'levenshtein_results___.json'

    print('Levenshtein calculation started')
    lev_results_daily(EMOTION_FILE_PATH, DATE_PHRASES_FILE_PATH, OUTPUT_FILE_PATH)
    print('Levenshtein calculation complete')'''




'''def transform_data(input_file_path, output_file_path):
    """
    Transform the data structure from a nested dictionary with 'general_terms' key
    to a simpler dictionary where each emotion has a list of terms.
    
    Parameters:
    input_file_path (str): The path to the input JSON file.
    output_file_path (str): The path to the output JSON file.
    """
    # Load the data from the input JSON file
    with open(input_file_path, 'r') as infile:
        data = json.load(infile)
    
    # Initialize an empty dictionary to store the transformed data
    transformed_data = {}

    # Iterate over the emotions in the original data
    for emotion, categories in data.items():
        # Assuming all terms are under the 'general_terms' category
        transformed_data[emotion] = categories.get('general_terms', [])

    # Save the transformed data to the output JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)
    
    print(f"Data has been transformed and saved to {output_file_path}")

# Example usage:
if __name__ == "__main__":
    INPUT_FILE_PATH = 'emotion_dictionaries.json'
    OUTPUT_FILE_PATH = 'emotion_dictionaries_lev.json'

    transform_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
'''
