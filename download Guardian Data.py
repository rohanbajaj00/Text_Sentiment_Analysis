
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import math
import yfinance as yf
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
import sys

import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count


from sklearn.decomposition import PCA
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from collections import defaultdict
from datetime import datetime, timedelta
import requests
import json
import re
import time
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from multiprocessing import Pool, Manager
    
import Levenshtein as lev
from Levenshtein import distance
from Levenshtein import ratio

#---------------------------------------------------------------------------------------------------------------------------------------
# Download Guardian Data
#---------------------------------------------------------------------------------------------------------------------------------------

# Function to fetch articles from The Guardian
def fetch_guardian_articles(api_key, date, page=1, page_size=200):
    url = 'https://content.guardianapis.com/search'
    sections = ['world','money', 'business', 'uk-news', 'politics']
    all_articles = []

    for section in sections:
        params = {
            'api-key': api_key,
            'section': section,
            'page': page,
            'page-size': page_size,
            'from-date': date,
            'to-date': date,
            'show-fields': 'all'
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            all_articles.append(data)
        else:
            print(f"Error fetching section {section}: {response.status_code}")
    
    return all_articles


def save_to_json(data, filename):
    """
    Save the fetched data to a JSON file.

    Parameters:
    data (dict): Data to save.
    filename (str): Filename for the JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def clear_file_data(filename):
    """
    Clear the data from the specified JSON file.

    Parameters:
    filename (str): Filename of the JSON file to be cleared.
    """
    if os.path.exists(filename):
        os.remove(filename)

import time
import json
from datetime import datetime, timedelta

def download_articles_over_period(start_date, end_date, api_key, theguardianfile):
    """
    Download articles from The Guardian over a specified period and save to JSON after clearing old data.

    Parameters:
    start_date (str): Start date in YYYY-MM-DD format.
    end_date (str): End date in YYYY-MM-DD format.
    api_key (str): API key for The Guardian API.
    """
    clear_file_data(theguardianfile)  # Clear the existing data from the file before starting new downloads
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    current_date = start

    all_articles = []

    while current_date <= end:
    
        print(f"Fetching articles for {current_date.strftime('%Y-%m-%d')}")
        success = False
        retry_count = 0
        wait_time = 305  # Set a constant wait time of 300 seconds (5 minutes)

        while not success:  # Keep retrying until successful
            try:
                day_articles = fetch_guardian_articles(api_key, current_date.strftime('%Y-%m-%d'))
                if day_articles:
                    all_articles.append(day_articles)
                success = True  # Exit the retry loop if successful
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_count += 1
                    print(f"Error 429: Rate limit exceeded. Retry attempt {retry_count}. "
                        f"Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)  # Wait for a fixed time before retrying
                else:
                    print(f"An HTTP error occurred: {e}. Retrying...")
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Retrying...")
        
        current_date += timedelta(days=1)

# Save all fetched articles to a single file
    save_to_json(all_articles,theguardianfile)


# # Load the data from a JSON file
# def load_from_json(filename='guardian_articles_04_(06).json'):
#     with open(filename, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     return data



#---------------------------------------------------------------------------------------------------------------------------------------
'''combining files'''
#---------------------------------------------------------------------------------------------------------------------------------------
def combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_00_07().json', startname='"guardian_"'):
    # Initialize an empty list to hold all the combined data
    combined_data = []
    
    # Loop through the files in the folder
    for file_name in sorted(os.listdir(folder_path)):
        # Check if the file name matches the pattern 'guardian_articles_*.json'
        if file_name.startswith(startname) and file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            # Open and load the JSON data from the file
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Append the data to the combined_data list
                combined_data.extend(data)
    
    # Save the combined data to a new JSON file
    output_path = os.path.join(folder_path, output_file)
    with open(output_path, 'w') as out_file:
        json.dump(combined_data, out_file, indent=4)
    
    print(f"Combined JSON saved as {output_file}")
#---------------------------------------------------------------------------------------------------------------------------------------

# download_articles_over_period(api_key='61f584e2-2feb-4ef2-9a13-607fe0faf0b4', start_date='2012-01-01', end_date='2012-06-30',theguardianfile = "guardian_articles_12_(06).json") 
# download_articles_over_period(api_key='c0526448-7863-4c39-9a4c-febf7b46301d', start_date='2012-07-01', end_date='2012-12-31',theguardianfile = "guardian_articles_12_(12).json") 
# download_articles_over_period(api_key='d63eb45d-df5a-4dc8-bdcf-8b77dc7e9084', start_date='2013-01-01', end_date='2013-06-30',theguardianfile = "guardian_articles_13_(06).json")
# download_articles_over_period(api_key='c19de6bf-ef77-41ef-b453-44203dd6309a', start_date='2013-07-01', end_date='2013-12-31',theguardianfile = "guardian_articles_13_(12).json")
# download_articles_over_period(api_key='2ac36c9a-ad6d-4813-91ec-eb535dda054e', start_date='2021-01-01', end_date='2021-06-30',theguardianfile = "guardian_articles_21_(06).json")
# download_articles_over_period(api_key='53d0470c-7acd-4496-b853-f746153893b4', start_date='2021-07-01', end_date='2021-12-31',theguardianfile = "guardian_articles_21_(12).json")
# download_articles_over_period(api_key='dfa6b3c8-8479-412b-9ae3-2a9fb7ce6a09', start_date='2022-01-01', end_date='2022-06-30',theguardianfile = "guardian_articles_22_(06).json")
# download_articles_over_period(api_key='bdcb1d7c-f46f-4212-9f55-f1f8d4cc6423', start_date='2022-07-01', end_date='2022-12-31',theguardianfile = "guardian_articles_22_(12).json")
# download_articles_over_period(api_key='760cc717-b9e3-4a78-8219-2fe2d29d845c', start_date='2023-01-01', end_date='2023-06-30',theguardianfile = "guardian_articles_23_(12).json")
# download_articles_over_period(api_key='12076675-b5a2-4049-8b70-cfc9267bce7a', start_date='2023-07-01', end_date='2023-12-31',theguardianfile = "guardian_articles_23_(06).json")
# download_articles_over_period(api_key='ef27b28a-536e-46ac-8408-f00b4f9dd3f5', start_date='2017-07-01', end_date='2017-12-31',theguardianfile = "guardian_articles_17_(12).json")
# download_articles_over_period(api_key='a2d3e255-5735-4c07-912f-e8a45696062b', start_date='2018-01-01', end_date='2018-06-30',theguardianfile = "guardian_articles_18_(06).json")
# download_articles_over_period(api_key='ffca5e7b-005a-4504-b7ad-893354c31767', start_date='2018-07-01', end_date='2018-12-31',theguardianfile = "guardian_articles_18_(12).json")
# download_articles_over_period(api_key='e3370f82-b946-49f6-b05f-9308e4022cfc', start_date='2019-01-01', end_date='2019-06-30',theguardianfile = "guardian_articles_19_(06).json")
# download_articles_over_period(api_key='e35bd7d3-aba7-4f7e-be3c-4102b93ee711', start_date='2019-07-01', end_date='2019-12-31',theguardianfile = "guardian_articles_19_(12).json")
# download_articles_over_period(api_key='dacd98d4-4f83-4098-bb71-3d06fd9044ca', start_date='2020-01-01', end_date='2020-06-30',theguardianfile = "guardian_articles_20_(06).json")
# download_articles_over_period(api_key='3866f6c4-c368-4be7-816a-1c545606314b', start_date='2020-07-01', end_date='2020-12-31',theguardianfile = "guardian_articles_20_(12).json")
# download_articles_over_period(api_key='d38e4a8d-acbc-4df8-b341-6bdb8e727842', start_date='2012-01-01', end_date='2012-06-30',theguardianfile = "guardian_articles_12_(06).json")


# combine_json_files(folder_path='Guardian_full/Guardian_Annual', output_file='guardian_articles_00_20.json', startname='guardian_articles_')
combine_json_files(folder_path='Guardian_full\Guardian_Annual\All', output_file='guardian_articles_00_23.json', startname='guardian_articles_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_10.json', startname='guardian_articles_10_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_11.json', startname='guardian_articles_11_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_12.json', startname='guardian_articles_12_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_13.json', startname='guardian_articles_13_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_14.json', startname='guardian_articles_14_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_15.json', startname='guardian_articles_15_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_16.json', startname='guardian_articles_16_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_17.json', startname='guardian_articles_17_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_18.json', startname='guardian_articles_18_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_19.json', startname='guardian_articles_19_')
# combine_json_files(folder_path='Guardian_full', output_file='guardian_articles_20.json', startname='guardian_articles_20_')

import json
import nltk
from nltk import pos_tag

def find_unique_pos_tags(dictionary_file_path):
    """
    Finds and returns all unique POS tags for words in a dictionary loaded from a JSON file.

    Parameters:
    dictionary_file_path (str): Path to the JSON file containing the dictionary.

    Returns:
    set: A set of unique POS tags found in the dictionary.
    """
    # Load the dictionary from the JSON file
    with open(dictionary_file_path, 'r') as file:
        word_dict = json.load(file)

    unique_pos_tags = set()

    # Iterate over all categories and their general terms
    for category, terms in word_dict.items():
        print(category)
        for term_list in terms.values():
            for term in term_list:
                # Get the POS tag for each term
                pos_tags = pos_tag([term])
                for _, tag in pos_tags:
                    unique_pos_tags.add(tag)

    return unique_pos_tags

# # Example usage
# dictionary_file_path = 'emotion_dictionaries.json'
# unique_pos_tags = find_unique_pos_tags(dictionary_file_path)
# print(f"Unique POS tags found: {unique_pos_tags}")


import json

def load_json(file_path):
    """
    Load data from a JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def print_year_and_month(data):
    """
    Print year and month from each date in the JSON data.
    """
    year_month_set = set()  # Use a set to avoid duplicates
    for date in data.keys():
        year_month = date[:7]  # Extract the year and month (YYYY-MM)
        year_month_set.add(year_month)
    
    # Print each unique year-month
    for year_month in sorted(year_month_set):
        print(year_month)

# if __name__ == "__main__":
#     file_path = 'Guardian_full\Guardian_Annual\guardian_articles_00_20_tradedays_processed.json'  # Replace with your JSON file path
#     data = load_json(file_path)
    
#     print_year_and_month(data)
