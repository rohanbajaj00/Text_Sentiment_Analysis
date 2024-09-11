import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed



# Function to check if a date is a Saturday or Sunday
def is_weekend(date_str):
    try:
        date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError as e:
        print(f"Error parsing date: {date_str}. Skipping this article. Error: {e}")
        return False
    return date.weekday() >= 5  # Saturday is 5, Sunday is 6

# Function to find the next Monday after a given date
def next_monday(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
    days_ahead = 7 - date.weekday() if date.weekday() != 0 else 0
    next_monday_date = date + timedelta(days=days_ahead)
    return next_monday_date.strftime('%Y-%m-%dT%H:%M:%SZ')

# Function to process each article
def process_article(article):
    publication_date = article.get('webPublicationDate', '')
    if is_weekend(publication_date):
        article['webPublicationDate'] = next_monday(publication_date)

    # Merge headline into bodyText
    headline = article.get('fields', {}).get('headline', '')
    body_text = article.get('fields', {}).get('bodyText', '')
    if headline:
        article['fields']['bodyText'] = headline + " " + body_text

    return article

def main():
    # Load the JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Collect all articles to process
    all_articles = [
        article 
        for response_list in data 
        for response in response_list 
        for article in response.get('response', {}).get('results', [])
    ]

    # Use ThreadPoolExecutor to process the articles in parallel
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_article, article): article for article in all_articles}

        # Replace articles with their processed versions
        for future in as_completed(futures):
            future.result()

    # Save the new data to a file with a "_tradedays" suffix
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Processed file saved as: {output_file_path}")

# file_path = 'Guardian_full\Guardian_Annual\guardian_articles_05.json'
# output_file_path = file_path.replace('.json', '_tradedays.json')
# if __name__ == "__main__":
#     main()

# file_path = 'Guardian_full\Guardian_Annual\guardian_articles_06.json'
# output_file_path = file_path.replace('.json', '_tradedays.json')
# if __name__ == "__main__":
#     main()

file_path = 'Guardian_full\Guardian_Annual\guardian_articles_00_23.json'
output_file_path = file_path.replace('.json', '_tradedays.json')
if __name__ == "__main__":
    main()


