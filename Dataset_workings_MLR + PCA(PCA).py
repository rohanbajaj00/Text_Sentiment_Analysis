'''PCA Code'''
import pandas as pd
from sklearn.decomposition import PCA

# Load the dataset
file_path = 'Similarity Results/combined_dataset_00_23.csv'
df = pd.read_csv(file_path)

# Set the date as the index
df['Date'] = pd.to_datetime(df['Unnamed: 0'])
df.set_index('Date', inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

# List of emotions to consider
emotions = [
    'joy', 'trust', 'fear', 'surprise', 'sadness', 
    'anticipation', 'anger', 'disgust'
]

# Initialize a dictionary to store the first principal component for each emotion
first_principal_components = {}

# Perform PCA for each emotion
for emotion in emotions:
    # Filter columns related to the current emotion
    emotion_columns = [col for col in df.columns if emotion in col and 'FTSE_Returns' not in col]
    
    # Subset the dataframe
    emotion_df = df[emotion_columns]
    
    # Perform PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(emotion_df)
    
    # Store the first principal component
    first_principal_components[emotion] = principal_component.flatten()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''Do PCA and save file'''
# Convert the dictionary to a DataFrame for easier viewing
pca_df = pd.DataFrame(first_principal_components, index=df.index)

# Add the FTSE returns to the DataFrame
pca_df['FTSE_Returns'] = df['FTSE_Returns']

# Display the resulting DataFrame
print(pca_df)

# Optionally, save the DataFrame to a CSV file
pca_df.to_csv('Similarity Results/pca_results_00_20.csv')
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
