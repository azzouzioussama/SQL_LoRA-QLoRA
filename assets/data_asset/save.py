import os
from dotenv import load_dotenv
from ibm_cloud_sdk_core import IAMTokenManager
from ibm_watson_studio_lib import access_project_or_space
import pandas as pd
import os
import json
from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.utils._errors import RepositoryNotFoundError, EntryNotFoundError


def get_credits():
    wslib = access_project_or_space({
            'token': 'p-2+M7N412nNMq+LSbsjANoOoQ==;LFeF3V6i+F/jEnezq8oOQA==:l0bUpeHOW5rp8xq20UiCjJQyak+tK37f7uTyFZsV7YvvbFmQbYhtaO3KgtiCa1qahvIu57LYjESD5n0TXPH5u0ZHGef4njBD5A==',
            'project_id': 'bdd13a82-ee92-406c-bc3d-fc0690f7cb1e'
    })
    wslib.download_file('config.env')
    load_dotenv('config.env')

    # Connection variables
    api_key = os.getenv("API_KEY", None)
    ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None) 
    project_id = os.getenv("PROJECT_ID", None)
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }
    access_token = IAMTokenManager(
        apikey = api_key,
        url = "https://iam.cloud.ibm.com/identity/token"
    ).get_token()

    return wslib, access_token

def save_model_name(model_name, increment_version=False):
    '''# Save the model name to a csv file with two other coloumns: model_name and model_version that will be used to track the model and can be used to increment the version of the model when a new model is trained (ex: model_version = model_version + 1)
    #and also version have tmbers that increments by 1 (ex: from 0.0,0.1,0.2..1.0....10.0)'''
    wslib, access_token = get_credits()
    # Define the file name
    file_name = 'model_tracking.csv'

    try:
        wslib.download_file(file_name)
    except:
        with open(file_name, 'w') as f:
            f.write('model_name,model_version\n')

    # Define the file name
    file_name = 'model_tracking.csv'
    
    # Check if the file exists
    if os.path.exists(file_name):
        # Load the existing CSV file
        df = pd.read_csv(file_name)
        
        # Check if the model name already exists
        if model_name in df['model_name'].values:
            # Get the current version of the model
            current_version = df.loc[df['model_name'] == model_name, 'model_version'].max()
            if increment_version:
                # Increment the version by 0.1
                new_version = round(current_version + 0.1, 1)
            else:
                # Keep the current version
                new_version = current_version
                model_name_version = model_name + '_sql-v' + str(new_version)
                return model_name_version, df
        else:
            # Start the versioning for the new model name at 0.0
            new_version = 0.0
    else:
        # If the file does not exist, start a new DataFrame
        df = pd.DataFrame(columns=['model_name', 'model_version'])
        new_version = 0.0
    
    # Create a new row as a DataFrame
    new_row = pd.DataFrame({'model_name': [model_name], 'model_version': [new_version]})
    
    # Concatenate the new row to the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the DataFrame back to the CSV file
    df.to_csv(file_name, index=False)
    
    print(f"Model '{model_name}' saved with version {new_version}")
    
    # Return the model name with version
    model_name_version = model_name + '_sql-v' + str(new_version)
    
    # Upload the CSV file to your project as a data asset
    try:
        wslib.upload_file(
            file_path=file_name,
            asset_name=file_name,  # You can customize the name if needed
            overwrite=True  # Overwrite existing file with the same name (optional)
        )
        print(f"Successfully uploaded {file_name} to your project!")
    except Exception as e:
        print(f"Failed to upload {file_name} to your project: {e}")
    
    return model_name_version, df


def get_model_name(file_name='model_tracking.csv'):
    '''Get the model name and model version from the CSV file'''
    try:
        df = pd.read_csv(file_name)
        model_name = df['model_name'].iloc[0]
        model_version = df['model_version'].iloc[0]
        return model_name, float(model_version)
    except FileNotFoundError:
        print(f"{file_name} not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading {file_name}: {e}")
        return None, None

def replace_model_name(model_name, model_version, file_name='model_tracking.csv'):
    '''Replace the model name and model version in the CSV file'''
    try:
        df = pd.DataFrame({'model_name': [model_name], 'model_version': [model_version]})
        df.to_csv(file_name, index=False)
        return model_name, model_version
    except Exception as e:
        print(f"An error occurred while writing to {file_name}: {e}")
        return None, None

def get_csv_file_as_dataframe(file_name = 'model_tracking.csv'):
    '''Read the CSV file and return it as a DataFrame'''
    try:
        return pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"{file_name} not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading {file_name}: {e}")
        return pd.DataFrame()

# Example usage:
# model_name, model_version = get_model_name()
# replace_model_name('new_model', 1.0)
# df = get_csv_file_as_dataframe('model.csv')

import os
import pandas as pd
from huggingface_hub import hf_hub_download, upload_file
from datasets import Dataset
from huggingface_hub import HfApi, HfFolder, Repository

# Function to upload the DataFrame to Hugging Face
def upload_dataframe_to_huggingface(df, repo_id='koukoudzz/gpt2_sql-v0.0', path_in_repo='', local_csv_path='training_results.csv'):
    try:
        # Initialize the Hugging Face API
        api = HfApi()

        # Set the full path in the repository
        path_in_repo = os.path.join(path_in_repo, local_csv_path)
        
        # Download the existing CSV from Hugging Face if it exists
        try:
            downloaded_file = hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type="model")
            existing_df = pd.read_csv(downloaded_file)
            print("Existing file found and loaded from Hugging Face Hub.")
        except (FileNotFoundError, pd.errors.EmptyDataError, Exception) as e:
            existing_df = pd.DataFrame()  # If no existing file, create a fresh DataFrame
            print(f"No existing file found, creating a new one. Error: {e}")

        # Check if the model already exists in the existing dataframe
        if not existing_df.empty and df["Model Name"].iloc[0] in existing_df["Model Name"].values:
            model_name = df["Model Name"].iloc[0]
            idx = existing_df.index[existing_df["Model Name"] == model_name].tolist()[0]

            # Update the corresponding row with the new metrics
            for col in df.columns:
                if col in existing_df.columns:
                    existing_df.at[idx, col] = df[col].iloc[0]
                else:
                    # If the column does not exist, add the new metric as a column
                    existing_df[col] = None
                    existing_df.at[idx, col] = df[col].iloc[0]
        else:
            # Append the new row if the model does not exist
            existing_df = pd.concat([existing_df, df], ignore_index=True)

        # Save the DataFrame to a CSV file locally
        existing_df.to_csv(local_csv_path, index=False)
    
        # Upload the file to the Hugging Face Hub
        upload_file(
            path_or_fileobj=local_csv_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model"  # Ensure you're using the correct repo type
        )
        
        print(f"Successfully uploaded {local_csv_path} to Hugging Face Hub under the model repository {repo_id}!")
        return existing_df
    except Exception as e:
        print(f"An error occurred while uploading the dataset to Hugging Face: {e}")

# Example usage:
# df = pd.DataFrame({'model_name': ['my_model'], 'model_version': [1.0]})
# upload_dataframe_to_huggingface(df, 'my_dataset', 'your_username/my_dataset_repo', 'your_huggingface_token')
