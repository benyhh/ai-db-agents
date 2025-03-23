import pandas as pd
import json
import openai
from dotenv import load_dotenv
import time
import os
from openai import OpenAI
from pyprojroot import here

load_dotenv()



def get_last_finished_batch(batches):
    """
    Finds the highest batch number that has completed processing.
    
    Args:
        batches: Iterable object with batches from the OpenAI API
        
    Returns:
        int: The highest completed batch number
    """
    last_completed_batch = 0
    for batch in batches:
        description = batch.metadata["description"]
        if "movies batch" in description:
            batch_number = int(description.split()[2])
            if batch.status == "completed" and batch_number > last_completed_batch:
                last_completed_batch = batch_number
                
    return last_completed_batch


def batch_in_progress(batches):
    """
    Checks if any batch is currently in progress.
    
    Args:
        batches: Iterable object with batches from the OpenAI API
        
    Returns:
        bool: True if any batch is in progress, False otherwise
    """
    for batch in batches:
        if batch.status == "in_progress":
            return True
    return False

def create_embedding_request(batch, client):
    """
    Creates a new batch embedding request for a specific batch number.
    
    Args:
        batch: Batch number to process
        client: OpenAI client instance
        
    Returns:
        Batch status object from the OpenAI API
    """
    file_response = openai.files.create(file=open(here(f"data/movie_batches/movie_batch_{batch}.jsonl"), "rb"), purpose="batch")
    batch_input_file_id = file_response.id
    
    batch_response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={
            "description": f"movies batch {batch} of 15"
        }
    )
    
    batch_status = client.batches.retrieve(batch_response.id)
    print(f"Batch {batch} created.")
    return batch_status

def get_batch_progress(batches):
    """
    Returns the progress of an in-progress batch.
    
    Args:
        batches: List of batch objects from the OpenAI API
        
    Returns:
        tuple: (number of completed requests, total number of requests)
    """
    for batch in batches:
        if batch.status == "in_progress":
            batch_number = int(batch.metadata["description"].split()[2])
            return batch_number, batch.request_counts.completed, batch.request_counts.total


def get_saved_embedding_ids():
    """
    Retrieves the IDs of embeddings that have already been saved.
    
    Returns:
        list: List of saved embedding batch IDs
    """
    path_embeddings = here('data/embeddings/')
    embeddings_files = os.listdir(path_embeddings)
    file_ids = [int(file.split("_")[-1].split(".")[0]) for file in embeddings_files]
    return file_ids

def fetch_completed_unsaved_filenames(batches):
    """
    Finds completed batches that haven't been saved yet.
    
    Args:
        batches: List of batch objects from the OpenAI API
        
    Returns:
        list: List of tuples containing (output_file_id, batch_number) for unsaved completed batches
    """
    saved_embedding_ids = get_saved_embedding_ids()
    unsaved_files = []
    for batch in batches:
        description = batch.metadata["description"]
        if "movies batch" in description:
            batch_number = int(description.split()[2])
            if batch.status == "completed" and batch_number not in saved_embedding_ids:
                unsaved_files.append((batch.output_file_id, batch_number))
                
    return unsaved_files


def get_file_contents(filename, client):
    """
    Retrieves and parses embedding results from a file.
    
    Args:
        filename: ID of the file to retrieve
        client: OpenAI client instance
        
    Returns:
        pandas.DataFrame: DataFrame containing custom_id and embedding for each record
    """
    result = client.files.content(filename)
    
    output_file = result.text

    embedding_results = []
    for line in output_file.split('\n')[:-1]:
        data =json.loads(line)
        custom_id = data.get('custom_id')
        embedding = data['response']['body']['data'][0]['embedding']
        embedding_results.append([custom_id, embedding])


    embedding_results = pd.DataFrame(embedding_results, columns=['custom_id', 'embedding'])
    
    return embedding_results



def run_batch_embedding_loop():
    """
    Runs a loop to manage batch embedding processes.
    The function continuously monitors batch status, automatically starting new batches
    when previous ones are complete, up to batch 15. It provides progress updates during execution
    and waits 5 minutes seconds between status checks.
    """
    
    client = OpenAI()
    batches = client.batches.list()
    
    while next_batch <= 15:
        #Check progress of ongoing batch and start new batch if needed
        last_completed_batch = get_last_finished_batch(batches)
        next_batch = last_completed_batch + 1
        if not batch_in_progress(batches):
            if next_batch <= 15:
                print(f"Starting batch {next_batch}")
                create_embedding_request(next_batch, client)
            else:
                print("All batches completed")
                
        else:
            batch_number, completed, total = get_batch_progress(batches)
            print(f"Batch {batch_number} in progress. Completed {completed} out of {total}"),

        
        # Fetch completed batch embeddings and save them locally.
        unsaved_completed_batch_files = fetch_completed_unsaved_filenames(batches)
        if unsaved_completed_batch_files:
            for output_file_id, batch_number in unsaved_completed_batch_files:
                embedding_results = get_file_contents(output_file_id, client)
                embedding_results.to_csv(here(f"data/embeddings/embeddings_batch_{batch_number}.csv"), index=False)
                print(f"Batch {batch_number} saved")
        
        
        time.sleep(300)
        
        
if __name__ == "__main__":
    
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
        
        
    run_batch_embedding_loop()
    