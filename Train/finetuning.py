from openai import OpenAI
from time import sleep
import pandas as pd
import logging
from utils import *
load_env_variables()

def upload_file(datafile):
    """
    Uploads a data file to OpenAI for fine-tuning.

    Args:
        datafile (str): The path to the data file to be uploaded.

    Returns:
        str: The OpenAI File ID associated with the uploaded file.
    """
    client = OpenAI()
    response = client.files.create(file=open(datafile, "rb"), purpose="fine-tune")
    file_id = response.id
    logging.debug(f"{datafile} is uploaded")
    logging.debug("Sleep 30 seconds...")
    sleep(30)  # wait until the dataset would be prepared
    return file_id


def track_fileid(file_name, file_id):
    """
    Tracks the OpenAI File ID in an Excel file associated with the data file.

    Args:
        file_name (str): The name of the data file.
        file_id (str): The OpenAI File ID to be tracked.
    """
    # Define the path, sheet name, and column name in the Excel file
    file_path = './Experiments.xlsx'
    sheet_name = 'Data'
    column_name = 'Data File'

    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Find the index of the value in the specified column
    index_of_value = df[df[column_name] == file_name].index[0]

    # Update the 'Stats' column with the provided statistics
    df.loc[index_of_value, ['OpenAI FileID']] = file_id

    # Write the updated DataFrame back to the Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def create_model(training_id, validation_id):
    """
    Creates a fine-tuning job using the specified training and validation files.

    Args:
        training_id (str): The OpenAI File ID of the training file.
        validation_id (str): The OpenAI File ID of the validation file.

    Returns:
        str: The job ID associated with the created fine-tuning job.
    """
    client = OpenAI()
    response = client.fine_tuning.jobs.create(
        training_file=training_id,
        validation_file=validation_id,
        model='gpt-3.5-turbo-1106',
    )
    job_id = response.id
    return job_id


def track_job(job_id):
    """
    Tracks the job ID in an Excel file associated with fine-tuning jobs.

    Args:
        job_id (str): The job ID to be tracked.
    """
    file_path = './Experiments.xlsx'
    sheet_name = 'Model'
    column_name = 'Job ID'

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    new_data = pd.DataFrame({column_name: [job_id]})

    df = pd.concat([df, new_data], ignore_index=True)

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    
    client = OpenAI()
    
    response = client.files.create(file=open("./data/train-2-5.jsonl", "rb"), purpose="fine-tune")
    training_id = response.id
    print("Training dataset is uploaded")
    print("Sleep 30 seconds...")
    sleep(30)  # wait until dataset would be prepared
    response = client.files.create(file=open("./data/valid-2.jsonl", "rb"), purpose="fine-tune")
    validation_id = response.id
    print("Validation dataset is uploaded")
    print("Sleep 30 seconds...")
    sleep(30)  # wait until dataset would be prepared
    
    response = client.fine_tuning.jobs.create(  training_file=training_id,
                                                validation_file=validation_id,
                                                model='gpt-3.5-turbo-1106',
                                              )
    print("Fine-tune job is started")
    print(response)
    job_id = response.id

