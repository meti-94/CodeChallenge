# Track training status

import logging
import pandas as pd 
from openai import OpenAI
import io


def check_job(job_id):
    """
    Check the status of a fine-tuning job.

    Parameters:
    - job_id (str): The identifier of the fine-tuning job.

    Returns:
    - bool: True if the job has succeeded, False otherwise.
    """
    client = OpenAI()
    job = client.fine_tuning.jobs.retrieve(job_id)
    logging.info(job.status)
    logging.info(type(job))
    if job.status in ['validating_files', 'queued', 'running', 'failed', 'cancelled']:
        logging.warning(job.status)
        if job.status=='failed':
            logging.warning(job.error)
        return False
    if job.status == 'succeeded':
        return True

def track_model(job_id):
    """
    Track the fine-tuned model by updating information in an Excel file.

    Parameters:
    - job_id (str): The identifier of the fine-tuning job.

    Returns:
    None
    """
    client = OpenAI()
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    # Define the path, sheet name, and column name in the Excel file
    file_path = './Experiments.xlsx'
    sheet_name = 'Model'
    column_name = 'Job ID'

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    index_of_value = df[df[column_name] == job_id].index[0]

    df.loc[index_of_value, ['Spec']] = str(job)
    
    # Write the updated DataFrame back to the Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

def track_result(job_id):
    """
    Track the result of a fine-tuning job by saving it to an Excel file.

    Parameters:
    - job_id (str): The identifier of the fine-tuning job.

    Returns:
    None
    """
    client = OpenAI()
    job = client.fine_tuning.jobs.retrieve(job_id)
    file_id = job.result_files[0]
    file_content = client.files.retrieve_content(file_id)
    result = pd.read_csv(io.StringIO(file_content), sep=",")
    result.to_excel(f'./runs/{file_id}.xlsx')


if __name__=='__main__':
    flag = check_job('ftjob-jvcqyi9ScYoyOHBfMscsDLT7')
    track_result('ftjob-jvcqyi9ScYoyOHBfMscsDLT7')
