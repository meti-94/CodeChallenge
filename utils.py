import os
from time import sleep
from dotenv import load_dotenv


def load_env_variables(file_path='conf.env'):
    """
    Load environment variables from a file and set them in the current environment.
    
    Parameters:
        file_path (str): Path to the environment variable file (default is 'conf.env').
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    # Load variables from the file
    load_dotenv(file_path)
    
    # # Set the variables in the environment
    # for key, value in os.environ.items():
    #     print(f"Setting environment variable: {key}={value}")
    #     os.environ[key] = value


def wait_untill_done(job_id, client): 
    events = {}
    while True: 
        response = client.fine_tuning.jobs.list_events(id=job_id, limit=10)
        # collect all events
        for event in response["data"]:
            if "data" in event and event["data"]:
                events[event["data"]["step"]] = event["data"]["train_loss"]
        messages = [it["message"] for it in response.data]
        for m in messages:
            if m.startswith("New fine-tuned model created: "):
                return m.split("created: ")[1], events
        sleep(10)
