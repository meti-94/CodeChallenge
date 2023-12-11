import os
import pandas as pd
from time import sleep
from openai import OpenAI
from dotenv import load_dotenv
from DAL.data import DEFAULT_SYSTEM_PROMPT



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

def provide_answer(question, model_name, maxtokens=100, tmp=0, choices_n=1):
    """
    Get model-generated responses for a given question using the OpenAI API.

    Parameters:
    - question (str): The user's input question.
    - model_name (str): The name of the OpenAI language model to use.
    - maxtokens (int): The maximum number of tokens in the generated response.
    - tmp (float): The temperature parameter controlling the randomness of the output.
    - choices_n (int): The number of response choices to generate.

    Returns:
    - list: A list of model-generated responses for the given question.
    """

    # Create an OpenAI client
    client = OpenAI()

    # Make a request to the OpenAI API for model-generated responses
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        max_tokens=maxtokens,
        temperature=tmp,
        n=choices_n
    )

    # Extract and return the content of the generated responses
    return [item.message.content for item in completion.choices]


if __name__=='__main__':
    load_env_variables()
    answers = provide_answer(
                    question = 'روی سرورهاتون سی پنل هم دارین؟',
                    model_name = "gpt-3.5-turbo-1106",
                    maxtokens= 30,
                    tmp=1,
                    choices_n=2)
    for ans in answers:
        print(ans)

