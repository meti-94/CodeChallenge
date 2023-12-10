import json  
import pandas as pd  

DEFAULT_SYSTEM_PROMPT = """
                        تو کارشناس شرکت پارس پک هستی. پارس پک در زمینه ارائه خدمات زیرساخت فعالیت میکند. باید سعی کنی به سوال مشتریان پاسخ صحیح بدهی
                        """  

def create_example(question, answer):
    """
    Create a conversation example with a system prompt, user question, and assistant answer.

    Parameters:
    - question (str): User's question content.
    - answer (str): Assistant's answer content.

    Returns:
    dict: Example in the form of a dictionary with messages containing system, user, and assistant roles.
    """
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }

def create_training_subset(paraphrase=1, count=10):
    """
    Create a training subset from a dataset, considering paraphrased versions.

    Parameters:
    - paraphrase (int): Number of paraphrased versions to consider (default is 1).
    - count (int): Number of examples to include in the subset.

    Returns:
    str: Filename of the created training subset in JSONL format.
    """
    subset = []
    df = pd.read_excel('./data/samples_v2.xlsx')
    
    if paraphrase <= 1:
        subset += [create_example(row['question'], row['fact']) for _, row in df[:count].iterrows()]
    if paraphrase <= 2:
        subset += [create_example(row['question_1'], row['fact']) for _, row in df[:count].iterrows()]
    if paraphrase <= 3:
        subset += [create_example(row['question_3'], row['fact']) for _, row in df[:count].iterrows()]
    if paraphrase <= 4:
        subset += [create_example(row['question_2'], row['fact']) for _, row in df[:count].iterrows()]

    filename = f"data/train-{paraphrase}-{count}.jsonl"

    with open(filename, "w") as f:
        for _, jsn in enumerate(subset):
            example_str = json.dumps(jsn)
            f.write(example_str + "\n")

    return filename

def create_validation_subset(count=10):
    """
    Create a validation subset from a dataset.

    Parameters:
    - count (int): Number of examples to include in the subset.

    Returns:
    str: Filename of the created validation subset in JSONL format.
    """
    subset = []
    df = pd.read_excel('./data/samples_v2.xlsx')
    subset += [create_example(row['question_2'], row['fact']) for _, row in df[:count].iterrows()]

    filename = f"data/valid-{count}.jsonl"

    with open(filename, "w") as f:
        for _, jsn in enumerate(subset):
            example_str = json.dumps(jsn)
            f.write(example_str + "\n")

    return filename

def create_subset(paraphrase=2, count=10):
    """
    Create both training and validation subsets.

    Parameters:
    - paraphrase (int): Number of paraphrased versions to consider (default is 2).
    - count (int): Number of examples to include in each subset.

    Returns:
    tuple: Filenames of the created training and validation subsets in JSONL format.
    """
    train_file = create_training_subset(paraphrase, count)
    valid_file = create_validation_subset(count)
    return train_file, valid_file

def track_samples(data_file):
    """
    Track training and validation datasets in an Excel file.

    Parameters:
    - training_file (str): File path or name for the training dataset.
    - validation_file (str): File path or name for the validation dataset.

    Returns:
    None
    """

    file_path = './Experiments.xlsx'
    sheet_name = 'Data'
    column_name = 'Data File'

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    new_data = pd.DataFrame({column_name: [data_file]})
    
    df = pd.concat([df, new_data], ignore_index=True)
    
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == "__main__":
    create_training_subset(2, 5)
    create_validation_subset(5)
    create_subset()