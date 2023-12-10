import json
import logging
import tiktoken  # for token counting
import numpy as np
from collections import defaultdict
import pandas as pd

# The script copied from here: https://cookbook.openai.com/examples/chat_finetuning_data_prep


def eval_dataset(data_path):
    information = {}
    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # Initial dataset stats
    length = len(dataset)
    logging.debug("Num examples: %d", length)
    logging.debug("First example:")
    for message in dataset[0]["messages"]:
        logging.debug(message)
    
    information['length']=length
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        logging.debug("Found errors:")
        for k, v in format_errors.items():
            logging.debug("%s: %s", k, v)
    else:
        logging.debug("No errors found")
    information['format_errors'] = format_errors
    encoding = tiktoken.get_encoding("cl100k_base")

    # not exact!
    # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name, information):
        logging.debug("\n#### Distribution of %s:", name)
        logging.debug("min / max: %s, %s", min(values), max(values))
        logging.debug("mean / median: %s, %s", np.mean(values), np.median(values))
        logging.debug("p5 / p95: %s, %s", np.quantile(values, 0.1), np.quantile(values, 0.9))
        information[name] = {'min':min(values), 'max':max(values), 'mean':np.mean(values), 'median':np.median(values), 'p5':np.quantile(values, 0.1), 'p95':np.quantile(values, 0.9)}

    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    logging.debug("Num examples missing system message: %d", n_missing_system)
    logging.debug("Num examples missing user message: %d", n_missing_user)
    information['n_missing_system']=n_missing_system
    information['n_missing_user']=n_missing_user
    
    print_distribution(n_messages, "num_messages_per_example", information)
    print_distribution(convo_lens, "num_total_tokens_per_example", information)
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example", information)
    n_too_long = sum(l > 4096 for l in convo_lens)
    logging.debug("\n%d examples may be over the 4096 token limit, they will be truncated during fine-tuning", n_too_long)

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    logging.debug("Dataset has ~%d tokens that will be charged for during training", n_billing_tokens_in_dataset)
    logging.debug("By default, you'll train for %d epochs on this dataset", n_epochs)
    logging.debug("By default, you'll be charged for ~%d tokens", n_epochs * n_billing_tokens_in_dataset)
    information['n_billing_tokens_in_dataset']=n_billing_tokens_in_dataset
    information['n_epochs']=n_epochs
    information['n_billing_tokens_in_dataset']=n_billing_tokens_in_dataset
    return information

def track_stats(file_name, stats):
    """
    Track statistics of a dataset in an Excel file.

    Parameters:
    - file_name (str): Name of the data file.
    - stats (dict): Dictionary containing the statistics to be tracked.

    Returns:
    None

    This function reads an Excel file containing experimental data and updates the specified sheet with the given statistics.
    It searches for the specified 'file_name' in the 'Data File' column and updates the corresponding row with the provided statistics.

    Example:
    >>> file_name = 'experiment_data_1.xlsx'
    >>> statistics = {'mean': 25.5, 'std_dev': 4.2, 'max': 30}
    >>> track_stats(file_name, statistics)
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
    df.loc[index_of_value, ['Stats']] = str(stats)
    
    # Write the updated DataFrame back to the Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__=='__main__':
    stats = eval_dataset('data/train-2-10.jsonl')
    track_stats('data/train-2-10.jsonl', stats)