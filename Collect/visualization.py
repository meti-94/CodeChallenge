import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 


def extract_experiments():
    experiments = []
    
    file_path = './Experiments.xlsx'
    sheet_name = 'Model'
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    column_name = 'Spec'
    #########
    # Dirtiest way possible to eval the result. It was best if I save named tuple as dict in the first step
    created_at = 'created_at'
    error = 'error'
    fine_tuned_model = 'fine_tuned_model'
    finished_at = 'finished_at'
    hyperparameters = 'hyperparameters'
    n_epochs = 'n_epochs'
    batch_size = 'batch_size'
    learning_rate_multiplier = 'learning_rate_multiplier'
    model = 'model'
    object = 'object'
    organization_id = 'organization_id'
    result_files = 'result_files'
    status = 'status'
    trained_tokens = 'trained_tokens'
    training_file = 'training_file'
    validation_file = 'validation_file'
    #########
    for item in df[column_name].to_list():
        item = item.replace('=', ':').replace('(', '{').replace(')', '}').replace('Hyperparameters', '')
        experiments.append(eval(item[13:]))

    file_path = './Experiments.xlsx'
    sheet_name = 'Data'
    column_name = 'OpenAI FileID'
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    for exp in experiments:

        index_of_value = df[df[column_name] == exp['training_file']].index[0]
        training_file_name = df['Data File'].to_list()[index_of_value]
        exp['training_file_name']=training_file_name

        index_of_value = df[df[column_name] == exp['validation_file']].index[0]
        validation_file_name = df['Data File'].to_list()[index_of_value]
        exp['validation_file_name']=validation_file_name
    
    get_tuples = lambda lst: [(idx, itm) for idx, itm in enumerate(lst)]
    for exp in experiments:
        df = pd.read_excel(f'./runs/{exp["result_files"][0]}.xlsx')
        for cl in df.columns[1:]:
            exp[cl]=df.dropna(subset=[cl], how='any')[cl].to_dict()

    return experiments



def plot_dictionaries(dictionaries, legends, figsize=(8, 6), title="Dictionaries Plot", xlabel="X-axis", ylabel="Y-axis"):
    """
    Plot multiple dictionaries on a resizable figure.

    Parameters:
    - dictionaries: A list of dictionaries in the form {x1: y1, x2: y2, ...}
    - legends: A list of legend names corresponding to each dictionary
    - figsize: Tuple specifying the figure size, default is (8, 6)
    - title: String specifying the plot title, default is "Dictionaries Plot"
    - xlabel: String specifying the x-axis label, default is "X-axis"
    - ylabel: String specifying the y-axis label, default is "Y-axis"
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i, (dictionary, legend) in enumerate(zip(dictionaries, legends)):
        x_values, y_values = zip(*sorted(dictionary.items()))
        plt.plot(x_values, y_values, label=legend)

    plt.legend()
    plt.grid(True)  # Add grid
    plt.tight_layout()
    plt.show()

def bar_plot_with_names(data, names, title="Bar Plot with Names", xlabel="Categories", ylabel="Values", figsize=(8, 6)):
    """
    Create a bar plot with names on top of each bar.

    Parameters:
    - data: List of values for each category.
    - names: List of names corresponding to each category.
    - title: String specifying the plot title, default is "Bar Plot with Names".
    - xlabel: String specifying the x-axis label, default is "Categories".
    - ylabel: String specifying the y-axis label, default is "Values".
    - figsize: Tuple specifying the figure size, default is (8, 6).
    """
    plt.figure(figsize=figsize)
    x = np.arange(len(data))

    plt.bar(x, data, align='center', alpha=0.7)
    plt.xticks(x, names)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i, value in enumerate(data):
        plt.text(i, value + 0.1, str(value), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    print(extract_experiments())