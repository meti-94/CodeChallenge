import pandas as pd


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
            print(df.dropna(subset=[cl], how='any')[cl], type(df.dropna(subset=[cl], how='any')[cl]))
            exp[cl]=df.dropna(subset=[cl], how='any')[cl].to_dict()

    return experiments

if __name__=='__main__':
    print(extract_experiments())