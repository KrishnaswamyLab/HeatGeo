import yaml
import os
import pandas as pd

def get_experiment_config(model_name, run_name):
    """
    Get the config of an experiment (with run name) - multirun
    """
    file_path = os.path.join("../logs/experiments/multiruns",model_name,run_name,"multirun.yaml") #if sweep params were provided in command line
    if not os.path.exists(file_path):    #if a sweeper config was used
        file_path = os.path.join("../logs/experiments/multiruns",model_name,run_name,"0",".hydra","hydra.yaml")
    
    with open(file_path, 'r') as file:
            exp_config = yaml.safe_load(file) 
    return exp_config

def get_run_config(model_name, run_name, run_id):
    """
    Get the config of a specific run (with run id)
    """
    file_path = os.path.join("../logs/experiments/multiruns",model_name,run_name,run_id,".hydra","config.yaml")
    with open(file_path, 'r') as file:
        run_config = yaml.safe_load(file)
    return run_config

def get_run_results(model_name, run_name, run_id):
    """
    Get the results of a specific run (with run id)
    """
    dir_path = os.path.join("../logs/experiments/multiruns",model_name,run_name,run_id)
    pkl_files = [f for f in os.listdir(dir_path) if "pkl" in f]
    if len(pkl_files)!=1:
        print("No PKL file found for {model_name} {run_name} {run_id}".format(model_name=model_name, run_name=run_name, run_id=run_id))
        print("Config for this run : ")
        print(get_run_config(model_name, run_name, run_id))
        return None
    else:
        pkl_file = pkl_files[0]
        return pd.read_pickle(os.path.join(dir_path,pkl_file))

def get_sweep_variables(exp_config):
    """
    Get the variables that were swept in the experiment
    """
    if exp_config["hydra"]["sweeper"]["params"] is not None: # this is used if a sweeper config was used
        variables = [(k,[v_.strip() for v_ in v.split(",")]) for k,v in exp_config["hydra"]["sweeper"]["params"].items()]
    else:
        variables = [(s.split("=")[0],s.split("=")[1].split(",")) for s in exp_config["hydra"]["overrides"]["task"]]
    variables = {v[0]:v[1] for v in variables if len(v[1]) > 1}
    return variables


def extract_variables_from_run(variables, run_config):
    """
    Extract the values of the variables that were swept in the experiment, from the config of a specific run
    """
    extracted_variables = {}
    for conf_var in variables.keys():
        conf_value = None
        if conf_var == "data":
            splitted_conf_var = ["dataset_name"]
        else:
            splitted_conf_var = conf_var.split(".")
        for conf_ in splitted_conf_var:
            if conf_value is None:
                conf_value = run_config[conf_]
            else:
                conf_value = conf_value[conf_]
        ### THIS IS A FIX TO DISTINGUISH BETWEEN SWISS ROLL DATASETS - REMOVE IN NEXT ITERATION ---
        if conf_var == "data":
            if conf_value == "tree":
                if run_config["data"]["n_dim"] == 30:
                    conf_value = "tree_high"
        ### ---------------------------------------------------------------------------------------
        
        extracted_variables[conf_var] = conf_value
    return extracted_variables

def get_extended_run_results(model_name, run_name, run_id, sweep_variables):
    run_config = get_run_config(model_name, run_name, run_id)

    variables_from_run = extract_variables_from_run(sweep_variables, run_config)

    run_results = get_run_results(model_name, run_name, run_id)

    if run_results is not None:
        for var in variables_from_run.keys():
            run_results[var] = variables_from_run[var]

    return run_results


def get_all_results_exp(model_name, run_name, sweep_variables):
    dir_name = os.path.join("../logs/experiments/multiruns",model_name,run_name)
    run_ids = [ f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name,f))]

    df_list = []
    for run_id in run_ids:
        run_results = get_extended_run_results(model_name, run_name, run_id, sweep_variables)
        if run_results is not None:
            df_list.append(run_results)

    df_results = pd.concat(df_list)
    return df_results

def get_results_to_plot(df_results, sweep_variables, constraints, metric):
    metric_columns= [c for c in df_results.columns if c not in list(sweep_variables.keys()) + ["Method","Seed","# group"]]
    mean_df = df_results.groupby(list(sweep_variables.keys()))[metric_columns].mean().reset_index()
    std_df = df_results.groupby(list(sweep_variables.keys()))[metric_columns].std().reset_index()

    metric = "PearsonR"
    x_label = "Tau"

    for key in constraints.keys():
        if key in mean_df.columns:
            mean_df = mean_df.loc[mean_df[key].isin(constraints[key])]
            std_df = std_df.loc[std_df[key].isin(constraints[key])]

    return mean_df, std_df


##### LATEX TABLES MAKING #####

def get_best_results(df,metric, sweep_variables, best_order):

    df_val = df.loc[(df["Seed"]>=42) & (df["Seed"]<=46)].copy()

    df_m = df_val.groupby(list(sweep_variables.keys()))[metric].mean().reset_index()
    df_s = df_val.groupby(list(sweep_variables.keys()))[metric].std().reset_index()

    idx = df_m.groupby(['data',"data.manifold_noise"])[metric].transform(best_order) == df_m[metric]
    df_best_m = df_m[idx].copy()
    
    
    df_best_m[metric + "_std"] = df_s[idx][metric]

    df_best_m[metric] = df_best_m[[metric,metric+"_std"]].apply(lambda x: "$" + str(round(x[metric],3)) + " \pm " + str(round(x[metric+"_std"],3)) + "$", axis = 1)

    df_best_m = df_best_m.drop_duplicates(subset = ["data","data.manifold_noise",metric], keep = "first" ).copy()

    return df_best_m


def get_best_results_test(df, metric, sweep_variables, best_order):
    """
    Retrieves the best hyper-parameters for each data/manifold noise combination and returns the test results for these hyper-parameters.

    Validation dataset : seed = [42->46]
    Test dataset : seed = [47->51]
    
    """
    df_val = df.loc[(df["Seed"]>=42) & (df["Seed"]<=46)].copy()
    df_test = df.loc[(df["Seed"]>=47)].copy()

    df_m = df_val.groupby(list(sweep_variables.keys()))[metric].mean().reset_index()
    df_s = df_val.groupby(list(sweep_variables.keys()))[metric].std().reset_index()

    idx = df_m.groupby(['data',"data.manifold_noise"])[metric].transform(best_order) == df_m[metric]
    df_best_m = df_m[idx].copy()

    df_best_test = pd.merge(df_test,df_best_m[sweep_variables.keys()],how = "inner", on = list(sweep_variables.keys()))

    df_m_test = df_best_test.groupby(["data","data.manifold_noise"])[metric].mean().reset_index()
    df_s_test = df_best_test.groupby(["data","data.manifold_noise"])[metric].std().reset_index()

    df_m_test[metric + "_std"] = df_s_test[metric]

    df_m_test[metric] = df_m_test[[metric,metric+"_std"]].apply(lambda x: "$" + str(round(x[metric],3)) + " \pm " + str(round(x[metric+"_std"],3)) + "$", axis = 1)

    df_m_test = df_m_test.drop_duplicates(subset = ["data","data.manifold_noise",metric], keep = "first" ).copy()

    return df_m_test, df_best_m

def add_hline(latex: str, index: int) -> str:
    """
    Adds a horizontal `index` lines before the last line of the table

    Args:
        latex: latex table
        index: index of horizontal line insertion (in lines)
    """
    lines = latex.splitlines()
    lines.insert(index +4, "\midrule")
    return '\n'.join(lines).replace('NaN', '')