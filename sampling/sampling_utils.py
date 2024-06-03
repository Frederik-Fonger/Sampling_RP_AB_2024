import copy
import os
from copy import deepcopy
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
import ConfigManager
from ConfigManager import Config_Manager as config
import inspect
import pm4py
import pandas as pd


def import_raw_data():
    """
    Import file with raw data from spotter buoy
    :return: Pandas dataframe of the csv file
    """
    data = pd.read_csv(config.source_path, index_col=False)
    print("data import of " + config.source_path + " complete")
    return data


def import_data():

    """
    Import file with data of type .csv or .xes
    :return: Pandas dataframe of the csv or xes file
    """
    data_type = config.source_path[-3:]
    config.data_type = data_type
    if data_type == "csv":
        return pd.read_csv(config.source_path, sep=config.seperator_in_log)
    elif data_type == "xes":
        config.ab_case_id_column = 'case:concept:name'
        config.ab_activity_column = 'concept:name'
        config.ab_timestamp_column = 'time:timestamp'
        event_log = pm4py.read_xes(config.source_path)
        df = pm4py.convert_to_dataframe(event_log)
        config.ab_format = "DF"
        return df


def time_calculation(start_time, start_time_sampling, start_time_pm_and_eval, end_time_sampling, end_time_pm_and_eval,
                     end_time):

    """
    Calculates and prints the calculation times
    :param start_time:
    :param start_time_sampling:
    :param start_time_pm_and_eval:
    :param end_time_sampling:
    :param end_time_pm_and_eval:
    :param end_time:
    :return:
    """
    # calculate runtime of sampling
    execution_time_sampling = end_time_sampling - start_time_sampling
    print("runtime sampling:", execution_time_sampling, "seconds")

    # calculate runtime of process mining and evaluation
    execution_time_pm_and_eval = end_time_pm_and_eval - start_time_pm_and_eval
    print("runtime process mining and evaluation:", execution_time_pm_and_eval, "seconds")

    # calculate full runtime of the program
    execution_time = end_time - start_time
    print("runtime complete:", execution_time, "seconds")

    # generate text file of runtime as output
    if config.output:
        with open(config.output_path + 'runtimes.txt', 'w') as f:
            f.write("runtime sampling:" + str(execution_time_sampling) + "seconds \n")
            f.write("runtime process mining and evaluation:" + str(execution_time_pm_and_eval) + "seconds \n")
            f.write("runtime complete:" + str(execution_time) + "seconds")


def save_info_file():
    """
    this function saves the Config-Manager as a file in the folder of the iteration
    :return:
    """
    if config.output:
        with open(config.output_path + "ConfigManager.txt", "w") as file:
            file.write(inspect.getsource(ConfigManager.Config_Manager))


def save_logs(base_log, samples):
    """
    saving the base log and the sampled logs as a file in the output directory
    :param base_log: the base log as a dataframe
    :param samples: a dict with the sampled logs as a dataframe
    :return:
    """
    if config.output:
        os.mkdir(config.output_path + "logs/")

        if config.log_output_format == "csv":
            for key in samples:
                samples[key].to_csv(config.output_path + "logs/" + str(key) + ".csv")
            base_log.to_csv(config.output_path + "logs/base_log.csv")
        elif config.log_output_format == "xes":
            for key in samples:
                pm4py.write_xes(samples[key], config.output_path + "logs/" + str(key) + ".xes")
            pm4py.write_xes(base_log, config.output_path + "logs/base_log.xes")

    return None


def time_calculation_multi_algo(start_time, end_time):
    """
    this function calculate the runtime of multiple algorithms combined
    :param start_time: time when the algorithms started
    :param end_time: time when the algorithms ended
    :return:
    """
    # calculate runtime
    execution_time = end_time - start_time
    print("runtime complete:", execution_time, "seconds")
    if config.output:
        # generate text file of runtime as output
        with open(config.output_path + 'runtimes.txt', 'w') as f:
            f.write("runtime complete:" + str(execution_time) + "seconds")

def compare_AL_and_TBR(methods):

    if (methods == 'alignments').all():
        return "AL"
    elif (methods == 'token based replay').all():
        return "TBR"
    else:
        return "AL and TBR"


def curve_plot_compare_algos(quality_dataframe, noise_threshold=None, miner="inductive miner"):

    """
    comparing the different sampling algorithms in a curve chart by their f1 score
    :param quality_dataframe:
    :param noise_threshold:
    :param miner:
    :return:
    """
    plt.clf()
    line_width = 2

    for current_miner_th in next(iter(quality_dataframe.values())).keys():
        for current_rare_act_th in next(iter(next(iter(quality_dataframe.values())).values())).keys():
            plt.clf()
            for algo in quality_dataframe.keys():
                plt.plot(range(len(quality_dataframe[algo][current_miner_th][current_rare_act_th][miner])),
                         quality_dataframe[algo][current_miner_th][current_rare_act_th][miner]['f1'], label=algo + " " + compare_AL_and_TBR(quality_dataframe[algo][current_miner_th][current_rare_act_th][miner]['method']),
                         linewidth=line_width)

            # Labeling the curve chart
            plt.xlabel('file')
            plt.ylabel('f1 score')
            plt.title('Curve Chart of ' + miner + "\n" + config.source_path)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.ylim(0, 1)
            plt.xticks(range(len(next(iter(quality_dataframe.values()))[current_miner_th][current_rare_act_th][miner])),
                       next(iter(quality_dataframe.values()))[current_miner_th][current_rare_act_th][miner]['file'],
                       rotation=45, ha='right')

            if config.output:
                if miner == "inductive miner":
                    plt.savefig(config.output_path + '/' + str(current_rare_act_th) + '_' + str(
                        current_miner_th) + '_Curve_Chart_' + miner + '.png', bbox_inches='tight')
                else:
                    plt.savefig(config.output_path + '/' + str(current_rare_act_th) + '_' + str(
                        current_miner_th) + '_Curve_Chart_' + miner + '.png', bbox_inches='tight')


    # calculate the f1 division though the f1 score from the base log
    for current_miner_th in next(iter(quality_dataframe.values())).keys():
        for current_rare_act_th in next(iter(next(iter(quality_dataframe.values())).values())).keys():
            plt.clf()
            for algo in quality_dataframe.keys():
                f1_from_base_log = quality_dataframe[algo][current_miner_th][current_rare_act_th][miner].loc[
                    quality_dataframe[algo][current_miner_th][current_rare_act_th][miner][
                        'file'] == "base_event_log", 'f1'].values[0]

                f1_div = quality_dataframe[algo][current_miner_th][current_rare_act_th][miner]['f1'] / f1_from_base_log
                quality_dataframe[algo][current_miner_th][current_rare_act_th][miner]['f1_division'] = f1_div

                plt.plot(range(len(quality_dataframe[algo][current_miner_th][current_rare_act_th][miner])),
                         quality_dataframe[algo][current_miner_th][current_rare_act_th][miner]['f1_division'],
                         label=algo,
                         linewidth=line_width)

            # labeling the curve chart
            plt.xlabel('file')
            plt.ylabel('(f1 sample model)/(f1 base log model) score')
            plt.title('Curve Chart of ' + miner + "with" + "\n" + config.source_path)
            plt.legend(loc="lower left")
            plt.ylim(0, 2)
            x_labels = next(iter(quality_dataframe.values()))[current_miner_th][current_rare_act_th][miner][
                           'file'].astype(str) + " " + \
                       next(iter(quality_dataframe.values()))[current_miner_th][current_rare_act_th][miner]['method']
            plt.xticks(range(len(next(iter(quality_dataframe.values()))[current_miner_th][current_rare_act_th][miner])),
                       x_labels, rotation=45, ha='right')

            # save generated chart as png
            if config.output:
                if miner == "inductive miner":
                    plt.savefig(config.output_path + '/div' + str(current_rare_act_th) + '_' + str(
                        current_miner_th) + '_Curve_Chart_' + miner + '.png', bbox_inches='tight')
                else:
                    plt.savefig(config.output_path + '/div' + str(current_rare_act_th) + '_' + str(
                        current_miner_th) + '_Curve_Chart_' + miner + '.png', bbox_inches='tight')





def curve_plot_compare_algos_for_CC(quality_dataframe, noise_threshold=None, miner="inductive miner"):

    """
    comparing the results in a curve chart by their f1 score using once the f1 score of the
    sampled file and of the not sampled file
    :param quality_dataframe:
    :param noise_threshold:
    :param miner:
    :return:
    """
    line_width = 2


    for current_miner_th in next(iter(quality_dataframe.values())).keys():
        for current_rare_act_th in next(iter(next(iter(quality_dataframe.values())).values())).keys():
            plt.clf()
            for algo in quality_dataframe.keys():
                plt.plot(range(len(quality_dataframe[algo][current_miner_th][current_rare_act_th][miner])),
                         quality_dataframe[algo][current_miner_th][current_rare_act_th][miner]['f1'], label=algo,
                         linewidth=line_width)

            # labeling the curve chart
            plt.xlabel('file')
            plt.ylabel('f1 score')
            plt.title('Curve Chart of ' + miner + "\n" + config.source_path)
            plt.legend(loc='center left')
            plt.ylim(0, 1)
            plt.xticks(range(len(next(iter(quality_dataframe.values()))[current_miner_th][current_rare_act_th][miner])),
                       next(iter(quality_dataframe.values()))[current_miner_th][current_rare_act_th][miner]['file'],
                       rotation=45, ha='right')

            # save output of chart as png
            if config.output:
                if miner == "inductive miner":
                    plt.savefig(config.output_path + '/' + str(current_rare_act_th) + '_' + str(
                        current_miner_th) + '_Curve_Chart_' + miner + '.png', bbox_inches='tight')
                else:
                    plt.savefig(config.output_path + '/' + str(current_rare_act_th) + '_' + str(
                        current_miner_th) + '_Curve_Chart_' + miner + '.png', bbox_inches='tight')

    # calculate the f1 division though the f1 score from the base log
    for current_miner_th in next(iter(quality_dataframe.values())).keys():
        for current_rare_act_th in next(iter(next(iter(quality_dataframe.values())).values())).keys():
            plt.clf()
            for algo in quality_dataframe.keys():
                f1_from_base_log = quality_dataframe[algo][current_miner_th][current_rare_act_th][miner].loc[
                    quality_dataframe[algo][current_miner_th][current_rare_act_th][miner][
                        'file'] == "base log", 'f1'].values[0]

                f1_div = quality_dataframe[algo][current_miner_th][current_rare_act_th][miner]['f1'] / f1_from_base_log
                quality_dataframe[algo][current_miner_th][current_rare_act_th][miner]['f1_division'] = f1_div

                plt.plot(range(len(quality_dataframe[algo][current_miner_th][current_rare_act_th][miner])),
                         quality_dataframe[algo][current_miner_th][current_rare_act_th][miner]['f1_division'],
                         label=algo,
                         linewidth=line_width)

            # labeling the curve chart
            plt.xlabel('file')
            plt.ylabel('(f1 sample)/(f1 base log) score')
            plt.title('Curve Chart of ' + miner + "with" + "\n" + config.source_path)
            plt.legend(loc="lower left")
            plt.ylim(0, 2)
            x_labels = next(iter(quality_dataframe.values()))[current_miner_th][current_rare_act_th][miner][
                           'file'].astype(str) + " " + \
                       next(iter(quality_dataframe.values()))[current_miner_th][current_rare_act_th][miner]['method']
            plt.xticks(range(len(next(iter(quality_dataframe.values()))[current_miner_th][current_rare_act_th][miner])),
                       x_labels, rotation=45, ha='right')

            # save generated chart as png
            if config.output:
                if miner == "inductive miner":
                    plt.savefig(config.output_path + '/div' + str(current_rare_act_th) + '_' + str(
                        current_miner_th) + '_Curve_Chart_' + miner + '.png', bbox_inches='tight')
                else:
                    plt.savefig(config.output_path + '/div' + str(current_rare_act_th) + '_' + str(
                        current_miner_th) + '_Curve_Chart_' + miner + '.png', bbox_inches='tight')


def average_sampling_results(quality_metrics):
    """
    This function averages all resulting quality metrics in the quality_metrics from one type of sampling algo. From
    which sampling algo the results are averaged is specified in the config.algos_to_average.
    :param quality_metrics: quality_metrics dict
    :return: the quality_metrics dict which the results from the specified algos as average
    """
    # iterate over all sampling algos where the results should be averaged
    for algo_to_repeat in config.algos_to_average:
        if algo_to_repeat in config.sampling_algo:
            result_list = []

            # the key need to be cased to a list, as the dict changes while used. This can lead to errors
            keys = list(quality_metrics.keys())

            # aggregate the results form one sampling algo in the result_list
            for algo in keys:
                if algo.startswith(algo_to_repeat):
                    result_list.append(quality_metrics.pop(algo))

            # get the keys from the filtering parameters, to iterate over the configurations
            miner_ths = result_list[0].keys()
            rare_act_ths = result_list[0][list(miner_ths)[0]].keys()

            # the iteration counter is for the result division at the end. It starts by 1 as the first value is added
            # before the for-loop

            miner_th_dict = {}

            # iterate over all parameter configs for the filter
            for miner_th in miner_ths:
                rare_act_dict = {}
                for race_act_th in rare_act_ths:
                    iteration_counter = 1
                    # initiate the fist values in the dataframe for calculation, as something need to be in it to add
                    # the next value
                    df_im = result_list[0][miner_th][race_act_th]["inductive miner"]  # init first result dataframe,
                    # to add the other results later

                    df_hm = result_list[0][miner_th][race_act_th]["heuristic miner"]

                    # remove the data that was added to the calculation dataframe, so that it is not used again in the
                    # for-loop
                    result_without_first_element = deepcopy(result_list)
                    result_without_first_element.pop(0)

                    # iterate over all results from the current sampling algo
                    for result in result_without_first_element:
                        # function to be applied on the columns of the dataframe
                        def avg_without_strings(col1, col2):
                            """
                            from CHAT-GPT
                            """
                            try:
                                # Convert columns to floats and calculate average

                                return col1.astype(float) + col2.astype(float)
                            except ValueError:
                                return col1  # Return None for non-numeric values

                        # apply the "add function" on the results of the inductive miner df
                        for col3 in df_im.columns:
                            df_im[col3] = avg_without_strings(df_im[col3],
                                                              result[miner_th][race_act_th]["inductive miner"][col3])

                        # apply the "add function" on the results of the heuristic miner df
                        for col in df_hm.columns:
                            df_hm[col] = avg_without_strings(df_hm[col],
                                                             result[miner_th][race_act_th]["heuristic miner"][col])

                        iteration_counter = iteration_counter + 1

                    # divide the value in the resulting df though the iteration_counter as they only added before
                    df_hm["f1"] = df_hm["f1"] / iteration_counter
                    df_hm["fitness"] = df_hm["fitness"] / iteration_counter
                    df_hm["precision"] = df_hm["precision"] / iteration_counter
                    df_hm["generalisation"] = df_hm["generalisation"] / iteration_counter
                    df_hm["simplicity"] = df_hm["simplicity"] / iteration_counter
                    df_im["f1"] = df_im["f1"] / iteration_counter
                    df_im["fitness"] = df_im["fitness"] / iteration_counter
                    df_im["precision"] = df_im["precision"] / iteration_counter
                    df_im["generalisation"] = df_im["generalisation"] / iteration_counter
                    df_im["simplicity"] = df_im["simplicity"] / iteration_counter

                    # build the results dict as it originally was
                    rare_act_dict[race_act_th] = {"inductive miner": df_im, "heuristic miner": df_hm}

                miner_th_dict[miner_th] = rare_act_dict

            quality_metrics[algo_to_repeat] = miner_th_dict

    return quality_metrics


def add_algos_to_repeat():
    """
    This function repeats the algorithm until the amount in the config is reached
    """
    for algo_to_repeat in config.algos_to_average:
        if algo_to_repeat in config.sampling_algo:
            counter = 2
            while counter <= config.algos_to_average_repeats:
                config.sampling_algo.append(algo_to_repeat + str(counter))
                counter += 1


def get_log_info():
    import pm4py
    from pm4py.objects.log.importer.xes import importer as xes_importer

    # Load the XES event log
    log_path = "data/Eventlogs/CoSeLoG WABO 3/CoSeLoG WABO 3.xes"
    # log = xes_importer.apply(log_path)
    log = pm4py.read_xes(log_path)

    # Calculate the number of unique traces
    num_unique_traces = len(pm4py.statistics.traces.generic.log.case_statistics.get_variant_statistics(log))
    # Calculate the number of cases (number of traces)
    num_cases = log['case:concept:name'].nunique()
    # Calculate the number of events
    num_events = len(log)

    num_unique_events = log['concept:name'].nunique()

    # Print the results
    print(log_path)
    print("Number of Cases:", num_cases)
    print("Number of Unique Traces:", num_unique_traces)
    print("Number of Events:", num_events)
    print("Number of Unique Events:", num_unique_events)

