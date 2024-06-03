import os
import time

import pandas as pd
import pm4py
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

from ConfigManager import Config_Manager as config


def process_mining(base_event_log, miner, noise_threshold, threshold, sampling_algo):
    """

    :param base_event_log:
    :param miner:
    :param noise_threshold:
    :param threshold:
    :param sampling_algo:
    :return:
    """
    if miner == "inductive miner":
        base_net, base_im, base_fm = pm4py.discover_petri_net_inductive(base_event_log, case_id_key='case_ID',
                                                                        activity_key='activity',
                                                                        timestamp_key='timestamp',
                                                                        noise_threshold=noise_threshold)
        if config.output:
            os.makedirs(config.output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                threshold) + '_inductive/petri_nets_' + sampling_algo)
        if config.output:
            os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
            pm4py.save_vis_petri_net(base_net, base_im, base_fm,
                                     config.output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                         threshold) + '_inductive/petri_nets_' + sampling_algo + '/base_dfg_' + str(
                                         noise_threshold) + '_' + str(threshold) + '.png')
            pm4py.write_pnml(base_net, base_im, base_fm,
                                     config.output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                         threshold) + '_inductive/petri_nets_' + sampling_algo + '/base_dfg_' + str(
                                         noise_threshold) + '_' + str(threshold) + '.pnml')
        base_dfg, base_start_activities, base_end_activities = pm4py.discover_dfg(
            base_event_log)
        if config.output:
            pm4py.save_vis_dfg(base_dfg, base_start_activities, base_end_activities,
                               config.output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                   threshold) + '_inductive/petri_nets_' + sampling_algo + '/base_net_' + str(
                                   noise_threshold) + '_' + str(threshold) + '.png')

    elif miner == "heuristic miner":

        if config.output:
            os.makedirs(config.output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                threshold) + '_heuristic/petri_nets_' + sampling_algo)
        base_net, base_im, base_fm = pm4py.discover_petri_net_heuristics(base_event_log, case_id_key='case_ID',
                                                                         activity_key='activity',
                                                                         timestamp_key='timestamp',
                                                                         dependency_threshold=noise_threshold)
        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
        if config.output:
            os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
            pm4py.save_vis_petri_net(base_net, base_im, base_fm,
                                     config.output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                         threshold) + '_heuristic/petri_nets_' + sampling_algo + '/base_dfg_' + str(
                                         noise_threshold) + '_' + str(threshold) + '.png')
            pm4py.write_pnml(base_net, base_im, base_fm,
                                     config.output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                         threshold) + '_heuristic/petri_nets_' + sampling_algo + '/base_dfg_' + str(
                                         noise_threshold) + '_' + str(threshold) + '.pnml')
        base_dfg, base_start_activities, base_end_activities = pm4py.discover_dfg(
            base_event_log)
        if config.output:
            pm4py.save_vis_dfg(base_dfg, base_start_activities, base_end_activities,
                               config.output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                   threshold) + '_heuristic/petri_nets_' + sampling_algo + '/base_net_' + str(
                                   noise_threshold) + '_' + str(threshold) + '.png')
    return base_net, base_im, base_fm


def conformance_checking(log, base_net, base_im, base_fm, sample_key, token):
    """

    :param log:
    :param base_net:
    :param base_im:
    :param base_fm:
    :param sample_key:
    :param all_or_tok:
    :return:
    """
    # track start time
    start_time = time.time()
    if token:
        all_or_tok = "token"
    else:
        all_or_tok = "alignment"

    if all_or_tok == "token":
        fitness = pm4py.fitness_token_based_replay(log, base_net, base_im, base_fm)['log_fitness']
        precision = pm4py.precision_token_based_replay(log, base_net, base_im, base_fm)
        method = 'TBR'
    elif all_or_tok == "alignment":
        try:
            fitness = pm4py.fitness_alignments(log, base_net, base_im, base_fm)['log_fitness']
            precision = pm4py.precision_alignments(log, base_net, base_im, base_fm)
            method = 'alignments'
        except Exception as e:
            print('catch: nust switch to token based replay')
            if str(e) == "trying to apply alignments on a Petri net that is not a easy sound net!!":
                fitness = pm4py.fitness_token_based_replay(log, base_net, base_im, base_fm)['log_fitness']
                precision = pm4py.precision_token_based_replay(log, base_net, base_im, base_fm)
                method = 'TBR_e'
            else:
                print("Caught a different exception:", e)
    else:
        raise Exception("As conformance checking methods must be \"alignment\" or \"token\" set.")

    gen = generalization_evaluator.apply(log, base_net, base_im, base_fm)
    simp = simplicity_evaluator.apply(base_net)
    # catch the case that precision + fitness == 0 and for the f1-score a division in needed
    if precision + fitness == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * fitness) / (precision + fitness)
    # calculate runtime
    # track the end time of the proces mining and eval
    end_time = time.time()
    # calculate and print the processing times
    runtime = round(end_time - start_time,4)
    new_rows = pd.DataFrame(
        {'file': [sample_key], 'f1': [f1], 'fitness': [fitness], 'precision': [precision],
         'generalisation': [gen], 'simplicity': [simp], 'method': [method], 'runtime': [runtime]})

    return new_rows


def pm_with_cc_for_samples(base_event_log, event_log_dic, sampling_algo):
    """

    :param base_event_log:
    :param event_log_dic:
    :param sampling_algo:
    :return:
    """
    output_path = config.output_path
    qualitys_dataframe = {}
    for noise_threshold in config.noise_threshold_list:
        qualitys_dataframe_noise_th = {}
        for threshold in [0]:
            # for path in config.path_list:
            if base_event_log is None:
                raise ValueError("base_event_log need to be specified")
                # base_event_log = load_base_csv_data(path[:path.rfind('\\')], threshold)
            else:
                base_event_log = format_base_csv_data(base_event_log, threshold)

            # print(base_event_log)
            if event_log_dic is None:
                raise ValueError("event_log_dic need to be specified")
                # event_log_dic = load_csv_data(path, threshold)
            else:
                event_log_dic = format_event_data(event_log_dic, threshold)

            # print(event_log_dic)
            result = {}
            for miner in ["inductive miner", "heuristic miner"]:
                base_net, base_im, base_fm = process_mining(base_event_log, miner, noise_threshold, threshold,
                                                            sampling_algo)


                CC_results_dataframe = pd.DataFrame(
                    columns=['file', 'f1', 'fitness', 'precision', 'generalisation', 'simplicity'])
                # conformance checking with samples
                for sample_key in event_log_dic:


                    # if sampling_algo == "Guided_Conformance_Sampling":
                    #     event_log_dic[sample_key] = external_sampling.Guided_Conformance_Sampling_interface.guided_conformance_sampling(base_event_log, base_net, base_im, base_fm, sample_key)

                    new_rows = conformance_checking(event_log_dic[sample_key], base_net, base_im, base_fm, sample_key, config.only_token_based)

                    if CC_results_dataframe.empty:
                        # CC_results_dataframe = {miner: new_rows}
                        CC_results_dataframe = new_rows
                    else:
                        CC_results_dataframe = pd.concat([CC_results_dataframe, new_rows], ignore_index=True)

                # conformance checking with base log
                new_rows = conformance_checking(base_event_log, base_net, base_im, base_fm, "base log", config.only_token_based)
                CC_results_dataframe = pd.concat([CC_results_dataframe, new_rows], ignore_index=True)
                CC_results_dataframe = CC_results_dataframe.iloc[::-1]
                CC_results_dataframe = CC_results_dataframe.reset_index(drop=True)
                result[miner] = CC_results_dataframe
                if config.output:
                    CC_results_dataframe.to_csv(config.output_path + "quality_metrics_" + miner + "_threshold_"+ str(noise_threshold) +".csv", index=False,
                                                sep=";")
            qualitys_dataframe_noise_th["rare_activ_th_" + str(threshold)] = result
        qualitys_dataframe["miner_th_" + str(noise_threshold)] = qualitys_dataframe_noise_th

    return qualitys_dataframe


def format_event_data(event_logs, filter_threshold):
    """

    :param event_logs:
    :param filter_threshold:
    :return:
    """

    event_log_dic = {}

    for key in event_logs:
        dataframe = event_logs[key]
        dataframe = pm4py.format_dataframe(dataframe, case_id='case_ID', activity_key='activity',
                                           timestamp_key='timestamp',
                                           timest_format=config.ab_format_timestamp_conversion_to_datetime_obj)
        if not filter_threshold == 0:
            dataframe = delete_rare_activities(dataframe, filter_threshold)
        # pm4py need the case id to be a string
        dataframe['case_ID'] = dataframe['case_ID'].astype(str)
        event_log = dataframe
        event_log_dic[key] = event_log
    return event_log_dic


def delete_rare_activities(event_log, threshold):
    """

    :param event_log:
    :param threshold:
    :return:
    """
    activities = event_log["activity"].value_counts()
    length = event_log.size
    highes_occorance = activities.iloc[0]
    lowest_allows = highes_occorance * threshold
    no_activities = activities.drop(activities[activities < lowest_allows].index)
    activities = activities.drop(activities[activities > lowest_allows].index)
    activities = activities.index.tolist()
    event_log = event_log.drop(event_log[event_log["activity"].isin(activities)].index)

    return event_log


def format_base_csv_data(dataframe, filter_threshold):
    """

    :param dataframe:
    :param filter_threshold:
    :return:
    """

    dataframe = pm4py.format_dataframe(dataframe, case_id='case_ID', activity_key='activity', timestamp_key='timestamp',
                                       timest_format=config.ab_format_timestamp_conversion_to_datetime_obj)
    if not filter_threshold == 0:
        dataframe = delete_rare_activities(dataframe, filter_threshold)
    dataframe['case_ID'] = dataframe['case_ID'].astype(str)
    event_log = dataframe
    return event_log
