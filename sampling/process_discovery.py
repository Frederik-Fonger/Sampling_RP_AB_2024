import copy
import os
import glob
import time
import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from ConfigManager import Config_Manager as config
from datetime import datetime
import inspect

# for Windows
mpl.use('Qt5Agg')


# for macOS
# mpl.use('macosx')


def run_model_creation_and_eval(base_event_log, event_log_dic, algo):

    """
    run the whole model with all algorithms and ratios
    :param base_event_log: general event log which wasn't sampled
    :param event_log_dic: dictionary of the different sampled event logs with their ratio
    :param algo: list of algorithms which are used in the model
    :return:
    """

    def load_csv_data(file_path, filter_threshold):

        """
        loading the csv data and generating an event log dictionary
        :param file_path: the path of the csv or xes file
        :param filter_threshold:
        :return:
        """
        os_path = os.getcwd() + file_path
        all_files = glob.glob(os.path.join(os_path, '*.{}'.format('csv')))

        event_log_dic = {}

        for filename in all_files:
            if filename.endswith('.csv'):
                file_path = os.path.join(os_path, filename)
                dataframe = pd.read_csv(file_path)
                dataframe = pm4py.format_dataframe(dataframe, case_id='case_ID', activity_key='activity',
                                                   timestamp_key='timestamp',
                                                   timest_format=config.ab_format_timestamp_conversion_to_datetime_obj)
                if not filter_threshold == 0:
                    dataframe = delete_rare_activities(dataframe, filter_threshold)
                event_log = dataframe
                event_log_dic[filename] = event_log
        return event_log_dic

    def format_event_data(event_logs, filter_threshold):

        """
        format the event log
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

    def format_base_csv_data(dataframe, filter_threshold):
        """
        formatting a csv dataframe to a eventlog
        :param dataframe:
        :param filter_threshold:
        :return:
        """
        dataframe = pm4py.format_dataframe(dataframe, case_id='case_ID', activity_key='activity',
                                           timestamp_key='timestamp',
                                           timest_format=config.ab_format_timestamp_conversion_to_datetime_obj)
        if not filter_threshold == 0:
            dataframe = delete_rare_activities(dataframe, filter_threshold)
        dataframe['case_ID'] = dataframe['case_ID'].astype(str)
        event_log = dataframe
        return event_log

    def calculate_quality_criteria(event_log_dic, base_event_log, inductive_miner, token, noise_threshold):
        """
        calculating the different quality criterias f1, fitness, precision, generalisation and simplicity
        :param event_log_dic:
        :param base_event_log:
        :param inductive_miner:
        :param token:
        :param noise_threshold:
        :return:
        """
        quality_dataframe = pd.DataFrame(columns=['file', 'f1', 'fitness', 'precision', 'generalisation', 'simplicity'])
        if inductive_miner:
            if config.output:
                os.makedirs(output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                    threshold) + '_inductive/petri_nets_' + algo)
            base_net, base_im, base_fm = pm4py.discover_petri_net_inductive(base_event_log, case_id_key='case_ID',
                                                                            activity_key='activity',
                                                                            timestamp_key='timestamp',
                                                                            noise_threshold=noise_threshold)
            if config.output:
                pm4py.save_vis_petri_net(base_net, base_im, base_fm,
                                         output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                             threshold) + '_inductive/petri_nets_' + algo + '/base_dfg_' + str(
                                             noise_threshold) + '_' + str(threshold) + '.png')
            base_dfg, base_start_activities, base_end_activities = pm4py.discover_dfg(
                base_event_log)
            if config.output:
                pm4py.save_vis_dfg(base_dfg, base_start_activities, base_end_activities,
                                   output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                       threshold) + '_inductive/petri_nets_' + algo + '/base_net_' + str(
                                       noise_threshold) + '_' + str(threshold) + '.png')

        else:
            if config.output:
                os.makedirs(output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                    threshold) + '_heuristic/petri_nets_' + algo)
            base_net, base_im, base_fm = pm4py.discover_petri_net_heuristics(base_event_log, case_id_key='case_ID',
                                                                             activity_key='activity',
                                                                             timestamp_key='timestamp',
                                                                             dependency_threshold=noise_threshold)
            os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
            if config.output:
                pm4py.save_vis_petri_net(base_net, base_im, base_fm,
                                         output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                             threshold) + '_heuristic/petri_nets_' + algo + '/base_dfg_' + str(
                                             noise_threshold) + '_' + str(threshold) + '.png')
            base_dfg, base_start_activities, base_end_activities = pm4py.discover_dfg(
                base_event_log)
            if config.output:
                pm4py.save_vis_dfg(base_dfg, base_start_activities, base_end_activities,
                                   output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                       threshold) + '_heuristic/petri_nets_' + algo + '/base_net_' + str(
                                       noise_threshold) + '_' + str(threshold) + '.png')
        for filename, event_log in event_log_dic.items():
            filename = "sample " + str(filename)
            if inductive_miner:
                net, im, fm = pm4py.discover_petri_net_inductive(event_log, case_id_key='case_ID',
                                                                 activity_key='activity',
                                                                 timestamp_key='timestamp',
                                                                 noise_threshold=noise_threshold)
                dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
                if config.output:
                    pm4py.save_vis_dfg(dfg, start_activities, end_activities,
                                       output_path + '/noise_' + str(
                                           noise_threshold) + '_threshold_' + str(
                                           threshold) + '_inductive/petri_nets_' + algo + '/' + str(
                                           filename) + '_dfg_' + str(
                                           noise_threshold) + '_' + str(threshold) + '.png')

                    pm4py.save_vis_petri_net(net, im, fm,
                                             output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                                 threshold) + '_inductive/petri_nets_' + algo + '/' + str(
                                                 filename) + '_petri_net_' + str(
                                                 noise_threshold) + '_' + str(
                                                 threshold) + '.png')
            else:
                net, im, fm = pm4py.discover_petri_net_heuristics(event_log, case_id_key='case_ID',
                                                                  activity_key='activity',
                                                                  timestamp_key='timestamp',
                                                                  dependency_threshold=noise_threshold)
                dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
                if config.output:
                    pm4py.save_vis_dfg(dfg, start_activities, end_activities,
                                       output_path + '/noise_' + str(
                                           noise_threshold) + '_threshold_' + str(
                                           threshold) + '_heuristic/petri_nets_' + algo + '/' + str(
                                           filename) + '_dfg_' + str(
                                           noise_threshold) + '_' + str(threshold) + '.png')
                    pm4py.save_vis_petri_net(net, im, fm,
                                             output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                                                 threshold) + '_heuristic/petri_nets_' + algo + '/' + str(
                                                 filename) + '_petri_net_' + str(
                                                 noise_threshold) + '_' + str(
                                                 threshold) + '.png')

            # conformance checking

            if token:
                fitness = pm4py.fitness_token_based_replay(base_event_log, net, im, fm)["log_fitness"]
                precision = pm4py.precision_token_based_replay(base_event_log, net, im, fm)
                method = 'token based replay'
            else:
                try:
                    fitness = pm4py.fitness_alignments(base_event_log, net, im, fm)['log_fitness']
                    precision = pm4py.precision_alignments(base_event_log, net, im, fm)
                    method = 'alignments'
                except Exception as e:
                    print('catch: nust switch to token based replay')
                    if str(e) == "trying to apply alignments on a Petri net that is not a easy sound net!!":
                        fitness = pm4py.fitness_token_based_replay(base_event_log, net, im, fm)['log_fitness']
                        precision = pm4py.precision_token_based_replay(base_event_log, net, im, fm)
                        method = 'token based replay e'
                    else:
                        print("Caught a different exception:", e)
            if config.only_f1_score:
                gen = -1
                simp = -1
            else:
                gen = generalization_evaluator.apply(base_event_log, net, im, fm)
                simp = simplicity_evaluator.apply(net)

            # catch the case that precision + fitness == 0 and for the f1-score a division in needed
            if precision + fitness == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * fitness) / (precision + fitness)
            new_rows = pd.DataFrame(
                {'file': [filename], 'f1': [f1], 'fitness': [fitness], 'precision': [precision],
                 'generalisation': [gen], 'simplicity': [simp], 'method': [method]})
            if quality_dataframe.empty:
                quality_dataframe = new_rows
            else:
                quality_dataframe = pd.concat([quality_dataframe, new_rows], ignore_index=True)

        # conformance checking for base net and base log
        if token:
            fitness = pm4py.fitness_token_based_replay(base_event_log, base_net, base_im, base_fm)["log_fitness"]
            precision = pm4py.precision_token_based_replay(base_event_log, base_net, base_im, base_fm)
            method = 'token based replay'
        else:
            try:
                fitness = pm4py.fitness_alignments(base_event_log, base_net, base_im, base_fm)['log_fitness']
                precision = pm4py.precision_alignments(base_event_log, base_net, base_im, base_fm)
                method = 'alignments'
            except Exception as e:
                print('catch: must switch to token based replay')
                if str(e) == "trying to apply alignments on a Petri net that is not a easy sound net!!":
                    fitness = pm4py.fitness_token_based_replay(base_event_log, base_net, base_im, base_fm)[
                        'log_fitness']
                    precision = pm4py.precision_token_based_replay(base_event_log, base_net, base_im, base_fm)
                    method = 'token based replay'
                else:
                    print("Caught a different exception:", e)
        if config.only_f1_score:
            gen = -1
            simp = -1
        else:
            gen = generalization_evaluator.apply(base_event_log, base_net, base_im, base_fm)
            simp = simplicity_evaluator.apply(base_net)
        # catch the case that precision + fitness == 0 and for the f1-score a division in needed
        if precision + fitness == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * fitness) / (precision + fitness)
        new_rows = pd.DataFrame(
            {'file': ['base_event_log'], 'f1': [f1], 'fitness': [fitness], 'precision': [precision],
             'generalisation': [gen], 'simplicity': [simp], 'method': [method]})
        if quality_dataframe.empty:
            quality_dataframe = new_rows
        else:
            quality_dataframe = pd.concat([quality_dataframe, new_rows], ignore_index=True)
        quality_dataframe = quality_dataframe.iloc[::-1]
        return quality_dataframe

    def box_plot_quality_dataframe(quality_dataframe, noise_threshold, inductive_miner):

        """
        generating a bar graph comparing the different ratios by quality criteria
        :param quality_dataframe:
        :param noise_threshold:
        :param inductive_miner:
        :return:
        """
        quality_dataframe = copy.deepcopy(quality_dataframe)
        plt.clf()
        if not (quality_dataframe['method'] == 'alignments').all():
            quality_dataframe['file'] = quality_dataframe['file'] + ' (' + quality_dataframe['method'] + ')'
        quality_dataframe = quality_dataframe.drop(columns=['method'])

        # set width of bars
        bar_width = 0.2

        # set positions of bars
        bar_positions1 = range(len(quality_dataframe))
        bar_positions2 = [pos + bar_width for pos in bar_positions1]
        bar_positions3 = [pos + bar_width * 2 for pos in bar_positions1]
        bar_positions4 = [pos + bar_width * 3 for pos in bar_positions1]

        # generate bar graph
        plt.bar(bar_positions1, quality_dataframe['fitness'], width=bar_width, label='fitness')
        plt.bar(bar_positions2, quality_dataframe['precision'], width=bar_width, label='precision')
        plt.bar(bar_positions3, quality_dataframe['generalisation'], width=bar_width, label='generalisation')
        plt.bar(bar_positions4, quality_dataframe['simplicity'], width=bar_width, label='simplicity')

        # labeling the chart
        plt.xlabel('file')
        plt.ylabel('quality criteria')
        plt.title('Bar Chart of ' + algo)
        plt.legend(loc="upper left")

        plt.xticks([pos + bar_width for pos in bar_positions1], quality_dataframe['file'], rotation=45, ha='right')

        plt.tight_layout()
        if config.output:
            if inductive_miner:
                plt.savefig(output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                    threshold) + '_inductive/' + str(noise_threshold) + '_' + str(threshold) + '_Bar_Chart_' +
                            algo + '.png')
            else:
                plt.savefig(output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                    threshold) + '_heuristic/' + str(noise_threshold) + '_' + str(threshold) + '_Bar_Chart_' +
                            algo + '.png')

    def curve_plot_quality_dataframe(quality_dataframe, noise_threshold, inductive_miner):
        """
        generating a curve graph comparing the different ratios by quality criteria
        :param quality_dataframe:
        :param noise_threshold:
        :param inductive_miner:
        :return:
        """
        quality_dataframe = copy.deepcopy(quality_dataframe)
        plt.clf()
        if not (quality_dataframe['method'] == 'alignments').all():
            quality_dataframe['file'] = quality_dataframe['file'] + ' (' + quality_dataframe['method'] + ')'
        # quality_dataframe = quality_dataframe.drop(columns=['method'])

        # setting line width
        line_width = 2

        # generating curve graph
        plt.plot(range(len(quality_dataframe)), quality_dataframe['fitness'], label='fitness', linewidth=line_width,
                 color='blue')
        plt.plot(range(len(quality_dataframe)), quality_dataframe['precision'], label='precision', linewidth=line_width,
                 color='orange')
        plt.plot(range(len(quality_dataframe)), quality_dataframe['generalisation'], label='generalisation',
                 linewidth=line_width, color='green')
        plt.plot(range(len(quality_dataframe)), quality_dataframe['simplicity'], label='simplicity',
                 linewidth=line_width, color='red')

        # labeling the graph
        plt.xlabel('file')
        plt.ylabel('quality criteria')
        plt.title('Curve Chart of ' + algo)
        plt.legend(loc="upper left")

        plt.xticks(range(len(quality_dataframe)), quality_dataframe['file'], rotation=45, ha='right')

        if config.output:
            if inductive_miner:
                plt.savefig(output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                    threshold) + '_inductive/' + str(noise_threshold) + '_' + str(threshold) + '_Curve_Chart_' +
                            algo + '.png')
            else:
                plt.savefig(output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                    threshold) + '_heuristic/' + str(noise_threshold) + '_' + str(threshold) + '_Curve_Chart_' +
                            algo + '.png')

    def table_plot_quality_dataframe(quality_dataframe, noise_threshold, inductive_miner):
        """
        generating a heatmap of the algorithms
        :param quality_dataframe:
        :param noise_threshold:
        :param inductive_miner:
        :return:
        """
        quality_dataframe = copy.deepcopy(quality_dataframe)
        plt.clf()
        plt.figure(figsize=(8, 6))
        # print(quality_dataframe['method'])
        if not (quality_dataframe['method'] == 'alignments').all():
            quality_dataframe['file'] = quality_dataframe['file'] + ' (' + quality_dataframe['method'] + ')'
            # print(quality_dataframe['file'])
        quality_dataframe = quality_dataframe.drop(columns=['method'])
        sns.heatmap(quality_dataframe.set_index('file'), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
        plt.title("Heatmap of " + algo)
        plt.tight_layout()
        if config.output:
            if inductive_miner:
                plt.savefig(output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                    threshold) + '_inductive/' + str(noise_threshold) + '_' + str(threshold) + '_Heat_Map_' +
                            algo + '.png')
            else:
                plt.savefig(output_path + '/noise_' + str(noise_threshold) + '_threshold_' + str(
                    threshold) + '_heuristic/' + str(noise_threshold) + '_' + str(threshold) + '_Heat_Map_' +
                            algo + '.png')
        # plt.show()

    def delete_rare_activities(event_log, threshold):
        """
        drops all activities of an eventlog which are under a set threshold
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

    # get starting time
    # start_time = time.time()

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

            # calculation of heuristic miner
            quality_dataframe_hm = calculate_quality_criteria(event_log_dic, base_event_log, False, config.only_token_based,
                                                              noise_threshold)
            # print(quality_dataframe)
            if config.output:
                quality_dataframe_hm.to_csv(config.output_path + "quality_metrics_heuristic_threshold_" + str(noise_threshold) +".csv", index=False, sep=";")
            box_plot_quality_dataframe(quality_dataframe_hm, noise_threshold, False)
            curve_plot_quality_dataframe(quality_dataframe_hm, noise_threshold, False)
            table_plot_quality_dataframe(quality_dataframe_hm, noise_threshold, False)

            # calculation of inductive miner
            quality_dataframe_im = calculate_quality_criteria(event_log_dic, base_event_log, True, config.only_token_based,
                                                              noise_threshold)
            # print(quality_dataframe)
            if config.output:
                quality_dataframe_im.to_csv(config.output_path + "quality_metrics_inductive_threshold_" + str(noise_threshold) +".csv", index=False, sep=";")
            box_plot_quality_dataframe(quality_dataframe_im, noise_threshold, True)
            curve_plot_quality_dataframe(quality_dataframe_im, noise_threshold, True)
            table_plot_quality_dataframe(quality_dataframe_im, noise_threshold, True)

            qualitys_dataframe_noise_th["rare_activ_th_" + str(threshold)] = {"heuristic miner": quality_dataframe_hm,
                                                                              "inductive miner": quality_dataframe_im}

        qualitys_dataframe["miner_th_" + str(noise_threshold)] = qualitys_dataframe_noise_th
    return qualitys_dataframe
