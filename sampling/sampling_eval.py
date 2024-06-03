import copy
import os

import numpy as np
import pandas as pd
import pm4py
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from ConfigManager import Config_Manager as config

import pm_utils


def sample_coverage(base_and_sample_DFR):
    """
    calculate the coverage of the samples.
    :param base_and_sample_DFR: The DFRs from the base log and the samples
    :return:
    """

    # count DFRs in base log
    total_base_DFR_count = base_and_sample_DFR.shape[0]

    # Count non-zero values in each column
    total_sample_DFR_count = (base_and_sample_DFR['sample count'] != 0).sum()

    return total_sample_DFR_count/total_base_DFR_count

def sample_mean_absolute_error_representativeness_and_more(base_and_sample_DFR, sample_metrics_results):
    """
    calculated the MAE, NMAE, EMSE, NRMSE MAPE on the bases of the DFR counts
    :param base_and_sample_DFR:
    :param sample_metrics_results:
    :return:
    """

    # get the true ratios and the sample ratios (predicted)
    ratio_sample = base_and_sample_DFR['sample count']

    # create an array of 1´s (target value for the representaviness) with the same length as the "ratio_sample"-array
    # ratio_true = np.ones(len(ratio_sample))

    #
    ratio_ideal = base_and_sample_DFR['ideal sample count']

    # use sklearn for the MAE
    sample_metrics_results["MAE"] = mean_absolute_error(ratio_ideal, ratio_sample)
    sample_metrics_results["NMAE"] = mean_absolute_error(ratio_ideal, ratio_sample) / np.mean(ratio_ideal)
    sample_metrics_results["RMSE"] = root_mean_squared_error(ratio_ideal, ratio_sample)
    sample_metrics_results["NRMSE"] = root_mean_squared_error(ratio_ideal, ratio_sample) / np.mean(ratio_ideal)
    sample_metrics_results["MAPE"] = mean_absolute_percentage_error(ratio_ideal, ratio_sample)
    # sample_metrics_results["sRMSPE"] = mean_absolute_percentage_error(ratio_ideal, ratio_sample, square_root=True, symmetric=True)


def calculate_percentages_of_behavior(base_and_sample_DFR, sample_metrics_results):
    """
    Calculates the mean sample ratio, standard deviation, percentage of unsampled DFRs, percentage of undersampled
    DFRs, percentage of oversampled DFRs, percentage of truly sampled DFRs and adds them to the sample_metrics_resuls
    dict.
    :param base_and_sample_DFR:
    :param sample_metrics_results:
    :return:
    """

    dfr_count = len(base_and_sample_DFR['DFR'])
    sample_metrics_results["mean sample ratio"] = np.mean(base_and_sample_DFR['sample count / base count'])
    sample_metrics_results["standard deviation"] = np.std(base_and_sample_DFR['sample count / base count'])

    sample_metrics_results['percentage of unsampled DFRs'] = (base_and_sample_DFR['sample DFR ratio / base DFR ratio'] == 0).sum() / dfr_count
    sample_metrics_results['percentage of undersampled DFRs'] =((base_and_sample_DFR['sample DFR ratio / base DFR ratio'] > 0) & (base_and_sample_DFR['sample DFR ratio / base DFR ratio'] < 1)).sum() / dfr_count
    sample_metrics_results['percentage of oversampled DFRs'] = (base_and_sample_DFR['sample DFR ratio / base DFR ratio'] > 1).sum() / dfr_count
    sample_metrics_results['percentage of truly sampled DFRs'] = (base_and_sample_DFR['sample DFR ratio / base DFR ratio'] == 1).sum() / dfr_count
    sample_metrics_results['percentage of DFRs within 1 % of ideal'] = ((base_and_sample_DFR['sample DFR ratio / base DFR ratio'] > 0.99) & (base_and_sample_DFR['sample DFR ratio / base DFR ratio'] < 1.01)).sum() / dfr_count
    sample_metrics_results['percentage of DFRs within 5 % of ideal'] = ((base_and_sample_DFR['sample DFR ratio / base DFR ratio'] > 0.95) & (base_and_sample_DFR['sample DFR ratio / base DFR ratio'] < 1.05)).sum() / dfr_count
    sample_metrics_results['percentage of DFRs within 10 % of ideal'] = ((base_and_sample_DFR['sample DFR ratio / base DFR ratio'] > 0.9) & (base_and_sample_DFR['sample DFR ratio / base DFR ratio'] < 1.1)).sum() / dfr_count



def get_DFR(sample, base_log, sample_ratio):
    """
    extracts all DFRs from the given log
    :param base_log:

    :return:
        This is the Dataframe with all DFR Information. The Columns are:
            'DFR': Name of the two Activities are build the DFR
            'base log count': count how often the activity is in the base log
            'base DFR ratio': Displays the ratio of this DFR in the base log to the total count of DFRs in the base log
            'sample count': count how often the activity is in the sample
            'sample DFR ratio (sample count / total DFR count)':  Displays the ratio of this DFR in the sample to the
                total count of DFRs in the sample
            'sample DFR ratio - base DFR ratio': Shows the difference between the sample ratio of a DFR and the goal
                ratio of the DFR from the base log
            'sample DFR ratio / base DFR ratio': Shows the representativeness for each DFR
            'sample count / base count': Shows the sample ratio of this DFR. The goal is to reach in every row the set
                sample ratio of the sample
    """

    # prepare DFRs from base log
    # get DFRs via pm4py
    dfg, start_activities, end_activities = pm4py.discover_dfg(base_log)
    base_DFR = pd.DataFrame(dfg.items(), columns=["DFR", 'base log count'])

    # calculate base information for DFRs
    dfr_count_base_log = base_DFR["base log count"].sum()
    base_DFR["base DFR ratio"] = base_DFR["base log count"]/dfr_count_base_log
    base_DFR['ideal sample count'] = base_DFR["base log count"] * sample_ratio

    # prepare DFRs from sample
    # get DFRs via pm4py
    dfg_sample, start_activities, end_activities = pm4py.discover_dfg(sample, activity_key=config.ab_activity_column)
    sample_DFR = pd.DataFrame(dfg_sample.items(), columns=["DFR", 'sample count'])

    # calculate base information for DFRs
    dfr_count_sample = sample_DFR["sample count"].sum()
    sample_DFR['sample DFR ratio (sample count / total DFR count)'] = sample_DFR["sample count"]/dfr_count_sample

    # calculate base ratios and metrics that are usefully for later calculating the quality metrics
    base_and_sample_DFR = pd.merge(base_DFR, sample_DFR, on='DFR', how='left')
    base_and_sample_DFR['sample count'] = base_and_sample_DFR['sample count'].fillna(0)
    base_and_sample_DFR['sample DFR ratio (sample count / total DFR count)'] = base_and_sample_DFR['sample DFR ratio (sample count / total DFR count)'].fillna(0)
    base_and_sample_DFR['sample DFR ratio - base DFR ratio'] = base_and_sample_DFR['sample DFR ratio (sample count / total DFR count)'] - base_and_sample_DFR['base DFR ratio']
    base_and_sample_DFR['sample DFR ratio / base DFR ratio'] = base_and_sample_DFR['sample DFR ratio (sample count / total DFR count)'] / base_and_sample_DFR['base DFR ratio']
    base_and_sample_DFR['sample count / base count'] = base_and_sample_DFR['sample count'] / base_and_sample_DFR['base log count']
    return base_and_sample_DFR


def calculate_emd(sample, base_log):

    language_sample = pm4py.get_stochastic_language(sample)
    language_base_log = pm4py.get_stochastic_language(base_log)

    emd_distance = pm4py.compute_emd(language_sample, language_base_log)


    return emd_distance


def samples_eval(samples, base_log, algo):
    """
    Function for managing the evaluation of the samples.
    :param samples:
    :param base_log:
    :param algo:
    :return:
    """

    samples_metrics_results = {}
    dfts = {}
    # iterate over the sample of one algo
    for sample in samples:

        sample_metrics_results = {}
        base_and_sample_DFR = get_DFR(samples[sample], base_log, sample)
        # sample_metrics_results['DFRs'] = base_and_sample_DFR

        # extract the DFRs once, instead of doing it in every funktion newly
        sample_metrics_results["coverage"] = sample_coverage(base_and_sample_DFR)
        # print("start emd")
        # sample_metrics_results["emd"] = calculate_emd(samples[sample], base_log)
        sample_mean_absolute_error_representativeness_and_more(base_and_sample_DFR, sample_metrics_results)
        calculate_percentages_of_behavior(base_and_sample_DFR, sample_metrics_results)
        samples_metrics_results[sample] = sample_metrics_results
        dfts[sample] = base_and_sample_DFR

    return samples_metrics_results, dfts


def plot_coverage(samples_metrics_results):
    fig, ax1 = plt.subplots()
    # algo_names = ["AllBehavior", "RemainderPlus", "C-min", "Random", "Stratified"]
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['coverage'] for data_point in
                           samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('coverage')
    plt.title('Curve Chart of ' + 'coverage' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'coverage' + '.png',
                    bbox_inches='tight')
    plt.clf()


def plot_nmae(samples_metrics_results):
    fig, ax1 = plt.subplots()
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['NMAE'] for data_point in
                           samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('NMAE')
    plt.title('Curve Chart of ' + 'NMAE' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'NMAE' + '.png',
                    bbox_inches='tight')
    plt.clf()
def plot_rmse(samples_metrics_results):
    fig, ax1 = plt.subplots()
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['RMSE'] for data_point in
                           samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('RMSE')
    plt.title('Curve Chart of ' + 'RMSE' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'RMSE' + '.png',
                    bbox_inches='tight')
    plt.clf()


def plot_nrmse(samples_metrics_results):
    fig, ax1 = plt.subplots()
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['NRMSE'] for data_point in
                           samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('NRMSE')
    plt.title('Curve Chart of ' + 'NRMSE' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'NRMSE' + '.png',
                    bbox_inches='tight')
    plt.clf()


def plot_mape(samples_metrics_results):
    fig, ax1 = plt.subplots()
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['MAPE'] for data_point in
                           samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('MAPE')
    plt.title('Curve Chart of ' + 'MAPE' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'MAPE' + '.png',
                    bbox_inches='tight')
    plt.clf()


def plot_mean_sample_ratio(samples_metrics_results):
    fig, ax1 = plt.subplots()
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['mean sample ratio'] for data_point in
                           samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('mean sample ratio')
    plt.title('Curve Chart of ' + 'mean sample ratio' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'mean sample ratio' + '.png',
                    bbox_inches='tight')
    plt.clf()


def plot_percentage_of_unsample_dfrs(samples_metrics_results):
    fig, ax1 = plt.subplots()
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['percentage of unsampled DFRs'] for data_point in
                           samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('percentage of unsampled DFRs')
    plt.title('Curve Chart of ' + 'percentage of unsampled DFRs' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)


    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'percentage of unsampled DFRs' + '.png',
                    bbox_inches='tight')
    plt.clf()


def plot_percentage_of_undersample_dfrs(samples_metrics_results):
    fig, ax1 = plt.subplots()
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['percentage of undersampled DFRs'] for data_point in
                           samples_metrics_results[algo].values()]


        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('percentage of undersampled DFRs')
    plt.title('Curve Chart of ' + 'percentage of undersampled DFRs' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'percentage of undersampled DFRs' + '.png',
                    bbox_inches='tight')
    plt.clf()



def plot_percentage_of_oversample_dfrs(samples_metrics_results):
    fig, ax1 = plt.subplots()
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['percentage of oversampled DFRs'] for data_point in
                           samples_metrics_results[algo].values()]


        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('percentage of oversampled DFRs')
    plt.title('Curve Chart of ' + 'percentage of oversampled DFRs' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'percentage of oversampled DFRs' + '.png',
                    bbox_inches='tight')
    plt.clf()


def plot_percentage_of_truly_sample_dfrs(samples_metrics_results):
    fig, ax1 = plt.subplots()
    #MAE
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['percentage of truly sampled DFRs'] for data_point in samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('percentage of truly sampled DFRs')
    plt.title('Curve Chart of ' + 'percentage of truly sampled DFRs' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'percentage of truly sampled DFRs' + '.png', bbox_inches='tight')
    plt.clf()

def plot_percentage_of_dfrs_within_1(samples_metrics_results):
    fig, ax1 = plt.subplots()
    #MAE
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['percentage of DFRs within 1 % of ideal'] for data_point in samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('percentage of DFRs within 1 % of ideal')
    plt.title('Curve Chart of ' + 'percentage of DFRs within 1 % of ideal' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'percentage of DFRs within 1 percentage of ideal' + '.png', bbox_inches='tight')
    plt.clf()

def plot_percentage_of_dfrs_within_5(samples_metrics_results):
    fig, ax1 = plt.subplots()
    #MAE
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['percentage of DFRs within 5 % of ideal'] for data_point in samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('percentage of DFRs within 5 % of ideal')
    plt.title('Curve Chart of ' + 'percentage of DFRs within 5 % of ideal' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'percentage of DFRs within 5 percentage of ideal' + '.png', bbox_inches='tight')
    plt.clf()

def plot_percentage_of_dfrs_within_10(samples_metrics_results):
    fig, ax1 = plt.subplots()
    #MAE
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['percentage of DFRs within 10 % of ideal'] for data_point in samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('percentage of DFRs within 10 % of ideal')
    plt.title('Curve Chart of ' + 'percentage of DFRs within 10 % of ideal' + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'percentage of DFRs within 10 percentage of ideal' + '.png', bbox_inches='tight')
    plt.clf()

def plot_mae(samples_metrics_results):
    fig, ax1 = plt.subplots()
    #MAE
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['MAE'] for data_point in samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)
        # plt.plot(config.ab_sample_ratio, mae_values_list, label=algo,)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('MAE')
    plt.title('Curve Chart of ' + "MAE" + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'MAE' + '.png', bbox_inches='tight')
    plt.clf()

def plot_emd(samples_metrics_results):
    fig, ax1 = plt.subplots()
    #MAE
    for algo in samples_metrics_results.keys():
        mae_values_list = [data_point['emd'] for data_point in samples_metrics_results[algo].values()]

        ax1.plot(range(len(config.ab_sample_ratio)), mae_values_list, label=algo)
        # plt.plot(config.ab_sample_ratio, mae_values_list, label=algo,)

    # labeling the curve chart
    plt.xlabel('sample ratio')
    plt.ylabel('emd')
    plt.title('Curve Chart of ' + "emd" + "\n" + config.source_path)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1.xaxis.set_ticks(range(len(config.ab_sample_ratio)))
    ax1.xaxis.set_ticklabels(config.ab_sample_ratio)

    if config.output:
        plt.savefig(config.output_path + '/graphs/' + 'Curve_Chart_' + 'emd' + '.png', bbox_inches='tight')
    plt.clf()


def vis_results(samples_metrics_results):
    """
    This function executes all visualisation functions for plotting the quality metrics.
    :param samples_metrics_results:
    :return:
    """

    plot_coverage(samples_metrics_results)
    plot_mae(samples_metrics_results)
    plot_nmae(samples_metrics_results)
    plot_rmse(samples_metrics_results)
    plot_nrmse(samples_metrics_results)
    plot_mape(samples_metrics_results)
    plot_mean_sample_ratio(samples_metrics_results)
    plot_percentage_of_unsample_dfrs(samples_metrics_results)
    plot_percentage_of_undersample_dfrs(samples_metrics_results)
    plot_percentage_of_oversample_dfrs(samples_metrics_results)
    plot_percentage_of_truly_sample_dfrs(samples_metrics_results)
    plot_percentage_of_dfrs_within_1(samples_metrics_results)
    plot_percentage_of_dfrs_within_5(samples_metrics_results)
    plot_percentage_of_dfrs_within_10(samples_metrics_results)
    # plot_emd(samples_metrics_results)


def save_results(samples_metrics_results, dfts, samples):
    """
    This function saves the quality metrics as results
    :param samples_metrics_results:
    :param dfts:
    :param samples:
    :return:
    """

    samples_metrics_results = copy.deepcopy(samples_metrics_results)
    for sample in samples_metrics_results:
        dfts[sample].to_csv(config.output_path + '/' + str(sample) + '_dfrs.csv', sep=';')
        samples[sample].to_csv(config.output_path + '/' + str(sample) + '_sample.csv', sep=';')

    pd.DataFrame(samples_metrics_results).to_csv(config.output_path + '/' + 'metrics.csv', sep=';')


def average_sampling_metrics(samples_metrics_results):
    """
    This function averages the quality metrics from all algorithms that are set as non-deterministic in the
    Config-Manager.
    :param samples_metrics_results:
    :return:
    """
    # iterate over all algorithms specified as non deterministic.
    for algo_to_repeat in config.algos_to_average:
        # iterate over all algos and only the selected above is used
        if algo_to_repeat in config.sampling_algo:
            result_list = []

            # the key need to be cased to a list, as the dict changes while used. This can lead to errors
            keys = list(samples_metrics_results.keys())

            # aggregate the results form one sampling algo in the result_list
            for algo in keys:

                # select only the repetitions of the currently selected algorithm,
                # by comparing the start of the algo name
                if algo.startswith(algo_to_repeat):

                    #create a list with all results from this algo type
                    result_list.append(samples_metrics_results.pop(algo))
            current_algo_averages = {}

            # iterate over all samples from this algo
            for sample_ratio in config.ab_sample_ratio:
                metrics_for_current_sr = []
                for repeated_algo in range(len(result_list)):
                    metrics_for_current_sr.append(result_list[repeated_algo][sample_ratio])

                #prepare the df
                metrics_for_current_sr_df = pd.DataFrame(metrics_for_current_sr).transpose()

                # get the average of the metrics
                metrics_for_current_sr_df['average'] = metrics_for_current_sr_df.mean(numeric_only=True, axis=1)

                # create a dict to have the same format as the deterministic algorithms
                metrics_for_current_sr = metrics_for_current_sr_df['average'].to_dict()
                current_algo_averages[sample_ratio] = metrics_for_current_sr
            samples_metrics_results[algo_to_repeat] = current_algo_averages

    return samples_metrics_results


def save_overall_results(samples_metrics_results):
    """
    Function for saving the overall results form all algos in one file.
    :param samples_metrics_results:
    :return:
    """


    os.mkdir(config.output_path + "/results/")

    for algo in samples_metrics_results:
        df1 = pd.DataFrame(samples_metrics_results[algo])
        df1 = df1.round(4)
        df1.to_csv(config.output_path + "/results/" + algo + ".csv", sep=";")

