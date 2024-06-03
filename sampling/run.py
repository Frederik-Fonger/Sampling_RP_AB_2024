import os

import ConfigManager
import conformance_checking
import process_discovery
import sampling_niclas.sampling_n as sampling
import sampling_eval
import pandas as pd
import sampling_utils
from ConfigManager import Config_Manager as config
import sampling
import pm_utils
from datetime import datetime
import time


def run(event_log_path=""):
    sampling_utils.add_algos_to_repeat()
    if event_log_path != "":
        config.source_path = event_log_path
    if len(config.sampling_algo) == 1 and config.sampling_mode == "for PM":
        run_single_algo()
    elif len(config.sampling_algo) > 1 and config.sampling_mode == "for PM":
        run_multi_algo()
    elif config.sampling_mode == "for CC":
        sampling_for_CC()
    elif config.sampling_mode == "only sampling":
        only_sample_and_eval()
    else:
        raise ValueError("No valid sampling mode selected")


def run_single_algo():
    # track start time
    start_time = time.time()
    # get the name of the source file
    stripped_event_log_name = config.source_path[config.source_path.rfind("/") + 1:config.source_path.rfind(".")]
    # set the output path for the iteration
    config.output_path = os.getcwd() + '/data/output/sampling_model_eval ' + str(
        datetime.now().strftime("%Y-%m-%d %H_%M_%S") + "_" + stripped_event_log_name + '/')
    print("Preparation finished")

    # import
    print("Start data import")
    # load the base log for sampling and evaluation. Path is set in the config manager
    base_log = sampling_utils.import_data()
    print("Data import of " + str(config.source_path) + " successful")
    # pre-processing

    # sampling
    print("Start sampling")
    # track start time for sampling
    start_time_sampling = time.time()
    # start the sampling process
    samples = sampling.sampling(base_log, config.sampling_algo[0])
    # track the end time of the sampling
    end_time_sampling = time.time()
    print("Sampling successful")

    # process mining
    # transfer the logs from the output format of the sampling to the input format of the process mining
    base_log, samples = pm_utils.prepare_log_for_pm(base_log=base_log, samples=samples)
    print("Log preparation successful")
    # track the start time of the process mining and evaluation
    start_time_pm_and_eval = time.time()
    # the process mining and the evaluation of the generated models
    process_discovery.run_model_creation_and_eval(base_log, samples, algo=config.sampling_algo[0])
    # track end time of the proces mining and eval
    end_time_pm_and_eval = time.time()
    # saving the samples as a file
    sampling_utils.save_logs(base_log, samples)
    # track the end time of the proces mining and eval
    end_time = time.time()
    # calculate and print the processing times
    sampling_utils.time_calculation(start_time, start_time_sampling, start_time_pm_and_eval, end_time_sampling,
                                    end_time_pm_and_eval, end_time)
    # save a copy of the configuration from the ConfigManager
    sampling_utils.save_info_file()
    print("Finished successfully with: " + config.source_path)
    # evaluation


def run_multi_algo():
    """
    function for starting multiple sampling algorithms in one execution
    :return:
    """
    # track start time
    start_time = time.time()
    # get the name of the source file
    stripped_event_log_name = config.source_path[config.source_path.rfind("/") + 1:config.source_path.rfind(".")]
    # set the output path for the iteration
    test = os.getcwd()
    config.output_path = os.getcwd() + '/data/output/sampling_model_eval ' + str(
        datetime.now().strftime("%Y-%m-%d %H_%M_%S") + "_" + stripped_event_log_name + '/')
    print("Preparation finished")

    # import
    print("Start data import")
    # load the base log for sampling and evaluation. Path is set in the config manager
    base_log = sampling_utils.import_data()
    print("Data import of " + str(config.source_path) + " successful")
    # pre-processing

    # sampling
    print("Start sampling")
    # init dict to collect the quality metrics of the models for each sampling configuration
    quality_metrics = {}
    # save original path to be accessible in all iterations
    original_path = config.output_path
    # iterate though all selected sampling algos

    if config.output:
        os.makedirs(config.output_path + '/graphs')
    for algo in config.sampling_algo:
        print("Start sampling with " + algo + " for " + config.source_path)
        # create output path for this specific algo
        config.output_path = original_path + algo
        os.mkdir(config.output_path)
        samples = sampling.sampling(base_log.copy(), algo)

        # process mining
        # transfer the logs from the output format of the sampling to the input format of the process mining
        # base log need to be copied, to preserve original base log format for next iteration
        base_log_mod, samples = pm_utils.prepare_log_for_pm(base_log=base_log.copy(deep=True), samples=samples)

        # the process mining and the evaluation of the generated models
        quality_metrics[algo] = process_discovery.run_model_creation_and_eval(base_log_mod, samples, algo)

        # saving the samples as a file
        sampling_utils.save_logs(base_log_mod, samples)
        print("End of " + algo + " for " + config.source_path)
    # reset the output path for the results
    config.output_path = original_path
    quality_metrics = sampling_utils.average_sampling_results(quality_metrics)
    sampling_utils.curve_plot_compare_algos(quality_metrics, miner="inductive miner")
    sampling_utils.curve_plot_compare_algos(quality_metrics, miner="heuristic miner")

    # track the end time of the proces mining and eval
    end_time = time.time()
    # calculate and print the processing times
    sampling_utils.time_calculation_multi_algo(start_time, end_time)
    # save a copy of the configuration from the ConfigManager
    sampling_utils.save_info_file()
    print("Finished successfully with: " + config.source_path)
    # evaluation


def sampling_for_CC():
    """
        function for starting multiple sampling algorithms in one execution
        :return:
        """
    # track start time
    start_time = time.time()
    # get the name of the source file
    stripped_event_log_name = config.source_path[config.source_path.rfind("/") + 1:config.source_path.rfind(".")]
    # set the output path for the iteration
    config.output_path = os.getcwd() + '/data/output/sampling_model_eval ' + str(
        datetime.now().strftime("%Y-%m-%d %H_%M_%S") + "_" + stripped_event_log_name + '/')
    print("Preparation finished")

    # import
    print("Start data import")
    # load the base log for sampling and evaluation. Path is set in the config manager
    base_log = sampling_utils.import_data()
    print("Data import of " + str(config.source_path) + " successful")
    # pre-processing

    # sampling
    print("Start sampling")
    # init dict to collect the quality metrics of the models for each sampling configuration
    quality_metrics = {}
    # save original path to be accessible in all iterations
    original_path = config.output_path
    # iterate though all selected sampling algos
    if config.output:
        os.makedirs(config.output_path + '/graphs')

    for algo in config.sampling_algo:
        print("Start sampling with " + algo + " for " + config.source_path)
        # create output path for this specific algo
        config.output_path = original_path + algo
        os.mkdir(config.output_path)
        samples = sampling.sampling(base_log, algo)

        base_log_mod, samples = pm_utils.prepare_log_for_pm(base_log=base_log.copy(deep=True), samples=samples)
        # process mining
        quality_metrics[algo] = conformance_checking.pm_with_cc_for_samples(base_log_mod, samples, algo)

        # saving the samples as a file
        sampling_utils.save_logs(base_log_mod, samples)
        print("End of " + algo + " for " + config.source_path)
    # reset the output path for the results
    config.output_path = original_path
    quality_metrics = sampling_utils.average_sampling_results(quality_metrics)
    sampling_utils.curve_plot_compare_algos_for_CC(quality_metrics, miner="inductive miner")
    sampling_utils.curve_plot_compare_algos_for_CC(quality_metrics, miner="heuristic miner")

    # track the end time of the proces mining and eval
    end_time = time.time()
    # calculate and print the processing times
    sampling_utils.time_calculation_multi_algo(start_time, end_time)
    # save a copy of the configuration from the ConfigManager
    sampling_utils.save_info_file()
    print("Finished successfully")
    # evaluation

def only_sample_and_eval():
    # track start time
    start_time = time.time()
    # get the name of the source file
    stripped_event_log_name = config.source_path[config.source_path.rfind("/") + 1:config.source_path.rfind(".")]
    # set the output path for the iteration
    test = os.getcwd()
    config.output_path = os.getcwd() + '/data/output/sampling_model_eval ' + str(
        datetime.now().strftime("%Y-%m-%d %H_%M_%S") + "_" + stripped_event_log_name + '/')
    print("Preparation finished")

    # import
    print("Start data import")
    # load the base log for sampling and evaluation. Path is set in the config manager
    base_log = sampling_utils.import_data()
    print("Data import of " + str(config.source_path) + " successful")
    # pre-processing

    # sampling
    print("Start sampling")
    # init dict to collect the quality metrics of the models for each sampling configuration
    quality_metrics = {}
    # save original path to be accessible in all iterations
    original_path = config.output_path
    if config.output:
        os.makedirs(config.output_path + '/graphs')

    # dicts for collecting the results
    samples_metrics_results = {}
    dfrs = {}

    # iterate though all selected sampling algos
    for algo in config.sampling_algo:
        print("Start sampling with " + algo + " for " + config.source_path)

        # create output path for this specific algo
        config.output_path = original_path + algo
        os.mkdir(config.output_path)

        samples = sampling.sampling(base_log.copy(), algo)

        # calculate the quality metrics for the samples of the current "algo"
        if config.with_evaluation:
            samples_metrics_results[algo], dfrs[algo] = sampling_eval.samples_eval(samples, base_log, algo)

        # save the results as file
        if config.with_evaluation:
            sampling_eval.save_results(samples_metrics_results[algo], dfrs[algo], samples)

    # reset the output path
    config.output_path = original_path

    # average the results from non-deterministic algos
    if config.with_evaluation:
        sampling_eval.average_sampling_metrics(samples_metrics_results)

    # save the overall results as file
    if config.with_evaluation:
        sampling_eval.save_overall_results(samples_metrics_results)

    # save the overalls plots  as file
    if config.with_evaluation:
        sampling_eval.vis_results(samples_metrics_results)

    # track the end time of the proces mining and eval
    end_time = time.time()
    # calculate and print the processing times
    sampling_utils.time_calculation_multi_algo(start_time, end_time)
    # save a copy of the configuration from the ConfigManager
    sampling_utils.save_info_file()
    print("Finished successfully")
    # evaluation