from ConfigManager import Config_Manager as config


def prepare_log_for_pm(base_log, samples):
    """
    This function collects the preparation steps that are needed between the sampling and the process mining. The
    steps are applied of the base_log and on the sampled logs in the samples-dict.
    :param base_log: dataframe, the base log
    :param samples: dict, Dictionary of the sampled logs
    :return: base_log and samples is the same format but prepared
    """
    # rename the columns of the dataframe to conform with the process mining class
    base_log = base_log.rename(columns={config.ab_case_id_column: "case_ID", config.ab_activity_column: "activity",
                                        config.ab_timestamp_column: "timestamp"},
                               errors="raise")

    # iterate though all samples
    for key in samples:
        # rename the columns of the dataframe to conform with the process mining class
        samples[key] = samples[key].rename \
            (columns={config.ab_case_id_column: "case_ID", config.ab_activity_column: "activity",
                      config.ab_timestamp_column: "timestamp"}, errors="raise")

    return base_log, samples
