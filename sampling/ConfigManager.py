

class Config_Manager:
    """
    Class ConfigManager stores all user-selected simulation configuration and makes it accessible throughout
    the program.
    """
    # For testing
    output = True  # sets if there is any file output
    output_path = "Not set yet!"  # sets the output path for the data. Is set in the run.py (ensuring one timestamp)
    sampling_mode = "only sampling"  # "for PM",  "for CC", "only sampling"
    with_evaluation = True

    # Different base logs for testing
    # Import
    # path to the log as is use as base log for sampling and process mining
    # source_path: str = "data/Eventlogs/DomesticDeclarations/DomesticDeclarations.xes"
    # source_path: str = "data/Eventlogs/BPIC 2012/BPI_Challenge_2012.xes"
    # source_path: str = "data/Eventlogs/InternationalDeclarations/InternationalDeclarations.xes"
    source_path: str = "data/Eventlogs/PrepaidTravelCost/PrepaidTravelCost.xes"
    # source_path: str = "data/Eventlogs/RequestForPayment/RequestForPayment.xes"
    # source_path: str = "data/Eventlogs/PermitLog/PermitLog.xes"
    # source_path: str = "data/Eventlogs/sepsis/Sepsis Cases - Event Log.xes"

    seperator_in_log = ";"  # the seperator in the file from the base log
    data_type = "xes"  # IS SET BY IMPORT FUNCTION !!!, specifies the file-type of the imported file

    # Sampling Define which sampling algorithms should be used choose as many as you like between
    # "allbehaviour_Sampling", "remainder_plus_sampling", "random_Sampling", "stratified_sampling", "c_min_sampling"
    sampling_algo = ["allbehaviour_Sampling", "remainder_plus_sampling", "random_Sampling", "stratified_sampling", "c_min_sampling"]  # ["allbehaviour_Sampling", "random_Sampling"]
    algos_to_average = ["random_Sampling", "stratified_sampling"]  # A list of all sampling algorithms which are not deterministic, that should be executed multiple time and the results are averaged
    algos_to_average_repeats = 4  # specifies how often the algorithms are repeated (only for non-deterministic algos)

    # Allbehaviour Sampling
    ab_format = "DF"               # IS SET BY IMPORT FUNCTION !!!  format (str): The format of the event log. Supported formats: 'CSV', 'XES or 'DF'. 'DF' stands for DataFrame
    ab_sample_ratio = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # [0.01,0.1,0.2, 0.5, 0.7] # [0.001, 0.005, 0.01, 0.05, 0.1]  # [0.2, 0.5, 0.7] #[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]   # [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]  #  [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]  # [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]  #    # sample_ratio (float): The proportion of the original event log to be included in the sample.
    ab_case_id_column = "Case ID"  # case_id_column (str): The name of the column representing case IDs in the event log. If the format 'XES' is given, the default value is 'case:concept:name', unless specified otherwise.
    ab_activity_column = "Activity"# activity_column (str): The name of the column representing activity in the event log. If the format 'XES' is given, the default value is 'concept:name', unless specified otherwise.
    ab_timestamp_column = "dd-MM-yyyy:HH.mm"  # timestamp_column (str): The name of the column representing timestamps in the event log. If the format 'XES' is given, the default value is 'time:timestamp', unless specified otherwise.
    ab_format_timestamp_conversion_to_datetime_obj = "%d-%m-%Y:%H.%M"  # format_timestamp_conversion_to_datetime_obj (str, optional): The format string for parsing timestamps into datetime objects. Defaults to 'YYYY-MM-DD HH:MM:SS.ssssss' (relevant in case format == 'CSV' or format == 'DF')
    ab_min_columns = True          # min_columns (bool, optional): Flag to determine if the output should contain only essential columns (Case ID, Activity, Timestamp). Defaults to False.

    # process mining and model eval

    # list of the used thresholds for the process mining algorithms
    noise_threshold_list = [0, 0.2, 0.4, 0.6, 0.9]

    only_f1_score = True
    only_token_based = True

    # Output
    log_output_format = "csv"  # xes  # File format in which the sampled logs are saved













    def __init__(self):
        """
        Constructor method of class Sampling_Parameters.Receives no parameters, all attributes need to be set manually.
        """

        # General settings
        self.first: int = 0  # Text
