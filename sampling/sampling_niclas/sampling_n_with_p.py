import pm4py
import pandas as pd
import numpy as np
from tqdm import tqdm

from ConfigManager import Config_Manager as config






def _get_list_of_pairs(preprocessed_event_log):

    list_of_pairs = []
    for variant in preprocessed_event_log:
        # variant[1] mal:
        for i in range(variant[1]):
            x = list(zip(variant[0], variant[0][1:]))
            list_of_pairs.append(x)
    # clean the pairs of brackets
    list_of_pairs_formated = []
    for i in list_of_pairs:
        for k in i:
            list_of_pairs_formated.append(k)

    return list_of_pairs_formated

def _get_list_of_single_unique_activities(list_of_pairs):
    unique_activities_set = set()
    for a_tuple in list_of_pairs:
        unique_activities_set.update(a_tuple)
    return list(unique_activities_set)



from collections import Counter
def _calculate_ratios(original_event_log, sampled_event_log):
    """This function calculates the ratios of behavior pairs between an original event log and a sampled event log. 
       It creates a ratio matrix to compare the frequencies of behavior pairs in both logs. 
       This function is used for intermediate calculations and should not be used directly by the user.



    Parameters: 
        original_event_log (list[tuple[Tuple[str], List[Trace]]]): The original event_log
        sampled_event_log (list[tuple[Tuple[str], List[Trace]]]): The sampled event log 

    Returns: 
        unsampled_behavior_list: A list of behavior pairs that were not sampled.
        list_of_pairs_with_sample_ratio: A list of behavior pairs along with their sampling ratios.
        list_of_pairs_with_count_ol: The frequency of different behavior pairs in the original log.
    """

    # extract behavior pairs of the original event log
    list_of_pairs_ol = _get_list_of_pairs(original_event_log)
    # extract behavior pairs of the sampled event log
    list_of_pairs_sl = _get_list_of_pairs(sampled_event_log)


    # count the frequency of the different pairs for the original log 
    list_of_pairs_with_count_ol = list(Counter(list_of_pairs_ol).items())

    # count the frequency of the different pairs for the sample log
    list_of_pairs_with_count_sl = list(Counter(list_of_pairs_sl).items())


    # extract list of unique activities
    list_of_single_unique_activities = _get_list_of_single_unique_activities(
        list_of_pairs_ol)
      
    
    # Sort activities
    list_of_single_unique_activities_sorted = sorted(
        [x for x in list_of_single_unique_activities])
    

    # initialize adjacency matrix:
    # number of rows and columns = number of unique activities in the original event log
    num_of_rows_and_columns = len(list_of_single_unique_activities)

    adjacency_matrix = np.zeros(
        (num_of_rows_and_columns, num_of_rows_and_columns), dtype=float)

    for behavior_with_count in list_of_pairs_with_count_ol:
        behavior_start = behavior_with_count[0][0]
        behavior_end = behavior_with_count[0][1]
        behavior_count = behavior_with_count[1]

        for idx, element in enumerate(list_of_single_unique_activities_sorted):
            if element == behavior_start:
                row = idx
            if element == behavior_end:
                col = idx

        adjacency_matrix[row][col] = behavior_count

    adjacency_matrix_sample = np.zeros(
        (num_of_rows_and_columns, num_of_rows_and_columns), dtype=float)

    for behavior_with_count in list_of_pairs_with_count_sl:
        behavior_start = behavior_with_count[0][0]
        behavior_end = behavior_with_count[0][1]
        behavior_count = behavior_with_count[1]

        for idx, element in enumerate(list_of_single_unique_activities_sorted):
            if element == behavior_start:
                row = idx
            if element == behavior_end:
                col = idx

        adjacency_matrix_sample[row][col] = behavior_count

    ################## build DFR-Ratio Matrix#############################################
    adjacency_matrix_sample_ratio = np.zeros(
        (num_of_rows_and_columns, num_of_rows_and_columns), dtype=float)

    list_of_pairs_with_sample_ratio = [
        [list(x), y] for x, y in list_of_pairs_with_count_ol]

    for behavior_with_count in list_of_pairs_with_sample_ratio:
        behavior_start = behavior_with_count[0][0]
        behavior_end = behavior_with_count[0][1]
        behavior_count = behavior_with_count[1]

        for idx, element in enumerate(list_of_single_unique_activities_sorted):
            # identify the right column and row
            if element == behavior_start:
                row = idx

            if element == behavior_end:
                col = idx

        # caluculate the ratio of the dfr ...
        if adjacency_matrix_sample[row][col] <= behavior_count and adjacency_matrix_sample[row][col] > 0:
            # adjacency_matrix_sample_ratio[row][col] = round(adjacency_matrix_sample[row][col]/behavior_count,2)
            adjacency_matrix_sample_ratio[row][col] = adjacency_matrix_sample[row][col]/behavior_count
            behavior_with_count[1] = adjacency_matrix_sample_ratio[row][col]
        elif adjacency_matrix_sample[row][col] == 0:
            behavior_with_count[1] = adjacency_matrix_sample_ratio[row][col]
        else:
            pass



        unsampled_behavior_list = []
        
        for x in list_of_pairs_with_sample_ratio:

            if x[1] == 0:
                unsampled_behavior_list.append(x)


    calculated_ratios = {
        "unsampled_behavior_list": unsampled_behavior_list,
        "list_of_pairs_with_sample_ratio": list_of_pairs_with_sample_ratio,
        "list_of_pairs_with_count_ol": list_of_pairs_with_count_ol,
       
    }

    return calculated_ratios










def _create_dataframe_for_sample(dataframe_of_the_original_log, sample, format, case_id_column, activity_column, timestamp_column, min_columns):
    """
    Creates a DataFrame for a sampled event log based on the provided sample.

    Parameters:
    - dataframe_of_the_original_log (DataFrame): The DataFrame of the original event log.
    - sample (list of tuples): The sample data to be included in the new DataFrame. Each tuple in the list represents a trace of the original log and its corresponding count.
    - format (str): The format of the event log. Supported formats include 'XES', 'CSV', etc. 'XES' is used to set default column names if no column names are provided.
    - case_id_column (str): The name of the column representing case IDs in the event log. Defaults to 'case:concept:name' if empty and format is 'XES'.
    - activity_column (str): The name of the column representing activities in the event log. Defaults to 'concept:name' if empty and format is 'XES'.
    - timestamp_column (str): The name of the column representing timestamps in the event log. Defaults to 'time:timestamp' if empty and format is 'XES'.
    - min_columns (bool, optional): Flag to determine if the output should contain only essential columns (Case ID, Activity, Timestamp). Defaults to False. 

    Returns:
    - DataFrame: A new DataFrame constructed from the original log, containing only the cases and traces specified in the 'sample' parameter. 
    The returned DataFrame has the same column structure as the original event log. In case the min_columns flag is set to True the returned Dataframe only consists of the essential columns (case id, activity, timestamp)

    """


    if format == "XES" and case_id_column == "" and activity_column == "" and timestamp_column == "": 
        case_id_column = 'case:concept:name'
        activity_column = 'concept:name'
        timestamp_column = 'time:timestamp'
  
   
    case_traces = dataframe_of_the_original_log.groupby(case_id_column)[ activity_column].apply(tuple)

    
    sample_dict = {trace: count for trace, count in sample}

    selected_cases = {trace: [] for trace in sample_dict.keys()}
    for case_id, activities in case_traces.items():
        for trace in sample_dict.keys():
            if activities == trace and len(selected_cases[trace]) < sample_dict[trace]:
                selected_cases[trace].append(case_id)

    data_for_sample_df = []
    for trace, case_ids in selected_cases.items():
        for case_id in case_ids:
            data_for_sample_df.extend(dataframe_of_the_original_log[dataframe_of_the_original_log[case_id_column] == case_id].values)

    sample_df = pd.DataFrame(data_for_sample_df, columns=dataframe_of_the_original_log.columns)



     # Should we just keep the essential columns of the dataframe sample (case id, activity, timestamp)? 
    if min_columns and format == "XES" and case_id_column == "" and activity_column == "" and timestamp_column == "":
        return sample_df[['case:concept:name', 'concept:name', 'time:timestamp']]
    
    elif min_columns and format == "XES" and case_id_column != "" and activity_column != "" and timestamp_column != "":
        return sample_df[[case_id_column, activity_column, timestamp_column]]
    
    elif min_columns and format != "XES": 
        return sample_df[[case_id_column, activity_column, timestamp_column]]
    else: 

        # We return the dataframe of the sample with all information
        return sample_df
    


















def formating_and_determing_variants(dataframe_or_filepath_to_log, format, case_id_column, activity_column, timestamp_column, format_timestamp_conversion_to_datetime_obj):
    """ Formats an event log and determines the frequency of the variants within it.

    Parameters:
    - dataframe_or_filepath_to_log (str or DataFrame): The file path to the event log or a DataFrame representing the event log. The source format is specified by the 'format' parameter.
    - format (str): The format of the event log. Supported formats include 'XES', 'CSV', and 'DF'. 'DF' stands for DataFrame.
    - case_id_column (str): The name of the column representing case IDs in the event log.  If the format 'XES' is given, the default value is 'case:concept:name', unless specified otherwise.
    - activity_column (str): The name of the column representing activities in the event log. If the format 'XES' is given, the default value is 'concept:name', unless specified otherwise.
    - timestamp_column (str): The name of the column representing timestamps in the event log. If the format 'XES' is given, the default value is 'time:timestamp', unless specified otherwise
    - format_timestamp_conversion_to_datetime_obj (str): The format string for converting string timestamps into Python datetime objects. This is relevant when the format is either 'CSV' or 'DF'.

    Returns:
    - tuple: A tuple containing three elements:
    - event_log (list of tuples): A processed version of the event log where each tuple represents a variant with its corresponding frequency.
    - num_of_traces_in_ol (int): The total number of traces in the original event log.
    - dataframe (DataFrame): A DataFrame representation of the event log

    Raises:
    - FileNotFoundError: If the specified file in 'dataframe_or_filepath_to_log' is not found.
    - ValueError: If an unsupported file format is provided or other value errors occur during processing.
    - Exception: For any other unexpected errors during execution.

    """

    
    event_log = None 
    dataframe = None
    try:

        if format == "XES":


            event_log = pm4py.read_xes(dataframe_or_filepath_to_log)
            df = pm4py.convert_to_dataframe(event_log)
            dataframe = df.copy()

            # event_log_pm4py = pm4py.convert_to_event_log(df)
            event_log_pm4py = df
            variants_pm4py = pm4py.get_variants_as_tuples(event_log_pm4py)

            # calculate the number of traces in the original event log and determine the frequency of the variants due counting the associated traces
            num_of_traces_in_ol = 0
            for variant in variants_pm4py:
                # variants_pm4py[variant] = len(variants_pm4py[variant])
                num_of_traces_in_ol += variants_pm4py[variant]

            # the preprocessed event log is a list of tuples. In each tuple a variant is beeing safed with the corresponding variant frequency
            event_log = list(variants_pm4py.items())


        elif format == "CSV":

            #read in CSV-file
            df = pd.read_csv(dataframe_or_filepath_to_log, sep=";")

             # convert to datetime64 object
            df[timestamp_column] = pd.to_datetime(df[timestamp_column], format=format_timestamp_conversion_to_datetime_obj, errors='coerce', utc=True)

            dataframe = df.copy()

            df = df.filter([case_id_column,activity_column,timestamp_column])

            # rearrange the order of columns
            df = df[[case_id_column, activity_column, timestamp_column]]

            # sort based on Case ID and Timestamp:
            df.sort_values([case_id_column, timestamp_column], ascending=[
                        True, True], inplace=True)


            df_pm4py = pm4py.format_dataframe(
                df, case_id=case_id_column, activity_key=activity_column, timestamp_key=timestamp_column, timest_format=format_timestamp_conversion_to_datetime_obj)

            # we have all information safed in case_id, activity_key and timestamp_key
            df_pm4py.drop(columns=[case_id_column, activity_column, timestamp_column], inplace=True)


            # event_log_pm4py = pm4py.convert_to_event_log(df_pm4py)
            event_log_pm4py = df_pm4py

            # get the variants and the associated traces
            variants_pm4py = pm4py.get_variants_as_tuples(event_log_pm4py)


            # calculate the number of traces in the original event log and determine the frequency of the variants due counting the associated traces
            num_of_traces_in_ol = 0
            for variant in variants_pm4py:
                # variants_pm4py[variant] = len(variants_pm4py[variant])
                num_of_traces_in_ol += variants_pm4py[variant]

            # the preprocessed event log is a list of tuples. In each tuple a variant is beeing safed with the corresponding variant frequency
            event_log = list(variants_pm4py.items())


        elif format == "DF":

            df = dataframe_or_filepath_to_log


            # convert to datetime64 object
            df[timestamp_column] = pd.to_datetime(df[timestamp_column], format=format_timestamp_conversion_to_datetime_obj, errors='coerce', utc=True)


            dataframe = df.copy()


            df = df.filter([case_id_column,activity_column,timestamp_column])

            # rearrange the order of columns
            df = df[[case_id_column, activity_column, timestamp_column]]


            # sort based on Case ID and Timestamp:
            df.sort_values([case_id_column, timestamp_column], ascending=[
                        True, True], inplace=True)


            df_pm4py = pm4py.format_dataframe(
                df, case_id=case_id_column, activity_key=activity_column, timestamp_key=timestamp_column, timest_format=format_timestamp_conversion_to_datetime_obj)

            # we have all information safed in case_id, activity_key and timestamp_key
            if not (format == "DF" and config.data_type == "xes"):
                df_pm4py.drop(columns=[case_id_column, activity_column, timestamp_column], inplace=True)


            # the event log class is depricard
            # event_log_pm4py = pm4py.convert_to_event_log(df_pm4py)
            event_log_pm4py = df_pm4py


            # get the variants and the associated traces
            variants_pm4py = pm4py.get_variants_as_tuples(event_log_pm4py)


            # calculate the number of traces in the original event log and determine the frequency of the variants due counting the associated traces
            num_of_traces_in_ol = 0
            for variant in variants_pm4py:
                # the following line can be skipped as the Eventlog-Class is not used anymore and the dataframe has allready the variant counts as an int
                # variants_pm4py[variant] = len(variants_pm4py[variant])
                num_of_traces_in_ol += variants_pm4py[variant]

            # the preprocessed event log is a list of tuples. In each tuple a variant is beeing safed with the corresponding variant frequency
            event_log = list(variants_pm4py.items())
        else:
            raise ValueError("Unsupported file format")

    except FileNotFoundError:
        print(f"File not found: {dataframe_or_filepath_to_log}")
    except ValueError as e:
        print(f"Data processing error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    
    return event_log, num_of_traces_in_ol, dataframe

























def remainder_plus_sampling(dataframe_or_filepath_to_log, format, sample_ratio, case_id_column, activity_column, timestamp_column, format_timestamp_conversion_to_datetime_obj = "YYYY-MM-DD HH:MM:SS.ssssss", min_columns = False): 
    """
    Processes an event log file or filepath to create a sampled event log based on the 'Remainder Plus Sampling' algorithm.

    Parameters:
    - datframe_or_filepath_to_log (str): The file path to the eventlog or a dataframe of the event log.
    - format (str): The format of the event log. Supported formats: 'CSV', 'XES or 'DF'. 'DF' stands for DataFrame  
    - sample_ratio (float): The proportion of the original event log to be included in the sample.
    - case_id_column (str): The name of the column representing case IDs in the event log. If the format 'XES' is given, the default value is 'case:concept:name', unless specified otherwise.
    - activity_column (str): The name of the column representing activity in the event log. If the format 'XES' is given, the default value is 'concept:name', unless specified otherwise.
    - timestamp_column (str): The name of the column representing timestamps in the event log. If the format 'XES' is given, the default value is 'time:timestamp', unless specified otherwise.
    - format_timestamp_conversion_to_datetime_obj (str, optional): The format string for parsing timestamps into datetime objects. Defaults to 'YYYY-MM-DD HH:MM:SS.ssssss' (relevant in case format == 'CSV' or format == 'DF')
    - min_columns (bool, optional): Flag to determine if the output should contain only essential columns (Case ID, Activity, Timestamp). Defaults to False.

    Returns:
    - pandas.DataFrame: A sampled DataFrame of the event log, which may be minimized to essential columns based on the 'min_columns' flag.

    """ 


    format = format.upper()

    event_log, num_of_traces_in_ol, dataframe_of_ol = formating_and_determing_variants(dataframe_or_filepath_to_log = dataframe_or_filepath_to_log, format = format, case_id_column = case_id_column, activity_column = activity_column, timestamp_column = timestamp_column, format_timestamp_conversion_to_datetime_obj = format_timestamp_conversion_to_datetime_obj)
    

    ######Start of Remainder_Plus_Sampling###################
   
    # we have to make a copy of the event log because we will assign new values to event_log object
    original_event_log = [x for x in event_log]

    # change tuple elements to lists in order to be able to assign new values
    event_log = [[list(x), y] for x, y in event_log]

    # sort variants based on their frequency in descending order
    event_log_sorted = sorted(
        event_log, key=lambda element: element[1], reverse=True)

    # calculate the number of traces in the sample:
    num_of_traces_in_sl = round(num_of_traces_in_ol * sample_ratio)

    # we have to track the available free slots left for the sample.
    remaining_free_slots_in_sample = num_of_traces_in_sl

    # In the beginning no trace is part of the sample. We build up the sample successively
    remainder_plus_sample = []

    # iterate through every variant in the event log...
    for variant in tqdm(event_log_sorted, desc='calculate the variants expected occurrences in the sample'):
        # and calculate the variants' expected occurrences in the sample
        the_variants_expected_occurrence = variant[1] * sample_ratio
        variant[1] = the_variants_expected_occurrence

        # append the variant to the sample if the variants expected occurrence is greater or equal to 1.
        # The sampled variant's occurrence is reassigned with the integer part of the calculated variants expected occurrence.
        if variant[1] >= 1:
            intnum = int(variant[1])
            remainder = variant[1] % intnum
            variant[1] = remainder
            remaining_free_slots_in_sample -= intnum
            corpus = []
            corpus.append(variant[0])
            corpus.append(intnum)
            remainder_plus_sample.append(corpus)

    # we have to sort on behaviour characteristics in case two or more variants have the same remainder. Therefore we first check which behaviour is undersampled or oversampled
    intermediate_results = _calculate_ratios(
        original_event_log=original_event_log, sampled_event_log=remainder_plus_sample)

    behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version = {tuple(
        x): y for x, y in intermediate_results.get("list_of_pairs_with_sample_ratio")}
    behaviour_pairs_in_original_log_with_count = {
        tuple(x): y for x, y in intermediate_results.get("list_of_pairs_with_count_ol")}

    # initiliaze every variant with rank = 0
    for variant in event_log_sorted:
        variant.append(0)

    progress_bar = tqdm(total=remaining_free_slots_in_sample, desc="Selecting traces with remainder plus")

    while remaining_free_slots_in_sample > 0:
        progress_bar.update(1)
        # Calculate each variant's rank (it's called "differenzwert" in the thesis)
        # The variant will get a positive rank if the variant has more undersampled behaviour than oversampled behaviour and will likely be sampled
        # The rank of the variant will increase by one for each undersampled behaviour and will decrease by one for each oversampled behaviour in the sample
        for variant in event_log_sorted:

            rank = 0

            pairs_in_one_variant = list(zip(variant[0], variant[0][1:]))

            for pair in pairs_in_one_variant:

                if behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version.get(pair) > sample_ratio:
                    #pair is oversampled
                    rank -= 1
                if behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version.get(pair) < sample_ratio:
                    #pair is undersampled
                    rank += 1
                else:
                    # pair is perfectly sampled
                    rank += 0

            # normalize rank by the the number of pairs in the variant
            if (len(pairs_in_one_variant) > 0):
                rank = rank/len(pairs_in_one_variant)

            # assign the rank to the variant
            variant[2] = rank
        remaining_free_slots_in_sample -= 1

        # sort by two attributes. First by the remainder and then sort by the rank
        event_log_sorted = sorted(
            event_log_sorted, key=lambda x: (x[1], x[2]), reverse=True)

        # the variant that has to sampled next has to be on index 0 of the list (highest remainder and compared to all variants with the same remainder it has the highest rank)
        variant_to_be_sampled = event_log_sorted[0]

        # update all corrsponding behaviour pairs:
        variants_pairs_to_be_sampled = list(
            zip(variant_to_be_sampled[0], variant_to_be_sampled[0][1:]))
        for pair in variants_pairs_to_be_sampled:
            behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version[pair] = ((behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version.get(
                pair) * behaviour_pairs_in_original_log_with_count.get(pair)) + 1)/behaviour_pairs_in_original_log_with_count.get(pair)

        # Add the variant to the sample. Therefore we have to check whether the variant is already part of our sample. In case the variant is already in our sample, we have to update the frequency of that variant in our sample,
        # Otherwise we add the variant to our sample with the frequency of one
        flag = False
        for already_sampled_variant in remainder_plus_sample:
            if variant_to_be_sampled[0] == already_sampled_variant[0]:
                already_sampled_variant[1] += 1
                flag = True
        if flag == False:
            corpus = []
            corpus.append(variant_to_be_sampled[0])
            corpus.append(1)
            remainder_plus_sample.append(corpus)
            flag = False

        # update the remainder because we have sampled the unique variant now
        variant_to_be_sampled[1] = 0
    progress_bar.close()

    # Conversion to the right syntax
    for variant in tqdm(remainder_plus_sample, desc='Converting the log to the correct format'):
        for idx, x in enumerate(variant):
            if idx == 0:
                variant[idx] = tuple(variant[idx])
    remainder_plus_sample = [tuple(x) for x in remainder_plus_sample]

    # print("SAMPLE IN OLD FORMAT")
    # print(remainder_plus_sample)
    # print("\n\n\n")
    


    ######################DATAFRAME CREATION#################################################################

    dataframe_of_sample = _create_dataframe_for_sample(dataframe_of_the_original_log = dataframe_of_ol, sample = remainder_plus_sample, format = format, case_id_column = case_id_column, activity_column = activity_column, timestamp_column = timestamp_column, min_columns = min_columns) 


    
    return dataframe_of_sample
   

   
      









def allbehaviour_Sampling(dataframe_or_filepath_to_log, format, sample_ratio, case_id_column, activity_column, timestamp_column, format_timestamp_conversion_to_datetime_obj = "YYYY-MM-DD HH:MM:SS.ssssss", min_columns = False):
    """
    Processes an event log file or filepath to create a sampled event log based on the 'Allbehaviour Sampling algorithm.

    Parameters:
    - datframe_or_filepath_to_log (str): The file path to the eventlog or a dataframe of the event log.
    - format (str): The format of the event log. Supported formats: 'CSV', 'XES or 'DF'. 'DF' stands for DataFrame  
    - sample_ratio (float): The proportion of the original event log to be included in the sample.
    - case_id_column (str): The name of the column representing case IDs in the event log. If the format 'XES' is given, the default value is 'case:concept:name', unless specified otherwise.
    - activity_column (str): The name of the column representing activity in the event log. If the format 'XES' is given, the default value is 'concept:name', unless specified otherwise.
    - timestamp_column (str): The name of the column representing timestamps in the event log. If the format 'XES' is given, the default value is 'time:timestamp', unless specified otherwise.
    - format_timestamp_conversion_to_datetime_obj (str, optional): The format string for parsing timestamps into datetime objects. Defaults to 'YYYY-MM-DD HH:MM:SS.ssssss' (relevant in case format == 'CSV' or format == 'DF')
    - min_columns (bool, optional): Flag to determine if the output should contain only essential columns (Case ID, Activity, Timestamp). Defaults to False.

    Returns:
    - pandas.DataFrame: A sampled DataFrame of the event log, which may be minimized to essential columns based on the 'min_columns' flag.

    """ 


    format = format.upper()

    event_log, num_of_traces_in_ol, dataframe_of_ol = formating_and_determing_variants(dataframe_or_filepath_to_log = dataframe_or_filepath_to_log, format = format, case_id_column = case_id_column, activity_column = activity_column, timestamp_column = timestamp_column, format_timestamp_conversion_to_datetime_obj = format_timestamp_conversion_to_datetime_obj)

    original_event_log = [x for x in event_log]

    # calculate the number of traces in the sample
    num_of_traces_in_sl = round(num_of_traces_in_ol * sample_ratio)

    # We have safed each variant,frequency combination as a tuple. We want to be able to change the frequency of each variant but tuples are immutable. Therefore change the event log to a form where we can
    # assign new values
    # [(('a', 'b', 'c'), 12), (('a', 'b', 'e'), 4)] --> [[['a', 'b', 'c'], 12], [['a', 'b', 'e'], 4]]
    event_log_formated = [[list(x), y] for x, y in event_log]

    # build up the sample successively. Initialized the sample with the most frequent variant.
    highest_frequency = 0
    index_of_the_variant_with_highest_frequency = 0

    for idx, variant in enumerate(event_log_formated):
        if variant[1] > highest_frequency:
            highest_frequency = variant[1]
            index_of_the_variant_with_highest_frequency = idx

    first_variant_added_to_the_sample = event_log_formated[index_of_the_variant_with_highest_frequency].copy(
    )
    first_variant_added_to_the_sample[1] = 1

    # add the first variant to the sample with frequency 1
    allbehaviour_sample = []
    allbehaviour_sample.append(first_variant_added_to_the_sample)

    # reduce the remainding free spots in the sample by one
    remainding_free_slots_in_sample = num_of_traces_in_sl-1

    # Determine the unsampled behaviour...Therefore check which behaviour is not part of our sample yet.
    metrics = _calculate_ratios(
        original_event_log=event_log, sampled_event_log=allbehaviour_sample)

    # The unsampled behaviour list is a list of behaviour pairs that are part of the event log and haven't been added to the sample yet.
    unsampled_behaviour = [tuple(x[0])
                           for x in metrics.get("unsampled_behavior_list")]

    # keep track of the variants that have been already sampled
    variants_that_have_been_sampled = []
    variants_that_have_been_sampled.append(
        first_variant_added_to_the_sample[0])

    progress_bar = tqdm(total=len(unsampled_behaviour), desc="Adding unsampled behavior")
    last_unsapled_count = len(unsampled_behaviour)

    while remainding_free_slots_in_sample > 0 and len(unsampled_behaviour) > 0:

        progress_bar.update(last_unsapled_count - len(unsampled_behaviour))
        last_unsapled_count = len(unsampled_behaviour)
        index_of_new_variant = 0
        max_normalized_count = 0

        # Iterate through every variant and determine the variant's dfr.
        # For each variant count the number of dfr that haven't been added to the sample yet.
        # Due to the fact that long variants (number of events in the sequence/trace is high) have a higher probability of having a high count (a high number of unsampled behaviour pairs),
        # divide the count of unsampled dfr in the variant by the count of dfr in the variant (normalization)
        for idx, variant in enumerate(event_log_formated):

            pairs_in_one_variant = list(zip(variant[0], variant[0][1:]))

            count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet = 0

            for pair in pairs_in_one_variant:
                if pair in unsampled_behaviour:
                    count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet += 1

            # normalization step
            if count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet > 0:
                count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet /= len(
                    pairs_in_one_variant)
            else:
                count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet = -1

            # We need to safe the index of the variant with the highest normalized count.
            if count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet > max_normalized_count:
                max_normalized_count = count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet
                index_of_new_variant = idx

        # The variant with the highest normalized count will be added to our sample and...
        variant_added_to_the_sample = event_log_formated[index_of_new_variant].copy(
        )
        variant_added_to_the_sample[1] = 1
        allbehaviour_sample.append(variant_added_to_the_sample)
        variants_that_have_been_sampled.append(variant_added_to_the_sample[0])
        remainding_free_slots_in_sample -= 1

        # we remove the behaviour of the new variant (variant with the highest normalized count) from our unsampled behaviour list
        behaviour_of_the_new_variant_to_be_removed = list(zip(
            event_log_formated[index_of_new_variant][0], event_log_formated[index_of_new_variant][0][1:]))

        unsampled_behaviour = list((set(unsampled_behaviour)).difference(
            set(behaviour_of_the_new_variant_to_be_removed)))
    progress_bar.close()
    # We have all behaviour pairs of the event log in our sample. In case we have remaining free slots in our sample we try to improve the sample's representativeness by using the remainderplus algorithm

    for variant in tqdm(event_log_formated):
        if (variant[0] in variants_that_have_been_sampled):
            variant[1] = (variant[1] * sample_ratio) - 1
        else:
            variant[1] = variant[1] * sample_ratio

    event_log_formated = sorted(
        event_log_formated, key=lambda frequency: frequency[1], reverse=True)

    for variant in tqdm(event_log_formated, desc='Adding variants to the sample'):

        # append the variant to the sample if the variants expected occurrence is greater or equal to 1.
        # The sampled variant's occurrence is reassigned with the integer part of the calculated variants expected occurrence.
        if variant[1] >= 1:

            intnum = int(variant[1])
            remainder = variant[1] % intnum
            variant[1] = remainder
            remainding_free_slots_in_sample -= intnum

            # we have to check whether the variant is already part of our sample. In case the variant is already in our sample, we have to update the frequency of that variant in our sample,
            flag = False
            for already_sampled_variant in allbehaviour_sample:
                if variant[0] == already_sampled_variant[0]:
                    already_sampled_variant[1] += intnum
                    flag = True

            if flag == False:
                corpus = []
                corpus.append(variant[0])
                corpus.append(intnum)
                allbehaviour_sample.append(corpus)

 #################From here same as RemainderPlus-Sampling#############################

    # we have to sort on behaviour characteristics in case two or more variants have the same remainder. Therefore we first check which behaviour is undersampled, unsampled, oversampled or truly sampled.

    intermediate_results = _calculate_ratios(
        original_event_log=original_event_log, sampled_event_log=allbehaviour_sample)

    behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version = {tuple(
        x): y for x, y in intermediate_results.get("list_of_pairs_with_sample_ratio")}
    behaviour_pairs_in_original_log_with_count = {
        tuple(x): y for x, y in intermediate_results.get("list_of_pairs_with_count_ol")}

    # initiliaze every variant with rank = 0
    for variant in tqdm(event_log_formated, desc='Adding variants to the sample'):
        variant.append(0)

    progress_bar = tqdm(total=remainding_free_slots_in_sample, desc="Filling rest of slots with remainder plus sampling")

    while remainding_free_slots_in_sample > 0:
        progress_bar.update(1)
        # Calculate each  variant's rank...
        # The variant will get a high rank if a variant has a lot of undersampled behaviour compared to oversampled behaviour.
        # The rank of the variant will increase by one for each undersampled behaviour and will decrease by one for each oversampled behaviour.
        for variant in event_log_formated:

            rank = 0

            pairs_in_one_variant = list(zip(variant[0], variant[0][1:]))

            for pair in pairs_in_one_variant:

                if behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version.get(pair) > sample_ratio:
                    #pair is oversampled
                    rank -= 1
                if behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version.get(pair) < sample_ratio:
                    #pair is undersampled
                    rank += 1
                else:
                    # pair is perfectly sampled
                    rank += 0

            # normalize rank by the the number of pairs in the variant
            if (len(pairs_in_one_variant) > 0):
                rank = rank/len(pairs_in_one_variant)

            variant[2] = rank
        remainding_free_slots_in_sample -= 1

        # sort by two attributes. First by the remainder and then sort by the rank
        event_log_formated = sorted(
            event_log_formated, key=lambda x: (x[1], x[2]), reverse=True)

        # the variant that has to sampled next has to be on index 0 of the list (highest remainder and compared to all variants with the same remainder it has the highest rank)
        variant_to_be_sampled = event_log_formated[0]

        # update all corrsponding behaviour pairs:
        variants_pairs_to_be_sampled = list(
            zip(variant_to_be_sampled[0], variant_to_be_sampled[0][1:]))
        for pair in variants_pairs_to_be_sampled:
            behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version[pair] = ((behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version.get(
                pair) * behaviour_pairs_in_original_log_with_count.get(pair)) + 1)/behaviour_pairs_in_original_log_with_count.get(pair)

        # Add the variant to the sample. Therefore we have to check whether the variant is already part of our sample. In case the variant is already in our sample, we have to update the frequency of that variant in our sample,
        # Otherwise we add the variant to our sample with the frequency of one
        flag = False
        for already_sampled_variant in allbehaviour_sample:
            if variant_to_be_sampled[0] == already_sampled_variant[0]:
                already_sampled_variant[1] += 1
                flag = True
        if flag == False:
            corpus = []
            corpus.append(variant_to_be_sampled[0])
            corpus.append(1)
            allbehaviour_sample.append(corpus)
            flag = False

        # update the remainder because we have sampled the unique variant now
        variant_to_be_sampled[1] = 0

    progress_bar.close()
    # Converting to the desired format
    allbehaviour_sample = [(tuple(trace[0]), trace[1]) for trace in allbehaviour_sample]

    # print("SAMPLE IN OLD FORMAT")
    # print(allbehaviour_sample)
    
    


    ######################DATAFRAME CREATIO#################################################################
    
    dataframe_of_sample = _create_dataframe_for_sample(dataframe_of_the_original_log = dataframe_of_ol, sample = allbehaviour_sample, format = format, case_id_column = case_id_column, activity_column = activity_column, timestamp_column = timestamp_column, min_columns = min_columns) 

    return dataframe_of_sample
    

    

   



