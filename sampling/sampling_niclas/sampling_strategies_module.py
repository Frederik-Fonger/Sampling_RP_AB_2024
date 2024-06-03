import sys
import random
import math
from c_min_sampling.CminSampler import CminSampler

flag = False

sys.path.append(
    r'C:\Users\Niclas Nebelung\Desktop\Bachelorarbeit\ba-pm\Bachelorarbeit_code\modules')

flag = True
if flag == True:
    # we are able to import the evaluation_module if we have changed our system path
    import sampling_niclas.evaluation_module


def remainder_plus_sampling(event_log, sample_ratio, num_of_traces_in_ol):
    """Gets an event log and returns a representative sample. The algorthm tries to   

    Args:
        event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
        sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
        num_of_traces_in_ol (Int): The number of traces in the (original) event log

    Returns:
        list: a list of variants with their respective occourences 
    """

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
    for variant in event_log_sorted:
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

    intermediate_results = sampling_niclas.evaluation_module.caluclate_matrices_and_metrics(
        original_event_log=original_event_log, sampled_event_log=remainder_plus_sample, sample_bandwidth=0.0)

    behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version = {tuple(
        x): y for x, y in intermediate_results.get("list_of_pairs_with_sample_ratio")}
    behaviour_pairs_in_original_log_with_count = {
        tuple(x): y for x, y in intermediate_results.get("list_of_pairs_with_count_ol")}

    # initiliaze every variant with rank = 0
    for variant in event_log_sorted:
        variant.append(0)

    while remaining_free_slots_in_sample > 0:

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

    # Conversion to the right syntax
    for variant in remainder_plus_sample:
        for idx, x in enumerate(variant):
            if idx == 0:
                variant[idx] = tuple(variant[idx])
    remainder_plus_sample = [tuple(x) for x in remainder_plus_sample]

    return remainder_plus_sample


def allbehaviour_Sampling(event_log, sample_ratio, num_of_traces_in_ol):
    """Gets an event log and returns a sample which reduces unsampled behaviour 

    Args:
        event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
        sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
        num_of_traces_in_ol (Int): The number of traces in the (original) event log

    Returns:
        list: a list of variants with their respective occurrences 
    """

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
    sample = []
    sample.append(first_variant_added_to_the_sample)

    # reduce the remainding free spots in the sample by one
    remainding_free_slots_in_sample = num_of_traces_in_sl-1

    # Determine the unsampled behaviour...Therefore check which behaviour is not part of our sample yet.
    metrics = sampling_niclas.evaluation_module.caluclate_matrices_and_metrics(
        original_event_log=event_log, sampled_event_log=sample, sample_bandwidth=0.2)

    # The unsampled behaviour list is a list of behaviour pairs that are part of the event log and haven't been added to the sample yet.
    unsampled_behaviour = [tuple(x[0])
                           for x in metrics.get("unsampled_behavior_list")]

    # keep track of the variants that have been already sampled
    variants_that_have_been_sampled = []
    variants_that_have_been_sampled.append(
        first_variant_added_to_the_sample[0])

    while remainding_free_slots_in_sample > 0 and len(unsampled_behaviour) > 0:

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
        sample.append(variant_added_to_the_sample)
        variants_that_have_been_sampled.append(variant_added_to_the_sample[0])
        remainding_free_slots_in_sample -= 1

        # we remove the behaviour of the new variant (variant with the highest normalized count) from our unsampled behaviour list
        behaviour_of_the_new_variant_to_be_removed = list(zip(
            event_log_formated[index_of_new_variant][0], event_log_formated[index_of_new_variant][0][1:]))

        unsampled_behaviour = list((set(unsampled_behaviour)).difference(
            set(behaviour_of_the_new_variant_to_be_removed)))

    # We have all behaviour pairs of the event log in our sample. In case we have remaining free slots in our sample we try to improve the sample's representativeness by using the remainderplus algorithm

    for variant in event_log_formated:
        if (variant[0] in variants_that_have_been_sampled):
            variant[1] = (variant[1] * sample_ratio) - 1
        else:
            variant[1] = variant[1] * sample_ratio

    event_log_formated = sorted(
        event_log_formated, key=lambda frequency: frequency[1], reverse=True)

    for variant in event_log_formated:

        # append the variant to the sample if the variants expected occurrence is greater or equal to 1.
        # The sampled variant's occurrence is reassigned with the integer part of the calculated variants expected occurrence.
        if variant[1] >= 1:

            intnum = int(variant[1])
            remainder = variant[1] % intnum
            variant[1] = remainder
            remainding_free_slots_in_sample -= intnum

            # we have to check whether the variant is already part of our sample. In case the variant is already in our sample, we have to update the frequency of that variant in our sample,
            flag = False
            for already_sampled_variant in sample:
                if variant[0] == already_sampled_variant[0]:
                    already_sampled_variant[1] += intnum
                    flag = True

            if flag == False:
                corpus = []
                corpus.append(variant[0])
                corpus.append(intnum)
                sample.append(corpus)

 ################# same as RemainderPlus-Sampling#############################

    # we have to sort on behaviour characteristics in case two or more variants have the same remainder. Therefore we first check which behaviour is undersampled, unsampled, oversampled or truly sampled.

    intermediate_results = sampling_niclas.evaluation_module.caluclate_matrices_and_metrics(
        original_event_log=original_event_log, sampled_event_log=sample, sample_bandwidth=0.0)

    behaviour_pairs_with_behaviour_pair_sample_ratio_dict_version = {tuple(
        x): y for x, y in intermediate_results.get("list_of_pairs_with_sample_ratio")}
    behaviour_pairs_in_original_log_with_count = {
        tuple(x): y for x, y in intermediate_results.get("list_of_pairs_with_count_ol")}

    # initiliaze every variant with rank = 0
    for variant in event_log_formated:
        variant.append(0)

    while remainding_free_slots_in_sample > 0:

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
        for already_sampled_variant in sample:
            if variant_to_be_sampled[0] == already_sampled_variant[0]:
                already_sampled_variant[1] += 1
                flag = True
        if flag == False:
            corpus = []
            corpus.append(variant_to_be_sampled[0])
            corpus.append(1)
            sample.append(corpus)
            flag = False

        # update the remainder because we have sampled the unique variant now
        variant_to_be_sampled[1] = 0

    return sample


def cmin_sampling(df_of_event_log, sample_ratio, num_of_traces_in_ol):
    """Gets a dataframe from an event log and returns a representative sample by reducing the Earth Movers Distance  

    Args:
        df_of_event_log (Datframe): Dataframe of the event log
        sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
        num_of_traces_in_ol (Int): The number of traces in the (original) event log

    Returns:
        list: a list of variants with their respective occourences 
    """

    # calculate the number of traces in the sample based on the given sample ratio
    num_of_traces_in_sl = round(sample_ratio * num_of_traces_in_ol)

    sampler = CminSampler(num_of_traces_in_sl)
    sampler.load_df(df_of_event_log, "case:concept:name", "concept:name")
    sample_unformated = sampler.sample(output="seq")

    sample_unformated = {x: sample_unformated.count(
        x) for x in sample_unformated}
    sample_formated = [(variant, occurrence)
                       for variant, occurrence in sample_unformated.items()]
    return sample_formated


def __half_to_even_rounding(num):
    """(Needed for Rounding values according to paper: Estimation and Analysis of the Quality of Event Log Samples for Process Discovery by Bart R. van Wensveen (for example needed for stratified squared sampling).
       Typical rounding rules except 0.5 is rounded to 0 
       Examples:
        0.1 --> 0
        0.4 --> 0
    !!! 0.5 --> 0
        0.6 --> 1
        1.2 --> 1
    !!! 1.5 --> 2
        2.5 --> 3
        3.4 --> 3

    Args:
       num (Float) : the number to bee rounded

    Returns:
        num: the rounded value according to paper: Estimation and Analysis of the Quality of Event Log Samples for Process Discovery by Bart R. van Wensveen
    """
    if num <= 0.5:
        return 0
    elif num < 1 and num > 0.5:
        return 1
    elif num % int(num) < 0.5:
        return int(num)
    else:
        return math.ceil(num)


def stratified_squared_sampling(event_log, sample_ratio, num_of_traces_in_ol):
    """Gets an event log and returns a representative sample
       Algorithm has been programmed according to paper: Estimation and Analysis of the Quality of Event Log Samples for Process Discovery by Bart R. van Wensveen

    Args:
        event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
        sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
        num_of_traces_in_ol (Int): The number of traces in the (original) event log

    Returns:
        list: a list of variants with their respective occourences 
    """

    # calculate the number of trace in the sample
    num_of_traces_in_sl = round(num_of_traces_in_ol * sample_ratio)

    # sort variants based on their frequency in descending order
    event_log_sorted = [[list(x), y] for x, y in sorted(
        event_log, key=lambda frequency: frequency[1], reverse=True)]

    # calculate every variant's expected occurence
    for variant in event_log_sorted:
        variant[1] = variant[1] * sample_ratio

    # build the sample successively
    sample = []

    list_of_variants_that_could_be_sampled = []

    remainding_free_slots_in_sample = num_of_traces_in_sl

    for idx, variant in enumerate(event_log_sorted):
        occurrence = __half_to_even_rounding(variant[1])
        if occurrence == 0:
            corpus = []
            corpus.append(variant[0])
            corpus.append(variant[1])
            list_of_variants_that_could_be_sampled.append(corpus)
        else:
            corpus = []
            corpus.append(variant[0])
            corpus.append(occurrence)
            sample.append(corpus)
            remainding_free_slots_in_sample -= occurrence

    # sort the variants that haven't been sampled yet based on their expected occurence
    list_of_variants_that_could_be_sampled = sorted(
        list_of_variants_that_could_be_sampled, key=lambda frequency: frequency[1], reverse=True)

    # add as many variants (occurrence = 1) as spaces left in the sample
    if remainding_free_slots_in_sample > 0:
        for variant in list_of_variants_that_could_be_sampled:
            corpus = []
            corpus.append(variant[0])
            corpus.append(1)
            sample.append(corpus)
            remainding_free_slots_in_sample -= 1
            if remainding_free_slots_in_sample == 0:
                break

    return sample


def random_sampling(event_log, sample_ratio):
    """Gets an event log and returns a random sample from the event log. The given sample ratio defines the size of the sample.
      in the sample.  

    Args:
        event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
        sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)

    Returns:
        list: a list of variants with their respective occourences 
    """

    # get the number of traces in the original event log
    num_of_traces_in_ol = sum(x[1] for x in event_log)

    # calculate the number of traces in the sample
    num_of_traces_in_sl = round(sample_ratio * num_of_traces_in_ol)

    list_of_all_traces_in_the_original_log = []
    for variant in event_log:
        occurrences_of_that_variant_in_the_event_log = variant[1]
        for i in range(occurrences_of_that_variant_in_the_event_log):

            list_of_all_traces_in_the_original_log.append(variant[0])

    random_sample = []
    list_of_indices_already_visited = []

    i = 1
    while i <= num_of_traces_in_sl:
        # get a random index
        random_index = random.randint(
            0, len(list_of_all_traces_in_the_original_log) - 1)

        while (random_index in list_of_indices_already_visited):
            random_index = random.randint(
                0, len(list_of_all_traces_in_the_original_log) - 1)

        list_of_indices_already_visited.append(random_index)

        # add a random trace to our sample
        random_sample.append(
            list_of_all_traces_in_the_original_log[random_index])
        i += 1

    # # convert the random traces into the form we need (currently highly inefficient!)

    random_sample = list({x: random_sample.count(x)
                         for x in random_sample}.items())

    return random_sample


#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################################################################


# The following algorithms are not part of the thesis but could be worth to analyse further


# def stratified_plus_sampling(event_log, sample_ratio, num_of_traces_in_ol):
#     """Gets an event log and returns a representative sample
#     Algorithm has been programmed according to paper: Estimation and Analysis of the Quality of Event Log Samples for Process Discovery by Bart R. van Wensveen

#     Args:
#         event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
#         sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
#         num_of_traces_in_ol (Int): The number of traces in the (original) event log

#     Returns:
#         list: a list of variants with their respective occourences
#     """

#     # calculate the number of trace in the sample
#     num_of_traces_in_sl = round(num_of_traces_in_ol * sample_ratio)

#     # sort variants based on their frequency in descending order
#     event_log_sorted = [[list(x), y] for x, y in sorted(
#         event_log, key=lambda frequency: frequency[1], reverse=True)]

#     # calculate every variant's expected occurence
#     for variant in event_log_sorted:
#         variant[1] = variant[1] * sample_ratio

#     # build the sample successively
#     sample = []

#     list_of_variants_that_could_be_sampled = []

#     remainding_free_slots_in_sample = num_of_traces_in_sl

#     for idx, variant in enumerate(event_log_sorted):
#         occurrence = __half_to_even_rounding(variant[1])
#         if occurrence == 0:
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(1)
#             list_of_variants_that_could_be_sampled.append(corpus)
#         else:
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(occurrence)
#             sample.append(corpus)
#             remainding_free_slots_in_sample -= occurrence

#     # sort the variants that haven't been sampled yet based on their expected occurence
#     list_of_variants_that_could_be_sampled = sorted(
#         list_of_variants_that_could_be_sampled, key=lambda frequency: frequency[1], reverse=True)

#     print(list_of_variants_that_could_be_sampled)

#     list_of_variants_indices_already_visited = []
#     while remainding_free_slots_in_sample > 0:
#         random_index = random.randint(
#             0, len(list_of_variants_that_could_be_sampled)-1)
#         while random_index in list_of_variants_indices_already_visited:
#             random_index = random.randint(
#                 0, len(list_of_variants_that_could_be_sampled)-1)
#         sample.append(list_of_variants_that_could_be_sampled[random_index])
#         remainding_free_slots_in_sample -= 1

#     # Conversion step:
#     for variant_in_sample in sample:
#         for idx, x in enumerate(variant_in_sample):
#             if idx == 0:
#                 variant_in_sample[idx] = tuple(variant_in_sample[idx])
#     sample = [tuple(x) for x in sample]

#     return sample


# def existential_stratified_sampling(event_log, sample_ratio, num_of_traces_in_ol):
#     """Gets an event log and returns a sample that maximizes coverage.
#        Algorithm has been programmed according to paper: Estimation and Analysis of the Quality of Event Log Samples for Process Discovery by Bart R. van Wensveen

#     Args:
#         event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
#         sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
#         num_of_traces_in_ol (Int): The number of traces in the (original) event log

#     Returns:
#         list: a list of variants with their respective occourences
#     """

#     # calculate the number of trace in the sample
#     num_of_traces_in_sl = round(num_of_traces_in_ol * sample_ratio)

#     # sort variants based on their frequency in descending order
#     event_log_sorted = [[list(x), y] for x, y in sorted(
#         event_log, key=lambda frequency: frequency[1], reverse=True)]

#     # calculate every variant's expected occurence
#     for variant in event_log_sorted:
#         variant[1] = variant[1] * sample_ratio

#     # build the sample successively
#     sample = []

#     list_of_variants_that_could_be_sampled = []

#     for idx, variant in enumerate(event_log_sorted):
#         occurrence = __half_to_even_rounding(variant[1])
#         if occurrence == 0:
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(1)
#             list_of_variants_that_could_be_sampled.append(corpus)
#         else:
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(occurrence)
#             sample.append(corpus)

#     # sort the variants that haven't been sampled yet based on their expected occurence
#     list_of_variants_that_could_be_sampled = sorted(
#         list_of_variants_that_could_be_sampled, key=lambda frequency: frequency[1], reverse=True)

#     for variant in list_of_variants_that_could_be_sampled:
#         corpus = []
#         corpus.append(variant[0])
#         corpus.append(1)
#         sample.append(corpus)

#     return sample


# def remainder_sampling(event_log, sample_ratio, num_of_traces_in_ol):
#     """Gets an event log and returns a frequency based sample from the event log.

#     Args:
#         event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
#         sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
#         num_of_traces_in_ol (Int): The number of traces in the (original) event log

#     Returns:
#         list: a list of variants with their respective occourences
#     """

#     # change tuple elements to lists in order to be able to assign new values
#     event_log = [[list(x), y] for x, y in event_log]

#     # sort variants based on their frequency in descending order
#     event_log_sorted = sorted(
#         event_log, key=lambda element: element[1], reverse=True)

#     # calculate the future total number of traces in the sample:
#     num_of_traces_in_sl = round(num_of_traces_in_ol * sample_ratio)
#     # num_of_traces_in_sl = math.ceil(num_of_traces_in_ol * sample_ratio)

#     # we have to track the available free spaces left for the sample.
#     remaining_free_slots_in_sample = num_of_traces_in_sl

#     # In the beginning no trace is part of the sample. We build up the sample successively
#     remainder_sample = []

#     # iterate through every variant in the event log...
#     for variant in event_log_sorted:
#         # and calculate the variants' expected occurrences in the sample
#         the_variants_expected_occurrence = variant[1] * sample_ratio
#         #the_variants_expected_occurrence = (variant[1]/num_of_traces_in_ol) * num_of_traces_in_sl
#         variant[1] = the_variants_expected_occurrence
#         #variant[1] = (variant[1]/num_of_traces_in_ol) * num_of_traces_in_sl

#         # append the variant to the sample if the variants expected occurrence is greater or equal to 1.
#         # The sampled variant's occurrence is reassigned with the integer part of the calculated variants expected occurrence.
#         if variant[1] >= 1:
#             intnum = int(variant[1])
#             remainder = variant[1] % intnum
#             variant[1] = remainder
#             remaining_free_slots_in_sample -= intnum
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(intnum)
#             remainder_sample.append(corpus)

#     # now we have to sort based on the remainder
#     event_log_sorted = sorted(
#         event_log, key=lambda element: element[1], reverse=True)

#     # we have to check wheter the next variant is already part of our sample.
#     # In case the variant is already in our sample, we have to update the frequency of that variant in our sample
#     # Otherwise we add the variant to our sample
#     for variant in event_log_sorted[0:remaining_free_slots_in_sample]:
#         flag = False
#         for idx, variant_in_sample in enumerate(remainder_sample):

#             if variant[0] == variant_in_sample[0]:
#                 remainder_sample[idx][1] = remainder_sample[idx][1] + 1
#                 flag = True

#         if flag == False:
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(1)
#             remainder_sample.append(corpus)
#         # else:
#         #     flag = False

#     # Conversion to the right syntax
#     for variant_in_sample in remainder_sample:
#         for idx, x in enumerate(variant_in_sample):
#             if idx == 0:
#                 variant_in_sample[idx] = tuple(variant_in_sample[idx])
#     remainder_sample = [tuple(x) for x in remainder_sample]

#     return remainder_sample


# def frequentorrarebehaviour_sampling(event_log, sample_ratio, num_of_traces_in_ol, prioritize_frequent_behaviour):
#     """Gets an event log and returns a sample which contains primarily frequent behaviour if prioritize_frequent_behaviour is set to True.
#     A sample which contains primarily rare behaviour is returned if prioritize_frequent_behaviour is set to false.

#     Args:
#         event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
#         sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
#         num_of_traces_in_ol (Int): The number of traces in the (original) event log
#         prioritize_frequent_behaviour (Bool): Determines if frequent behaviour or rare behaviour is prioritized || True --> Prioritize frequent behaviour , False --> Prioritize rare behaviour

#     Returns:
#         Sample (list[tuple]): a list of variants with their respective occourences
#     """

#     ########## Behaviour extraction##################

#     # extract list of pairs (behaviour) in one variant
#     list_of_pairs = []
#     for variant in event_log:
#         for i in range(variant[1]):
#             pairs_in_one_variant = list(zip(variant[0], variant[0][1:]))
#             list_of_pairs.append(pairs_in_one_variant)

#     # iterate through all pairs in every variant and extract every pair. In addition determine the total number of all pairs
#     num_of_all_pairs = 0
#     list_of_pairs_formated = []
#     for pairs_in_one_variant in list_of_pairs:
#         for pair in pairs_in_one_variant:
#             list_of_pairs_formated.append(pair)
#             num_of_all_pairs += 1

#
#     # count the occurrences of every behaviour pair
#     list_of_pairs_formated_with_count = {x: list_of_pairs_formated.count(
#         x)/num_of_all_pairs for x in list_of_pairs_formated}

#     ####################################################

#     # we convert the event log to a dictionary for faster access to the variants occurrences
#     event_log_with_variants_occurrences_dictversion = {
#         (x): y for x, y in event_log}

#     event_log_with_variants_rank_listversion = [
#         [list(x), 0] for x, y in event_log]

#     # determine the number of traces in the sample
#     num_of_traces_in_sl = round(num_of_traces_in_ol * sample_ratio)

#     ########## Ranking the variants based on behaviour occurrences in the event log###########

#     # iterate through every variant and calculate each rank based on behaviour occurrences
#     for variant in event_log_with_variants_rank_listversion:
#         num_of_pairs_in_one_variant = 0
#         pairs_in_one_variant = list(zip(variant[0], variant[0][1:]))
#         num_of_pairs_in_one_variant = len(pairs_in_one_variant)

#         for pair in pairs_in_one_variant:
#             variant[1] = variant[1] + \
#                 list_of_pairs_formated_with_count.get(pair)

#         # Normalization step:
#         if (num_of_pairs_in_one_variant > 0):
#             variant[1] = variant[1]/num_of_pairs_in_one_variant

#     ########################################################################

#     # sort the variants based on their ranks
#     # lowest ranks on top --> prioritize variants with rare behaviour
#     # highest ranks on top --> prioritize variants with frequent behaviour
#     event_log_with_variants_rank_listversion = sorted(
#         event_log_with_variants_rank_listversion, key=lambda x: x[1], reverse=prioritize_frequent_behaviour)

#     for variant in event_log_with_variants_rank_listversion:
#         variant[1] = math.ceil(event_log_with_variants_occurrences_dictversion.get(
#             tuple(variant[0]))/num_of_traces_in_ol * num_of_traces_in_sl)

#     event_log_with_variants_rounded_up_expected_occourence = [
#         x for x in event_log_with_variants_rank_listversion]

#     # we keep track of the available free spaces left for the sample.
#     remainding_free_slots_in_sample = num_of_traces_in_sl

#     # we build up our sample successively
#     sample = []

#     for variant in event_log_with_variants_rounded_up_expected_occourence:

#         if variant[1] < remainding_free_slots_in_sample and remainding_free_slots_in_sample > 0:
#             sample.append(variant)
#             remainding_free_slots_in_sample -= variant[1]
#             continue
#         if variant[1] >= remainding_free_slots_in_sample and remainding_free_slots_in_sample > 0:
#             variant[1] = remainding_free_slots_in_sample
#             sample.append(variant)
#             remainding_free_slots_in_sample = 0
#             continue
#         if remainding_free_slots_in_sample == 0:
#             break

#     # Conversion to the right syntax
#     for variant_in_sample in sample:
#         for idx, x in enumerate(variant_in_sample):
#             if idx == 0:
#                 variant_in_sample[idx] = tuple(variant_in_sample[idx])
#     sample = [tuple(x) for x in sample]

#     return sample


# def similarity_sampling(event_log, sample_ratio, num_of_traces_in_ol, prioritize_frequent_behaviour):
#     """Gets an event log and returns a similarity based sample from the event log.

#     Args:
#         event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
#         sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
#         num_of_traces_in_ol (Int): The number of traces in the (original) event log
#         prioritize_frequent_behaviour (Bool): Determines if frequent behaviour or rare behaviour is prioritized || True --> Prioritize frequent behaviour , False --> Prioritize rare behaviour

#     Returns:
#         list: a list of variants with their respective occourences
#     """

#     ########## Behaviour extraction##################

#     # extract list of pairs (behaviour) in one variant
#     list_of_pairs = []
#     for variant in event_log:
#         for i in range(variant[1]):
#             pairs_in_one_variant = list(zip(variant[0], variant[0][1:]))
#             list_of_pairs.append(pairs_in_one_variant)

#     # iterate through all pairs in every variant and extract every pair. In addition determine the total number of all pairs
#     num_of_all_pairs = 0
#     list_of_pairs_formated = []
#     for pairs_in_one_variant in list_of_pairs:
#         for pair in pairs_in_one_variant:
#             list_of_pairs_formated.append(pair)
#             num_of_all_pairs += 1

#
#     # count the occurrences of every behaviour pair
#     list_of_pairs_formated_with_count = {x: list_of_pairs_formated.count(
#         x)/num_of_all_pairs for x in list_of_pairs_formated}

#     ####################################################

#     # we convert the event log to a dictionary for faster access to the variants occurrences
#     event_log_with_variants_occurrences_dictversion = {
#         (x): y for x, y in event_log}

#     event_log_with_variants_rank_listversion = [
#         [list(x), 0] for x, y in event_log]

#     # determine the final number of traces in the sample
#     num_of_traces_in_sl = round(num_of_traces_in_ol * sample_ratio)
#     #num_of_traces_in_sl = math.ceil(num_of_traces_in_ol * sample_ratio)

#     ########## Ranking the variants based on behaviour occurrences in the event log###########

#     # iterate through every variant and calculate each rank based on behaviour occurrences
#     for variant in event_log_with_variants_rank_listversion:
#         num_of_pairs_in_one_variant = 0
#         pairs_in_one_variant = list(zip(variant[0], variant[0][1:]))
#         num_of_pairs_in_one_variant = len(pairs_in_one_variant)

#         for pair in pairs_in_one_variant:
#             variant[1] = variant[1] + \
#                 list_of_pairs_formated_with_count.get(pair)

#         # Normalization step:
#         if (num_of_pairs_in_one_variant > 0):
#             variant[1] = variant[1]/num_of_pairs_in_one_variant

#     ########################################################################

#     # sort the variants based on their ranks
#     # lowest ranks on top --> prioritize variants with rare behaviour
#     # highest ranks on top --> prioritize variants with frequent behaviour
#     event_log_with_variants_rank_listversion = sorted(
#         event_log_with_variants_rank_listversion, key=lambda x: x[1], reverse=prioritize_frequent_behaviour)

#     ############# determine each variants frequency in the sample############

#     for variant in event_log_with_variants_rank_listversion:
#         variant[1] = event_log_with_variants_occurrences_dictversion.get(
#             tuple(variant[0]))/num_of_traces_in_ol * num_of_traces_in_sl

#     event_log_with_expected_occourences = [
#         x for x in event_log_with_variants_rank_listversion]

#     # we keep track of the available free spaces left for the sample.
#     remainding_free_slots_in_sample = num_of_traces_in_sl

#     # we build up our sample successively
#     similarity_sample = []

#     for variant in event_log_with_expected_occourences:

#         if variant[1] >= 1:
#             intnum = int(variant[1])
#             remainder = variant[1] % intnum
#             variant[1] = remainder
#             remainding_free_slots_in_sample -= intnum
#             # build the corpus for our sample
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(intnum)
#             similarity_sample.append(corpus)

#     # we have to check wheter the next variant is already part of our sample.
#     # In case the variant is already in our sample, we have to update the frequency of that variant in our sample
#     # Otherwise we add the variant to our sample
#     for variant in event_log_with_expected_occourences[0:remainding_free_slots_in_sample]:
#         flag = False
#         for idx, variant_in_sample in enumerate(similarity_sample):

#             if variant[0] == variant_in_sample[0] and variant[1] > 0:
#                 similarity_sample[idx][1] = similarity_sample[idx][1] + 1
#                 flag = True

#         if flag == False:
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(1)
#             similarity_sample.append(corpus)

#     # Conversion to the right syntax
#     for variant_in_sample in similarity_sample:
#         for idx, x in enumerate(variant_in_sample):
#             if idx == 0:
#                 variant_in_sample[idx] = tuple(variant_in_sample[idx])
#     similarity_sample = [tuple(x) for x in similarity_sample]

#     return similarity_sample


# # The Jaccard similarity is used to measure similarities between two sets. In our case we measure the similarity between
# # Mathematically, the calculation of Jaccard similarity is taking the ratio of set intersection over set union
# def jaccard_similarity(list1, list2):
#     """Gets two sets of behaviour pairs and calculates the Jaccard similarity.
#     Args:
#         event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
#         sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
#         num_of_traces_in_ol (Int): The number of traces in the (original) event log

#         Example:

#     Returns:
#         float: The similarity value with 0 <= similarity value <= 1
#     """
#     s1 = set(list1)
#     s2 = set(list2)
#     if (len(s1.union(s2)) == 0):
#         return 0
#     else:
#         return float(len(s1.intersection(s2)) / len(s1.union(s2)))


# # weiterentwicklung: wir schauen nur auf die elemente und vergleichen das mit dem was wir noch nicht haben länge der varianten hat kein einfluss
# # evtl. Rundung nochmal anpassen


# def jaccard_sampling(event_log, sample_ratio, num_of_traces_in_ol):
#     """Gets an event log and returns a sample based on jaccard similarity

#     Args:
#         event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
#         sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
#         num_of_traces_in_ol (Int): The number of traces in the (original) event log

#     Returns:
#         list: a list of variants with their respective occourences
#     """

#     # we have to convert the entries of the list to a format... lements of the event log  in order to be able to assign new values to the log.
#     # [(('a', 'b', 'c'), 12), (('a', 'b', 'e'), 4)] --> [[['a', 'b', 'c'], 12], [['a', 'b', 'e'], 4]]
#     event_log_formated = [[list(x), y] for x, y in event_log]

#     #result: [(('a', 'b', 'e', 'g', 'h'), 2), (('a', 'c', 'e', 'f', 'g', 'h'), 1), (('a', 'd', 'e', 'g', 'i'), 1)]

#     # sort the event log based on the variants' frequency in the event log
#     # nicht nötig glaube ich
#     event_log_formated = sorted(
#         event_log_formated, key=lambda x: x[1], reverse=True)

#     #num_of_traces_in_sl = math.ceil(sample_ratio * num_of_traces_in_ol)
#     num_of_traces_in_sl = round(num_of_traces_in_ol * sample_ratio)

#     # keep track of the remaining slots in the sample
#     remainding_free_slots_in_sample = num_of_traces_in_sl

#     # we build up the sample successively.
#     jaccard_sample = []

#     # calculate each variant's expected occurrence in the sample
#     for variant in event_log_formated:

#         #variant[1] = (variant[1]/num_of_traces_in_ol) * num_of_traces_in_sl
#         variant[1] = variant[1] * sample_ratio

#     # keep track of the variants that are not part of the sample and could be sampled
#     list_of_variants_that_could_be_sampled = []

#     # iterate through every variant in the event log and add the variant to the sample whenever the rounded expected occurrence of the corresponding variant is greater or equal to 1.
#     # If the rounded expected occurrence of the variant is smaller than 1 add the variant to the list of variants that could be sampled in the second iteration
#     for idx, variant in enumerate(event_log_formated):

#         # get the rounded expected occurrence of the variant
#         intnum = round(variant[1])

#         # we have to determine the remainder in order to know if we have rounded up or rounded down in case the variant's expected occurrence is greater or equal to 1
#         if variant[1] >= 1:
#             remainder = variant[1] % intnum
#         else:
#             remainder = variant[1]

#         if remainder == variant[1] and intnum >= 1:
#             # expected occurrence is greater or equal to 1 and was rounded up (intnum >=1)

#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(intnum)
#             if (remainding_free_slots_in_sample > 0):
#                 remainding_free_slots_in_sample -= intnum
#                 jaccard_sample.append(corpus)

#         elif remainder < 1 and intnum >= 1:  # es wurde abgerundet, aber es handelt sich bei der Variante um eine, die viele zugehörige Traces besitzt
#             # expected occurrence is greater or equal to 1 and was rounded down (intnum >= 1)

#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(intnum)
#             if (remainding_free_slots_in_sample > 0):
#                 remainding_free_slots_in_sample -= intnum
#                 jaccard_sample.append(corpus)

#         else:
#             # expected occurrence is smaller than 1. Add the variant to the list of variants that could be sampled in the second iteration
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(1)
#             list_of_variants_that_could_be_sampled.append(corpus)

#     print("Sample nach der ersten Iteration: ")
#     print(jaccard_sample)

#     ######## Behaviour pair extraction from all variants that have been sampled after the first iteration###########################
#     # Extract all behaviour pairs from the variants that have been sampled after the first iteration and add the behaviour pairs to a set. The
#     # set will include all behaviour pairs that have already been sampled.
#     # wahrscheinlich nicht nötig --> umbenennung möglich

#     list_of_pairs = []
#     for variant in jaccard_sample:
#         pairs_in_one_variant = list(zip(variant[0], variant[0][1:]))
#         list_of_pairs.append(pairs_in_one_variant)

#     # iterate through all pairs in every variant and extract every pair. In addition determine the total number of all pairs
#     list_of_pairs_formated = []
#     for pairs_in_one_variant in list_of_pairs:
#         for pair in pairs_in_one_variant:
#             list_of_pairs_formated.append(pair)

#     # add the behaviour pairs to a set.
#     sampled_set_of_behaviour = set(list_of_pairs_formated)

#     #####################################################################################################################

#     # We first iterate through every variant that haven't been sampled yet and safe each variant's behaviour pairs as a set.
#     # Then we compare the set of behaviour pairs for the corresponding variant with the set of behaviour pairs of all variants we have already added to our sample.
#     # We use the Jaccard-Similarity metric in order to compare the both sets of behaviour pairs.
#     # If both sets are identical, for example A = {a, b, c} and B = {a, b, c}, then their Jaccard similarity = 1.
#     # If sets A and B don't have common elements, for example, say A = {a, b, c} and B = {d, e, f}, then their Jaccard similarity = 0
#     # Example:

#     # Iterate through every variant that haven't been sampled yet
#     # besser mit while schleife wahrscheinlich
#     print(list_of_variants_that_could_be_sampled)
#     while remainding_free_slots_in_sample > 0:
#         print("\n\n")
#         print("remainding_free_slots_in_sample")
#         print(remainding_free_slots_in_sample)
#         highest_possible_similarity_value = 1
#         index_of_the_new_variant = 0

#         for idx, element in enumerate(event_log_formated):
#             # Determine the variant's behaviour pairs
#             variants_set_of_behaviour = set(zip(element[0], element[0][1:]))
#             print(element)
#             print(variants_set_of_behaviour)

#             # Compare the set of behaviour pairs for the corresponding variant with the set of behaviour pairs of all variants that we have already added to our sample.
#             # Compare the set of behaviour pairs by calculating the Jaccard similarity between the two sets.
#             similarity_to_current_sample = jaccard_similarity(
#                 sampled_set_of_behaviour, variants_set_of_behaviour)

#             print(similarity_to_current_sample)
#             print("\n")
#             # print("Variant:")
#             # print(element)
#             # # print("Score:")
#             # # print(similarity_to_current_sample)

#             # Check the calculated
#             if similarity_to_current_sample < highest_possible_similarity_value and not list(event_log_formated[idx]) in jaccard_sample:
#                 highest_possible_similarity_value = similarity_to_current_sample
#                 index_of_the_new_variant = idx
#                 new_variants_set_of_behaviour = variants_set_of_behaviour

#         # Add the variant that has the lowest similarity value
#         if (highest_possible_similarity_value < 1 and remainding_free_slots_in_sample > 0):
#             event_log_formated[index_of_the_new_variant][1] = 1

#             # add the variant to the sample
#             jaccard_sample.append(
#                 list(event_log_formated[index_of_the_new_variant]))
#             print("Diese Variant wird hinzugefügt: ")
#             print(list(event_log_formated[index_of_the_new_variant]))

#             # update the set of behaviour pairs that we have already added to our sample
#             sampled_set_of_behaviour = sampled_set_of_behaviour.union(
#                 new_variants_set_of_behaviour)

#             remainding_free_slots_in_sample -= 1
#         else:
#             raise Exception

#     # Conversion to the right syntax
#     for variant_in_sample in jaccard_sample:
#         for idx, x in enumerate(variant_in_sample):
#             if idx == 0:
#                 variant_in_sample[idx] = tuple(variant_in_sample[idx])
#     jaccard_sample = [tuple(x) for x in jaccard_sample]

#     return jaccard_sample


# def stratified_sampling(event_log, sample_ratio):
#     """Gets an event log and returns a stratified sample from the event log.

#     Args:
#         event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
#         sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)

#     Returns:
#         list: a list of variants with their respective occourences
#     """

#     # sort variants based on their frequency in descending order
#     event_log_sorted = [[list(x), y] for x, y in sorted(
#         event_log, key=lambda frequency: frequency[1], reverse=True)]

#     stratified_sample = []
#     for variant in event_log_sorted:
#         variant[1] = int(variant[1] * sample_ratio)
#         if (variant[1] >= 1):
#             stratified_sample.append(variant)

#     return stratified_sample


# def length_sampling(event_log, sample_ratio, num_of_traces_in_ol, prioritize_longer):
#     """Gets an event log and returns a length based sample from the event log.

#     Args:
#         event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
#         sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
#         num_of_traces_in_ol (Int): The number of traces in the (original) event log
#         prioritize_longer (Bool): Determines whether longer traces are prioritized or not || True --> Prioritize longer traces, False --> Prioritize shorter traces

#     Returns:
#         list: a list of variants with their respective occourences
#     """

#     # change tuple elements to lists in order to be able to assign new values
#     event_log = [[list(x), y] for x, y in event_log]

#     # we sort the variants based on their length (number of elements).
#     event_log_sorted_based_on_trace_length = sorted(
#         [[x, len(x)] for x, y in event_log], key=lambda element: element[1], reverse=prioritize_longer)

#     #num_of_traces_in_sample = math.ceil(num_of_traces_in_ol * sample_ratio)
#     num_of_traces_in_sample = round(num_of_traces_in_ol * sample_ratio)

#     for variant in event_log_sorted_based_on_trace_length:
#         for variant_with_frequency in event_log:
#             if variant[0] == variant_with_frequency[0]:
#                 variant[1] = (variant_with_frequency[1] /
#                               num_of_traces_in_ol) * num_of_traces_in_sample

#     length_sample = []
#     for variant in event_log_sorted_based_on_trace_length:
#         variant[1] = math.ceil(variant[1])
#         if sum(x[1] for x in length_sample) < num_of_traces_in_sample:
#             length_sample.append(variant)

#     return length_sample


# def jaccard_similarity(list1, list2):
#     s1 = set(list1)
#     s2 = set(list2)
#     if (len(s1.union(s2)) == 0):
#         return 0
#     else:
#         return float(len(s1.intersection(s2)) / len(s1.union(s2)))

# # weiterentwicklung: wir schauen nur auf die elemente und vergleichen das mit dem was wir noch nicht haben länge der varianten hat kein einfluss
# # evtl. Rundung nochmal anpassen


# def max_jaccard_distance(preprocessed_event_log, sample_ratio, num_of_traces_in_ol):
#     num_of_traces_in_sl = math.ceil(sample_ratio * num_of_traces_in_ol)

#     log = sorted(preprocessed_event_log, key=lambda x: x[1], reverse=True)

#     # ordered based on Jaccard distance
#     sampled_ordered_variants = []
#     sampled_ordered_variants.append(list(log[0]))

#     sampled_set_of_behaviour = set((zip(log[0][0], log[0][0][1:])))

#     for variant in log:
#         pairs_in_one_variant = set(zip(variant[0], variant[0][1:]))
#         #set1 = pairs_in_one_variant
#         # initialization with the highest possible value for jaccard similarity
#         smallest_similarity_value = 1
#         index_of_the_new_variant = 0
#         for idx, element in enumerate(log):
#             variants_set_of_behaviour = set(zip(element[0], element[0][1:]))
#             distance_to_current_sample = jaccard_similarity(
#                 sampled_set_of_behaviour, variants_set_of_behaviour)

#             if distance_to_current_sample <= smallest_similarity_value and not list(log[idx]) in sampled_ordered_variants:
#                 smallest_similarity_value = distance_to_current_sample
#                 index_of_the_new_variant = idx
#                 new_variants_set_of_behaviour = variants_set_of_behaviour

#         if smallest_similarity_value == 1:  # we havent found something
#             break
#         elif (smallest_similarity_value < 1):
#             sampled_ordered_variants.append(
#                 list(log[index_of_the_new_variant]))
#             sampled_set_of_behaviour = sampled_set_of_behaviour.union(
#                 new_variants_set_of_behaviour)

#     num_of_traces_in_sl = math.ceil(sample_ratio * num_of_traces_in_ol)
#     remainding_free_slots_in_sample = num_of_traces_in_sl
#     sample = []
#     # ########determine each variant's frequency:##############

#     for variant in sampled_ordered_variants:
#         variant[1] = (variant[1]/num_of_traces_in_ol) * num_of_traces_in_sl

#     for variant in sampled_ordered_variants:

#         if variant[1] >= 1:
#             intnum = int(variant[1])
#             remainder = variant[1] % intnum
#             variant[1] = remainder
#             remainding_free_slots_in_sample -= intnum
#             # build the corpus for our sample
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(intnum)
#             sample.append(corpus)
#             # sampled_ordered_variants.remove(variant)

#     for variant in sampled_ordered_variants[0:remainding_free_slots_in_sample]:
#         flag = False
#         for idx, variant_in_sample in enumerate(sample):

#             if variant[0] == variant_in_sample[0]:
#                 sample[idx][1] = sample[idx][1] + 1
#                 flag = True

#         if flag == False:
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(1)
#             sample.append(corpus)

#     return sample


# def stratified_plus_sampling_optimized(event_log, sample_ratio, num_of_traces_in_ol):
#     """Gets an event log and returns a sample which reduces unsampled behaviour

#     Args:
#         event_log (list[tuple[Tuple[str], List[Trace]]]): The event_log
#         sample_ratio (Float): The sample ratio with (0 < sample_ratio <= 1)
#         num_of_traces_in_ol (Int): The number of traces in the (original) event log

#     Returns:
#         list: a list of variants with their respective occurrences
#     """

#     #num_of_traces_in_sl = math.ceil(num_of_traces_in_ol * sample_ratio)
#     num_of_traces_in_sl = round(num_of_traces_in_ol * sample_ratio)

#     # bring the event log in the right form
#     # we have to convert the entries of the list to a format... lements of the event log  in order to be able to assign new values to the log.
#     # [(('a', 'b', 'c'), 12), (('a', 'b', 'e'), 4)] --> [[['a', 'b', 'c'], 12], [['a', 'b', 'e'], 4]]
#     event_log_formated = [[list(x), y] for x, y in event_log]

#     # calculate the variants expected occurrences.
#     for variant in event_log_formated:
#         #variant[1] = (variant[1]/num_of_traces_in_ol) * num_of_traces_in_sl
#         variant[1] = variant[1] * sample_ratio

#     remainding_free_slots_in_sample = num_of_traces_in_sl

#     sample = []

#     list_of_variants_that_could_be_sampled = []

#     # iterate through every variant in the event log and add the variant to the sample whenever the rounded expected occurrence of the corresponding variant is greater or equal to 1.
#     # If the rounded expected occurrence of the variant is smaller than 1 add the variant to the list of variants that could be sampled in the second iteration
#     for idx, variant in enumerate(event_log_formated):

#         # get the rounded expected occurrence of the variant
#         intnum = round(variant[1])

#         # we have to determine the remainder in order to know if we have rounded up or rounded down in case the variant's expected occurrence is greater or equal to 1
#         if variant[1] >= 1:
#             remainder = variant[1] % intnum
#         else:
#             remainder = variant[1]

#         if remainder == variant[1] and intnum >= 1:
#             # expected occurrence is greater or equal to 1 and was rounded up (intnum >=1)

#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(intnum)
#             if (remainding_free_slots_in_sample > 0):
#                 remainding_free_slots_in_sample -= intnum
#                 sample.append(corpus)

#         elif remainder < 1 and intnum >= 1:  # es wurde abgerundet, aber es handelt sich bei der Variante um eine, die viele zugehörige Traces besitzt
#             # expected occurrence is greater or equal to 1 and was rounded down (intnum >= 1)

#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(intnum)
#             if (remainding_free_slots_in_sample > 0):
#                 remainding_free_slots_in_sample -= intnum
#                 sample.append(corpus)

#         else:
#             # expected occurrence is smaller than 1. Add the variant to the list of variants that could be sampled in the second iteration
#             corpus = []
#             corpus.append(variant[0])
#             corpus.append(1)
#             list_of_variants_that_could_be_sampled.append(corpus)

#     print("Sample zwischenergebnis:")
#     print(sample)

#     print("Could be sampled:")
#     print(list_of_variants_that_could_be_sampled)

#
#     # compare the sample after the first iteration and determine the unsampled behaviour
#     metrics = evaluation_module.caluclate_matrices_and_metrics(
#         original_event_log=event_log, sampled_event_log=sample, sample_bandwidth=0.2)

#     # The unsampled behaviour list is a list of behaviour pairs that are part of the event log and haven't been added to the sample yet.
#     unsampled_behaviour = [tuple(x[0])
#                            for x in metrics.get("unsampled_behavior_list")]

#     print("unsampled_behaviour:")
#     print(unsampled_behaviour)

#     # keep track of the variants that have been already sampled in the second iteration
#     variants_that_have_been_sampled = []

#     while remainding_free_slots_in_sample > 0:
#         print("\n\n\n")

#         index_of_new_variant = 0

#         max_normailized_count = 0

#         # Iterate through every variant that haven't been sampled yet and determine the variant's behaviour pairs.
#         # For each variant count the number of behaviour pairs that haven't been added to the sample yet.
#         # Due to the fact that long variants (number of events in the sequence/trace is high) have a higher probability of having a high count,
#         # divide the count of behaviour pairs that are part of the variant but are not part of the sample yet by the length of the corresponding variant (normalization)
#         for idx, variant in enumerate(list_of_variants_that_could_be_sampled):

#             pairs_in_one_variant = list(zip(variant[0], variant[0][1:]))

#             count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet = 0

#             for pair in pairs_in_one_variant:
#                 if pair in unsampled_behaviour:
#                     count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet += 1

#             # normalization step
#             if count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet > 0:
#                 count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet /= len(
#                     pairs_in_one_variant)

#             print("Variant: ")
#             print(variant)
#             print("Score:")
#             print(
#                 count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet)
#             print("\n")
#             # We need to safe the index of the variant with the highest normalized count.
#             if count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet > max_normailized_count and not list_of_variants_that_could_be_sampled[idx] in variants_that_have_been_sampled:
#                 max_normailized_count = count_of_behaviour_pairs_that_are_part_of_the_variant_but_not_sampled_yet
#                 index_of_new_variant = idx

#         # The variant with the highest normalized count will be added to our sample and...
#         sample.append(
#             list_of_variants_that_could_be_sampled[index_of_new_variant])
#         variants_that_have_been_sampled.append(
#             list_of_variants_that_could_be_sampled[index_of_new_variant])
#         remainding_free_slots_in_sample -= 1

#         # we remove the behaviour of the new variant (variant with the highest normalized count) from our unsampled behaviour list
#         behaviour_of_the_new_variant_to_be_removed = list(zip(
#             list_of_variants_that_could_be_sampled[index_of_new_variant][0], list_of_variants_that_could_be_sampled[index_of_new_variant][0][1:]))
#         unsampled_behaviour = list(set(
#             behaviour_of_the_new_variant_to_be_removed).symmetric_difference(set(unsampled_behaviour)))

#     # Conversion to the right syntax
#     for variant_in_sample in sample:
#         for idx, x in enumerate(variant_in_sample):
#             if idx == 0:
#                 variant_in_sample[idx] = tuple(variant_in_sample[idx])
#     sample = [tuple(x) for x in sample]

#     return sample
