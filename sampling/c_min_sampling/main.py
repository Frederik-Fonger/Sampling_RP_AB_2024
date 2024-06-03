
import pandas as pd
import pm4py
from CminSampler import CminSampler


event_log = pm4py.read_xes("Sepsislog.xes")
df = pm4py.convert_to_dataframe(event_log)


# event_log_pm4py = pm4py.convert_to_event_log(df)
event_log_pm4py = df
variants_pm4py = pm4py.get_variants_as_tuples(event_log_pm4py)

num_f_traces_in_ol = 0
for variant in variants_pm4py:
    # variants_pm4py[variant] = len(variants_pm4py[variant])
    num_f_traces_in_ol += variants_pm4py[variant]


sample_ratio = 0.01
num_f_traces_in_sl = round(sample_ratio * num_f_traces_in_ol)
print("num_of_traces_in_ol: ")
print(num_f_traces_in_ol)


sampler = CminSampler(num_f_traces_in_sl)
sampler.load_df(df, "case:concept:name", "concept:name")
sample_unformated = sampler.sample(output="seq")


sample_unformated = {x: sample_unformated.count(
    x) for x in sample_unformated}

sample_formated = [(variant, occurrence)
                   for variant, occurrence in sample_unformated.items()]

print(sample_formated)
