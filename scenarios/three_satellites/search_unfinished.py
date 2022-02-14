import os
import pandas as pd
import numpy as np
import re
import pickle

root_dir = os.path.join("results", "three_satellites")
length_list = np.linspace(0, 8800e3, num=96)
variations = np.linspace(-0.4, 0.4, num=96 + 1)
variations_plus = np.linspace(0, 0.4, num=96 // 2 + 1)


def check_unfinished(df):
    if 0.4 in df.index and np.allclose(df.index == variations):
        # special case for case 9
        return False
    if 8800e3 in df.index:
        return False
    if df["key_per_time"].iloc[-1] < 0:
        return False
    return True


def generate_case_tuple(subdir, data):
    match = re.search("sat_positions", subdir)
    if match:
        case_number = 0
        subcase = float(re.search("(?P<subcase>[0-9]\.[0-9]*)_first_sat", subdir).group("subcase"))
        length = length_list[len(data.index)]
        return case_number, subcase, length
    match = re.search("divergence_theta/(?P<case_number>[0-9]*)/", subdir)
    if match:
        case_number = int(match.group("case_number"))
        subcase = float(re.search("(?P<subcase>[0-9]\.[0-9]*)_first_sat", subdir).group("subcase"))
        length = length_list[len(data.index)]
        return case_number, subcase, length
    match = re.search("memories/(?P<case_number>[0-9]*)/", subdir)
    if match:
        case_number = int(match.group("case_number"))
        subcase = float(re.search("(?P<subcase>[0-9]*)_t_dp", subdir).group("subcase")) / 1000
        length = length_list[len(data.index)]
        return case_number, subcase, length
    match = re.search("orbital_heights", subdir)
    if match:
        case_number = 7
        subcase = int(re.search("(?P<subcase>[0-9]*)_orbital_height", subdir).group("subcase")) * 1000
        length = length_list[len(data.index)]
        return case_number, subcase, length
    match = re.search("cutoff_multiplier", subdir)
    if match:
        case_number = 8
        submatch = re.search("(?P<subcase>[A-Za-z0-9]*)_cutoff_multiplier", subdir)
        try:
            subcase = int(submatch.group("subcase")) / 100
        except ValueError as e:
            if submatch.group("subcase") == "None":
                subcase = None
            else:
                print(submatch.group("subcase"))
                raise e
        length = length_list[len(data.index)]
        return case_number, subcase, length
    match = re.search("satellite_path", subdir)
    if match:
        case_number = 9
        submatch = re.search("(?P<subcase>-?[0-9]+)_configuration", subdir)
        label = submatch.submatch.group("subcase")
        configuration_dict = {"-1": np.array([-0.1, 0.5, 1.1]),
                              "0": np.array([0, 0.5, 1]),
                              "1": np.array([0.1, 0.5, 0.9]),
                              "2": np.array([0.2, 0.5, 0.8])
                              }
        base_multipliers = configuration_dict[label]
        variation_index = (len(data.index) - 1) // 2 + 1
        new_variation = variations_plus[variation_index]
        subcase = (base_multipliers, new_variation, label)
        length = 4400e3
        return case_number, subcase, length


fourlink_tuples = []
twolink_downlink_tuples = []
for subdir, dirs, files in os.walk(root_dir):
    if re.search("archive", subdir) or re.search("explore", subdir):
        continue
    for file in files:
        if file == "result.csv":
            data = pd.read_csv(os.path.join(subdir, file), index_col=0)
            if check_unfinished(data):
                case_tuple = generate_case_tuple(subdir, data)
                if re.search("twolink_downlink", subdir):
                    twolink_downlink_tuples += [case_tuple]
                elif re.search("fourlink", subdir):
                    fourlink_tuples += [case_tuple]
                print(case_tuple)

with open(os.path.join("scenarios", "three_satellites", "twolink_downlink_tuples.pickle"), "wb") as f:
    pickle.dump(twolink_downlink_tuples, f)
with open(os.path.join("scenarios", "three_satellites", "fourlink_tuples.pickle"), "wb") as f:
    pickle.dump(fourlink_tuples, f)

print("Twolink_downlink_cases: ", len(twolink_downlink_tuples))
print("Fourlink_cases: ", len(fourlink_tuples))
