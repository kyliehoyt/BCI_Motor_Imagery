import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

header = ["subject", "electrode", "correct", "incorrect", "no_decision"]
try:
    os.remove('subject_results.csv')
except:
    pass

with open('subject_results.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

electrodes = ['Gel', 'Poly']

for sub_idx in range(4, 7):
    for e_idx in range(0,2):
        print(" Running Subject: " + str(sub_idx) + " Electrode type: " + electrodes[e_idx])
        command = "python3 'Signal Processing.py' -subject " + str(sub_idx) + " -electrode " + electrodes[e_idx]
        print(command)
        os.system(command)

