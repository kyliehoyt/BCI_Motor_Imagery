import matplotlib.pyplot as plt
import pandas as pd
#csv header:
#["subject", "electrode", "session_id", "class1_accuracy", "class2_accuracy", "total_accuracy", "sample_bias"]
samples_data = pd.read_csv('sample_results.csv')
fig = plt.figure(figsize=(3, 2))
fig.suptitle("Sample Level Accuracy")
index = 1
for subject_id in range(4, 7):
    gel_header = ['Class_1', 'Class_2', 'Average', 'Bias']
    poly_header = ['Class_1', 'Class_2', 'Average', 'Bias']

    gel_data = [samples_data.iloc[subject_id - 4]['class1_accuracy'],
                samples_data.iloc[subject_id - 4]['class2_accuracy'],
                samples_data.iloc[subject_id - 4]['total_accuracy'],
                samples_data.iloc[subject_id - 4]['sample_bias']]
    poly_data = [samples_data.iloc[subject_id - 3]['class1_accuracy'],
                samples_data.iloc[subject_id - 3]['class2_accuracy'],
                samples_data.iloc[subject_id - 3]['total_accuracy'],
                 samples_data.iloc[subject_id - 3]['sample_bias']]
    loc = 320 + index
    plt.subplot(loc)
    index += 1

    plt.bar(gel_header, gel_data, width=0.5, align='center', color=['gray', 'gray', 'dimgray', 'lightcoral'])
    plt.title("Subject " + str(subject_id) + " Gel")

    loc = 320 + index
    plt.subplot(loc)
    index += 1
    plt.bar(poly_header, poly_data, width=0.5, align='center', color=['gray', 'gray', 'dimgray', 'lightcoral'])
    plt.title("Subject " + str(subject_id) + " Poly")

plt.show()

