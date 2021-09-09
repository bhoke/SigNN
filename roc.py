import pandas as pd
from numpy import arange
import matplotlib.pyplot as plt

if __name__ == "__main__":

    myData = pd.read_csv('results.csv')
    threshold_vals = arange(0.4, 0.9, 0.05)
    samePerson = myData[myData['PERSON_ID_1'] == myData['PERSON_ID_2']]
    difPerson = myData[myData['PERSON_ID_1'] != myData['PERSON_ID_2']]
    TP_arr = []
    FP_arr = []
    for thresh in threshold_vals:
        TP = samePerson['SCORE'] > thresh
        FP = difPerson['SCORE'] > thresh
        tp_count = len(TP.to_numpy().nonzero()[0])
        fp_count = len(FP.to_numpy().nonzero()[0])
        TP_arr.append(tp_count)
        FP_arr.append(fp_count)

plt.scatter(FP_arr, TP_arr)
plt.show()


