import json
from collections import OrderedDict
import numpy as np
# file available at
#https://github.com/de-Boer-Lab/DREAM-2022/blob/main/sample_submission.json
import pandas as pd

with open('sample_submission.json', 'r') as f:
    ground = json.load(f)
indices = np.array([int(indice) for indice in list(ground.keys())])
PRED_DATA = OrderedDict()
import random

Y_pred=(list(np.random.randint(low = 0,high=18,size=71103)))

import pandas as pd

df=pd.read_csv('r2_53_finetuned_prediction.csv')
print(df.head())
Y_pred= list(df['Predicted_Exp'])

print(len(Y_pred))
for i in indices:
#Y_pred is an numpy array of dimension (71103,) that contains your
#predictions on the test sequences
    PRED_DATA[str(i)] = float(Y_pred[i])
def dump_predictions(prediction_dict, prediction_file):
    with open(prediction_file, 'w') as f:
        json.dump(prediction_dict, f)
dump_predictions(PRED_DATA, 'pred.json')