import pandas as pd
import seaborn as sns

data = {'y_Predicted': [1,1,0,1,0,1,1,0,1,0,0,0],
        'y_Actual': [1,0,0,1,0,1,0,0,1,0,1,0]}

df = pd.DataFrame(data, columns=['y_Predicted', 'y_Actual'])
df

confusion_matrix = pd.crosstab(df.loc[:, 'y_Actual'], df.loc[:, 'y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True)
confusion_matrix

sns.heatmap(confusion_matrix, annot=True)
