import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generateFeatureCDF(name, label ,xlim, output):
  # Load data
  df = pd.read_csv('data/notjs.csv')
  df = df.loc[:, ['script_name','method_name', 'label', name]]
  # Map the labels
  label_mapping = {0: 'non-tracking', 1: 'tracking'}
  df['label'] = df['label'].replace(label_mapping)

  # Create the CDF plot with y-axis ranging from 0 to 1
  kdep = sns.kdeplot(data=df, x=name, hue='label',
      cumulative=True, common_norm=False, common_grid=True)

  plt.xlim(0, xlim)
  plt.xlabel(label)
  plt.show()
  plt.savefig('plots/'+ output + '.png')

generateFeatureCDF('Feature 23', 'Number of Storage Access',200, 'Figure_7_b')