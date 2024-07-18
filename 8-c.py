import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generateSurrogateCDF(name1, name2, label, xlim, output):
  df = pd.read_csv("data/surrogate-generate-cdf.csv")

  # Melt the DataFrame to make it suitable for sns.ecdfplot
  df_melted = df.melt(id_vars=['website'], value_vars=[name1, name2])

  # Create the CDF plot
  kdep = sns.ecdfplot(data=df_melted, x='value', hue='variable')
  plt.xlabel(label)
  plt.ylabel('CDF')
  plt.xlim(0, xlim)
  # plt.title('CDF of functional requests')
  plt.show()
  plt.savefig('plots/' + output + '_cdf.png')

generateSurrogateCDF('normal_functional requests', 'surr_functional requests', 'Number of Non-tracking Requests', 1000, 'Figure_8_c')
