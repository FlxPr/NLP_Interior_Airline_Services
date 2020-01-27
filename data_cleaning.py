import pandas as pd

df = pd.read_csv('skytrax_reviews_data.csv')
corrs = df.corr()
