import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('auto-mpg.csv')


train, test = train_test_split(df, test_size=0.1, random_state=22)

train.to_csv('train.csv', index = False)
test.to_csv('test.csv', index = False)