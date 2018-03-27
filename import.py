import json
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json("MDP.log", lines=True)

df.plot(y='capacity', use_index=True)
df.plot(y='capacity', kind='box')

plt.show()

print(df)