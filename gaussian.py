import pandas as pd
import random
import diffprivlib.mechanisms

data = [[random.uniform(1.0, 30.0) for _ in range(6)] for _ in range(100000)]
df = pd.DataFrame(data, columns=[f"col{i}" for i in range(1, 7)])

e = 1.0
d = 0.01
s = 1.0

gau = diffprivlib.mechanisms.GaussianAnalytic(epsilon=e, delta=d, sensitivity=s)

for col in df.columns:
    df[col] = df[col].apply(gau.randomise)

e = 1.0
s = 1.0

lap = diffprivlib.mechanisms.Laplace(epsilon=e, sensitivity=s)

for col in df.columns:
    df[col] = df[col].apply(lap.randomise)

for col in df.columns:
    print(df[col])