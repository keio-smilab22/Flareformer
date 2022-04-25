import csv
import numpy as np
import matplotlib.pyplot as plt


with open('loss.csv') as f:
    reader = csv.reader(f)
    losses = []
    for i,row in enumerate(reader):
        if i == 0: continue
        losses.append(float(row[1]))

losses = np.array(losses)
x = np.linspace(0,losses.shape[0],losses.shape[0])
print(np.log(losses))
print(losses.shape,x.shape)
# plt.plot(x,losses)
plt.plot(x,np.log(losses))
plt.show()