""" MLW_sr.py results:
N:  500
Time:  18.31296420097351
The mean cost:  6444.5
rRMSE:  0.382511403641432

N:  1000
Time:  39.48930501937866
The mean cost:  13630.0
rRMSE:  0.2565747316233734

N:  5000
Time:  409.39933943748474
The mean cost:  70715.0
rRMSE:  0.11308309851958524

N:  10000
Time:  1603.2839212417603
The mean cost:  142000.0
rRMSE:  0.0921778958333096
"""

import matplotlib.pyplot as plt

cost = [6444.5, 13630.0, 70715.0, 142000.0]
rrmse = [0.382511403641432, 0.2565747316233734, 0.11308309851958524, 0.09217789583330968]

plt.plot(rrmse, cost, 'o-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('relative error')
plt.ylabel('Cost')
plt.show()