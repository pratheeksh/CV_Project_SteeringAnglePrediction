import matplotlib.pyplot as plt
import sys
i = 1
for filename in sys.argv[1:]:
    train = []
    val = []
    ax = plt.subplot(2, 2, i)
    i += 1
    for row in open(filename, 'rb'):
        if "Train" in row:
            train.append(float(row.split()[3].split(';')[0]))
        if "Val" in row:
            val.append(float(row.split()[3].split(';')[0]))
    x = range(len(train))
    assert (len(x) == len(val))
    ax.plot(x, train, label="Train Loss")
    ax.scatter(x, val, label="Validation Loss", color='green', s=1)

plt.legend()
plt.show()
