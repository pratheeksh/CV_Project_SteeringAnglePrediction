import matplotlib.pyplot as plt
import sys
i = 1
lr = .00001
plt.suptitle('Train and Validation losses for RESNET Network', fontsize=12)
for filename in sys.argv[1:]:
    train = []
    val = []
    ax = plt.subplot(len(sys.argv)-1, 1, i)
    i += 1
    for row in open(filename, 'rb'):
        if "Train" in row:
            train.append(float(row.split()[3].split(';')[0]))
        if "Val" in row:
            val.append(float(row.split()[3].split(';')[0]))
    x = range(len(train))
    assert (len(x) == len(val))
    ax.plot(x, train, label="Train Loss")
    ax.plot(x, val, label="Validation Loss", color='red')
    ax.set_title("LR: " + str(lr))
    lr = lr * 10
    ax.legend()
plt.show()
