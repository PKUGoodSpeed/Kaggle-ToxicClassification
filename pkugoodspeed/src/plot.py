## Visualization
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from IPython.display import display
import seaborn as sns

def plotResult(res, filename="../output/rslt.png"):
    train_accu = res.history['acc']
    train_loss = res.history['loss']
    test_accu = res.history['val_acc']
    test_loss = res.history['val_loss']
    steps = [i for i in range(len(train_accu))]
    
    
    print("VISUALIZATION:")
        ## Plotting the results
    fig, axes = plt.subplots(1,2, figsize = (12, 6))
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)

    axes[0].set_title('Loss')
    axes[0].plot(steps, train_loss, label = 'train loss')
    axes[0].plot(steps, test_loss, label = 'test loss')
    axes[0].set_xlabel('# of steps')
    axes[0].legend()

    axes[1].set_title('Accuracy')
    axes[1].plot(steps, train_accu, label = 'train accuracy')
    axes[1].plot(steps, test_accu, label = 'test accuracy')
    axes[1].set_xlabel('# of steps')
    axes[1].legend()
    
    plt.savefig(filename)