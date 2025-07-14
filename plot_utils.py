import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_object(obj, label=0):
    fig, ax = plt.subplots(figsize=(3,3))
    
    ax.imshow(obj, origin='lower', cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    
    if label:
        ax.text(0.9, 0.1, 'n = ' + str(label), color='white', ha='right', va='bottom', transform=ax.transAxes)
        
    return fig


def plot_object_and_pattern(obj, diff, label=0):
    fig, ax = plt.subplots(1,3, figsize=(6.25,3), gridspec_kw={'width_ratios': [1,1,0.05]})
    fig.subplots_adjust(wspace=0.1)
    
    cmap = plt.cm.bone
    vmax = 10**np.round(np.log10(diff.max()))
    norm = mpl.colors.LogNorm(vmin=1., vmax=vmax, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    ax[0].imshow(obj, origin='lower', cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    if label:
        ax[0].text(0.9, 0.1, 'n = ' + str(label), color='white', ha='right', va='bottom', transform=ax[0].transAxes)

    ax[1].imshow(diff, origin='lower', cmap=cmap, norm=norm)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.colorbar(sm, cax=ax[2])
    return fig


def plot_object_grid(objs, labels=None):
    fig, ax = plt.subplots(4,4, figsize=(8,8))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axes = ax.ravel()
    
    if len(objs) <= 16:
        indices = np.arange(len(objs))
    else:
        indices = np.random.randint(len(objs), size=16)

    for i, index in enumerate(indices):
        axes[i].imshow(objs[index], origin='lower', cmap='gray')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    for j in range(i + 1,16):
        axes[j].remove()
        
    try: len(labels)
    except: pass
    else:
        for i, index in enumerate(indices):
            axes[i].text(0.9, 0.1, 'n = ' + str(labels[index]), color='white', ha='right', va='bottom',
                         transform=axes[i].transAxes)
    return fig


def plot_pattern_grid(diffs, labels=None):
    fig, ax = plt.subplots(4,5, figsize=(8,8), gridspec_kw={'width_ratios': [1]*4 + [0.05]})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axes = ax[:,:-1].ravel()

    if len(diffs) <= 16:
        indices = np.arange(len(diffs))
    else:
        indices = np.random.randint(len(diffs), size=16)

    cmap = plt.cm.bone
    vmax = 10**np.round(np.log10(diffs.max()))
    norm = mpl.colors.LogNorm(vmin=1., vmax=vmax, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, index in enumerate(indices):
        axes[i].imshow(diffs[index], origin='lower', cmap=cmap, norm=norm)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    for j in range(i + 1,16):
        axes[j].remove()
        
    for i in range(3):
        ax[i,-1].remove()
    
    try: len(labels)
    except: pass
    else:
        for i, index in enumerate(indices):
            axes[i].text(0.9, 0.1, 'n = ' + str(labels[index]), color='white', ha='right', va='bottom',
                         transform=axes[i].transAxes)
        
    plt.colorbar(sm, cax=ax[-1,-1])
    return fig


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, ax = plt.subplots(1,2, figsize=(7,3))
    fig.subplots_adjust(wspace=0.3)
    ax[0].plot(range(1, len(train_losses) + 1), train_losses, label='Train.')
    ax[0].plot(range(1, len(val_losses) + 1), val_losses, label='Val.')
    ax[0].set_yscale('log')

    ax[1].plot(range(1, len(train_accuracies) + 1), train_accuracies)
    ax[1].plot(range(1, len(val_accuracies) + 1), val_accuracies)

    ax[0].legend(frameon=False)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    return fig


def plot_classification_statistics(test_probabilities, incorrect, correct):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.boxplot((test_probabilities[incorrect], test_probabilities[correct]))
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Incorrect', 'Correct'])
    ax.set_ylabel('Probability')
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(1,2, figsize=(3.15,3), gridspec_kw={'width_ratios': [1,0.05]})
    fig.subplots_adjust(wspace=0.1)

    cmap = plt.cm.bone
    vmax = cm.max()
    norm = plt.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    ax[0].imshow(cm, cmap=cmap, norm=norm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j] < 0.3*cm.max():
                color = 'white'
            else:
                color = 'black'
            ax[0].text(j,i, cm[i,j], ha='center', va='center', color=color)

    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(range(2,6))
    ax[0].set_yticks(range(4))
    ax[0].set_yticklabels(range(2,6))
    ax[0].set_xlabel('Predicted aggregate size')
    ax[0].set_ylabel('True aggregate size')
    
    plt.colorbar(sm, cax=ax[1])
    return fig