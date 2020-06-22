import matplotlib.pyplot as plt
import itertools
import os
import numpy as np
import pandas as pd

from pathlib2 import Path
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_path=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=100)
        plt.close()
        
def summary(y_true, y_pred, target_names, save_root=None, epoch=0):
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(data=cm, columns=target_names, index=target_names)
    cm_string = df.to_string()
    summary_content = classification_report(y_true, y_pred, target_names=target_names)
    if save_root is None:
        print(summary_content)
        save_path=None
    else:
        save_root=Path(save_root)
        if not save_root.exists():
            os.makedirs(str(save_root))
        with open(str(save_root/"summary.txt"), "a+") as f:
            f.write(f"Epoch: {epoch}\n")
            f.write("=====================================================\n")
            f.write("[Confusion Matrix]\n")
            f.write(cm_string)
            f.write("\n\n[Summary]\n")
            f.write(summary_content)
            f.write("\n\n")
            if not (save_root/"cm").exists():
                os.makedirs(str(save_root/"cm"))
            save_path = str(save_root/"cm"/f"epoch_{epoch}.png")
    plot_confusion_matrix(cm           = cm,
                          normalize    = False,
                          target_names = target_names,
                          title        = "Confusion Matrix",
                          save_path    = save_path)