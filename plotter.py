from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.tree import export_graphviz
from sklearn import tree

def write_chart(forest, est, feature_names=None, class_names=None):
    tree_numbers = [10, 20, 50, 100, 200, 500, 1000]
    for num in tree_numbers:
        i = 0
        for tree in forest:
            if (i == num):
                break
            # TODO: add feature_names = feature_names, class_names = class_names,
            export_graphviz(tree, out_file='tree.dot',  rounded = True, proportion = False, precision = 2, filled = True)
            i += 1

        filename = 'tree-'+str(est)+'-estimators-' + str(num) + '-trees.png'

        from subprocess import call
        call(['dot', '-Tpng', 'tree.dot', '-o', 'results/'+filename, '-Gdpi=72'])