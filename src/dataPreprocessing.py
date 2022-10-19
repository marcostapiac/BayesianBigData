import csv
import sys

sys.path.extend(['../utils'])
from math_functions import append, vectorise
import numpy as np


def main():
    """ Return dataset, and its size """
    file = open('../covtype.csv')
    csvreader = csv.reader(file)
    ys = np.array(np.empty(shape=1))
    xs = np.array(np.empty((1,9,)))
    for row in csvreader:
        ys = np.vstack([ys, [float(row[9])]])
        x = vectorise(np.squeeze([float(x) for x in row[:9]]))
        xs = np.vstack([xs, x.reshape(1, x.shape[0],)])
    ys = ys[1:]
    xs = xs[1:]
    file.close()
    with open('GammaRegression.csv', 'w') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(np.squeeze(ys))
        for x in xs:
            writer_object.writerow(x)
        f_object.close()

if __name__ == "__main__":
    main()
