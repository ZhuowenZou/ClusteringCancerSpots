import xlrd
import numpy as np
import math

# read data from xlsx file
def read_data(filename):
    file = xlrd.open_workbook(filename)
    sheet = file.sheet_by_index(0)

    x = np.asarray(sheet.col_values(0)[1:])
    y = np.asarray(sheet.col_values(1)[1:])
    #d = np.asarray(sheet.col_values(4)[1:])

    return x,y#,d

# labels -> f[n] (= number of clusters of size n), and number of outliers
def clst_stats(labels):

    clusters = dict()
    num_out = 0
    for label in labels:
        if label != -1:
            if label in clusters:
                clusters[label] += 1
            else:
                clusters[label] = 1
        else:
            num_out += 1
    max_size = max( clusters.values() )
    stats = np.zeros(max_size + 1)
    for size in clusters.values():
        stats[size] += 1

    return stats, num_out

def SSE(nodes, labels, centroids):

    def dist(a, b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2
    
    # find the real center
    clsts = dict()
    error = 0
    for i in range(len(nodes)):
        if labels[i] in clsts:
            clsts[labels[i]].append(nodes[i])
        else:
            clsts[labels[i]] = [nodes[i]]
    for clst in clsts.values():
        mean = np.mean(np.asarray(clst), axis = 0)
        for member in clst:
            error += dist(member, mean)
    
    #for i in range(len(nodes)):
    #    if labels[i] != -1:
    #        error += dist( nodes[i], centroids[labels[i]])
    return error

def generate_colorscheme(size):
    val = "0123456789abcdef"
    cmap = []
    for i in range(size):
        s = "#"
        for i in range(6):
            s = s + val[np.random.randint(16)]
        cmap.append(s)
    return cmap

