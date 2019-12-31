import xlrd
import xlsxwriter
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

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


def plotter(clst_stats, eps, clst_type, data_type, n_from, n_to, num_clusters):
    figwidth = 5 * (n_to - n_from) // 50
    plt.figure(figsize=(figwidth, 5))

    agg_stats = clst_stats[n_from: n_to + 1]
    while len(agg_stats) != (n_to - n_from + 1):
        agg_stats = np.append(agg_stats, [0])

    xi = list(range(n_from, n_to + 1, 1))
    plt.bar(xi, agg_stats)
    plt.xlabel('Cluster size')
    plt.ylabel('Number of clusters')
    title = 'Number of clusters of sizes from ' + str(n_from) + " to " + str(
        n_to) + " in " + clst_type + " for " + data_type
    plt.title(title)
    plt.savefig('plot/' + clst_type + "_" + str(eps) + "_" + data_type + ".png")
    plt.clf()
    #plt.show()


def weighted_plotter(clst_stats, eps, clst_type, data_type, n_from, n_to, num_clusters):
    figwidth = 5 * (n_to - n_from) // 50
    plt.figure(figsize=(figwidth, 5))

    agg_stats = [i * clst_stats[i] for i in range(n_from, min(n_to + 1, len(clst_stats)))]
    while len(agg_stats) != (n_to - n_from + 1):
        agg_stats = np.append(agg_stats, [0])

    xi = list(range(n_from, n_to + 1, 1))
    plt.bar(xi, agg_stats)
    plt.xlabel('Cluster size')
    plt.ylabel('Number of nodes in the clusters')
    title = 'Number of nodes in clusters of sizes from ' + str(n_from) + " to " + str(
        n_to) + " in " + clst_type + " for " + data_type
    plt.title(title)
    plt.savefig('plot/w_' + clst_type + "_" + str(eps) + "_" + data_type + ".png")
    plt.clf()
    #plt.show()

def plot_graph(x, y, eps, num_clusters, labels, clst_type, data_type, figsize = 50, s2s = 300):
    cmap = generate_colorscheme(num_clusters * 2)
    c = [cmap[labels[i]+1] for i in range(len(labels))]
    plt.figure(figsize=(figsize,figsize))
    plt.title(clst_type + " results for "+data_type+" data\n num of clusters = " + str(num_clusters))
    plt.scatter(x, y, s = figsize/s2s * ((figsize/50)**2), c = c)
    ax = plt.gca()
    ax.set_facecolor((0, 0, 0))
    plt.savefig('fig/'+clst_type+"_"+str(eps)+"_"+data_type+"_"+str(figsize)+".png")
    plt.clf()
    #plt.show()


def plot_chart(folder, filenames, epsilons, clst_stats):
    # create excel
    workbook = xlsxwriter.Workbook(folder + '.xlsx')
    bold = workbook.add_format({'bold': True})

    for i in range(len(epsilons)):

        epsilon = epsilons[i]

        worksheet = workbook.add_worksheet("Epsilon = %.3f" % epsilon)
        worksheet.set_column('A:A', 12)

        stats = clst_stats[:, i]
        height = max([len(clst) for clst in stats])

        # populate col & row name
        worksheet.write(0, 0, "clst_sz\\data", bold)
        for i in range(len(filenames)):
            worksheet.write(0, i + 1, filenames[i], bold)
        for i in range(1, height):
            worksheet.write(i, 0, i)

        # populate cells
        for i in range(len(filenames)):
            for j in range(1, len(stats[i])):
                worksheet.write(j, i + 1, stats[i][j])
            for j in range(len(stats[i]), height):
                worksheet.write(j, i + 1, 0)

    workbook.close()


