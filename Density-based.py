import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy


figure_id = 1

def read_file(filename):
    """
    read and parse the data file

    Parameters:
    -----------
    filename : string
              file that contains the data

    Returns:
    -----------
    _ground_truth : 2-D array
                   [gene_id][group_number]

    _gene_expression : 2-D array
                   [gene_id][gene_expressions]
    """
    raw_data = pd.read_csv(filename, sep=r'\s+', header=None)
    _ground_truth = raw_data.iloc[:, 0:2].values
    _gene_expression = pd.concat([raw_data.iloc[:, 1], raw_data.iloc[:, 2:]], axis=1).values

    return _ground_truth, _gene_expression

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='iyer.txt', help='file contains data')
    parser.add_argument('-k', '--cluster_number', type=int, default=10, help='number of clusters')
    args = parser.parse_args()
title_part = ': file = ' + args.filename
truth_cluster, gene_expression = read_file(args.filename)

eps=1.2
MinPts=2
#data=np.ndarray.tolist(gene_expression[:,1:])
data=np.ndarray.tolist(gene_expression)


#euclidean distance
def dist(p1,p2):
    return scipy.spatial.distance.euclidean(p1[1:],p2[1:])

def regionQuery(P, eps, all_points):
	neighbourPts = []
	for point in all_points:
		if dist(point,P)<=eps:
			neighbourPts.append(point)
	return neighbourPts

'''
visited=data
neighbourPts=regionQuery(data[1],1.5,data)
print(neighbourPts)
for i in neighbourPts:
    if i not in visited:
        visited.append(i)
'''

def DBSCAN(all_points, eps, MinPts):
	other = []
	core=[]
	#unvisited=set(all_points)
	visited = []
	C = []
	c_n = -1
	for point in all_points:
		neighbourPts = regionQuery(point, eps, all_points)
		if len(neighbourPts) < MinPts:
			other.append(point)
		else:
			core.append(point)
	#print("No. of core points: " , len(core))
	#print("No. of other points:", len(other))

	for point in core:
		if point not in visited:
			visited.append(point)
			C.append([])
			c_n+=1
			neighbourPts = regionQuery(point, eps, all_points)
			expandCluster(point, neighbourPts, C, c_n, eps, MinPts, visited, all_points)
	C.append(other)
	ID=clusterID(C)
	T=truth(C)
	da=dat(C)
	#print(len(set(T)))
	#print("labels:",ID)
	#print("Ground truth:",T)
	print("Number of clusters:",len(C))
	#print("data",len(da))
	print("Jaccard Conefficient is: %f" % jaccard_coefficient(ID, T))
	#print (gene_expression[:,1:])
	pca(da,ID)

def expandCluster(point, neighbourPts, C, c_n, eps, MinPts, visited, all_points):

	C[c_n].append(point)
	while len(neighbourPts):
		p = neighbourPts[0]
		neighbourPts.remove(p)
		if p not in visited:
			visited.append(p)
			neighbourPts_2 = regionQuery(p, eps, all_points)
			if len(neighbourPts_2) >= MinPts:
				neighbourPts += neighbourPts_2
				C[c_n].append(p)

		empty=[]
		for i in C:
			empty += i
		if p not in empty: #if p is not yet member of any cluster
			C[c_n].append(p)
			if p not in visited:
				visited.append(p)


def pca(data, labels):
	visual_model = PCA(n_components=2)
	dxy = visual_model.fit_transform(data)
#	labels = labels + 1
	#print(len(labels))
	#print(dxy.shape)
	#print(dxy)
	plot_result(dxy, labels, "Density based clustering" + title_part)

	'''
	data = np.array(data)
	mean_vector = np.mean(data, axis=0)
	data_adjusted = data - mean_vector
	covariance = np.dot(data_adjusted, np.transpose(data_adjusted)) / np.size(data, axis=0)
	eigenvalues, eigenvectors = np.linalg.eig(covariance)
	top_indices = np.argpartition(eigenvalues, -2)[-2:]
	components = eigenvectors[:, top_indices]
	print (components)
	plot_result(components, labels, "DBSCAN_PCA")
'''
def plot_result(data, labels, title):
	plt.scatter(data[:, 0], data[:, 1], c=labels)
	plt.title(title)
	plt.show()

def __calc_agree_number(clustering, truth):
    _f00 = 0
    _f11 = 0

    for obj in range(len(clustering)):
        for obj2 in range(len(clustering)):
            if clustering[obj] == clustering[obj2] and truth[obj] == truth[obj2]:
                _f11 += 1
                continue
            if clustering[obj] != clustering[obj2] and truth[obj] != truth[obj2]:
                _f00 += 1

    return _f00, _f11

def jaccard_coefficient(clustering, truth):

    f00, f11 = __calc_agree_number(clustering, truth)
    total_objects = len(clustering)
    return f11 * 1.0 / (total_objects * total_objects - f00)

def dat(C):
	da=[]
	for i in range(len(C)):
		for j in range(len(C[i])):
			da.append(np.array(C[i][j][1:]))
	return da

def clusterID(C):
	la=[]
	for i in range(len(C)):
		for j in range(len(C[i])):
			la.append(i)
	return la


def truth(C):
	groundTruth=[]
	for i in range(len(C)):
		for j in range(len(C[i])):
			groundTruth.append(C[i][j][0])
	return groundTruth

DBSCAN(data,eps,MinPts)

