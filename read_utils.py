#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import pickle as pkl
import csv
import scipy.sparse as sp
import sys

loadpath = "../"
load_filename = "march.week3.csv.uniqblacklistremoved"

savepath = ""
save_filename = "clear_data.csv"

sparse_save_filename = "sparse_martix"
scale_sparse_save_filename = "scale_sparse"

ipset_save_filename = "ip_set"
ipdic_save_filename = "ip_dic"

features_save_filename = "features_martix"
n_features_save_filename = "n_features_martix"

labels_save_filename = "labers"
labels_for_test_save_filename = "labels_for_test"


def pre_process(filname):
	ip_set = set([])

	cout = 0
	
	with open(filname,'r') as f:
		load_csv_file = csv.reader(f)
		for row in load_csv_file:
			'''
			if cout > 100000000:
				break
			'''
			try:
				ip_set.add(row[2])
				cout += 1
			except:
				print("warning: nothing add once")
			finally:
				if cout%100000 == 0:
					sys.stdout.write("already add:%d"%cout)
					sys.stdout.write("\r")
					sys.stdout.flush()
	print("!---------------have:%d------------------------------"%cout)
	sumn = cout
	cout = 0
	cout_sumn = 0
	with open(filname,'r') as f:
		load_csv_file = csv.reader(f)
		save_csv_file = open(save_filename,"w+")
		try:
			writer = csv.writer(save_csv_file)
			for row in load_csv_file:
				if cout > 6000000:
					break
				try:
					if row[3] in ip_set:
						writer.writerow(row)
						cout += 1
				except:
					print("warning:nothing write once")
				finally:
					cout_sumn += 1
					if cout_sumn%100000 == 0:
						pre = float(cout)*100/float(sumn)
						pre_sumn = float(cout_sumn)*100/float(sumn)
						sys.stdout.write("%.4f%%______________%.4f%%"%(pre, pre_sumn))
						sys.stdout.write("\r")
						sys.stdout.flush()
		finally:
			save_csv_file.close()
	print("?--------------------have:%f%%------------------------------"%pre)

#pre_process(loadpath+load_filename)

def build_graph(filname):
	ip_list = []
	ip_dic = {}

	sparse = {}
	#之后再化成np数组

	#for std
	ip_num = 0
	edge_num = 0
	#

	i = 0
	with open(filname,'r') as f:
		csv_file = csv.reader(f)
		for row in csv_file:
			if row[2] not in ip_dic:
				ip_list.append(row[2])
				ip_dic[row[2]] = i
				i += 1
			if row[3] not in ip_dic:
				ip_list.append(row[3])
				ip_dic[row[3]] = i
				i += 1
			key = (int(ip_dic[row[2]]),int(ip_dic[row[3]]))
			ip_num = i
			if key not in sparse:
				sparse[key] = int(row[11])
				edge_num += 1
			elif key in sparse:
				sparse[key] += int(row[11])
			if ip_num%1000 == 0:
				sys.stdout.write("ip_num:%d   edge_num:%d"%(ip_num,edge_num))
				sys.stdout.write("\r")
				sys.stdout.flush()
		sys.stdout.write("ip_num:%d   edge_num:%d"%(ip_num,edge_num))
		sys.stdout.flush()
	
	with open(ipset_save_filename,'w+') as f:
		pkl.dump(ip_list,f)
	with open(ipdic_save_filename,'w+') as f:
		pkl.dump(ip_dic,f)
	print("done")

	sparse_row = []
	sparse_col = []
	data = []
	for key in sparse:
		sparse_row.append(key[0])
		sparse_col.append(key[1])
		data.append(sparse[key])
	sparse_row = np.array(sparse_row)
	sparse_col = np.array(sparse_col)
	data = np.array(data)

	l = len(ip_list)
	sparse_m = sp.csr_matrix((data,(sparse_row,sparse_col)),shape=(l,l))

	with open(sparse_save_filename,'w+') as f:
		pkl.dump(sparse_m,f)
	print("done")
	
	return i

#ip_num = build_graph(save_filename)

def build_scale_graph():
	with open(sparse_save_filename,'r') as f:
		sparse_m = pkl.load(f)
	m = sparse_m.max()
	sparse_m = sparse_m/m
	with open(scale_sparse_save_filename, 'w+') as f:
		pkl.dump(sparse_m,f)
	print("done")

build_scale_graph()


def build_features(ip_num, features_num):
	features_martix = np.zeros((ip_num,features_num))
	ip_dic = {}
	port_dic = {}
	features_dic = {'IPv6':10, 'RSVP':11, 'GRE':12, 'ICMP':13, 'TCP':14, 'UDP':15, 'IPIP':16, 'ESP':17}
	
	with open(ipdic_save_filename,'r') as f:
		ip_dic = pkl.load(f)
		print("ip_dic done load")
	
	row_cout = 0
	with open(save_filename,'r') as f:
		csv_file = csv.reader(f)
		print("begin")
		for row in csv_file:
			source_index = ip_dic[row[2]]
			aim_index = ip_dic[row[3]]

			#端口流量占比先不考虑
			features_martix[source_index][0] += float(row[11])
			features_martix[source_index][1] += float(row[10])
			features_martix[source_index][4] += 1  #端口总数
			features_martix[source_index][5] += 1
			features_martix[source_index][6] += 1
			if row[6] in features_dic:
				features_martix[source_index][features_dic[row[6]]] += float(row[11])   #各协议占比 
			#features_martix[source_index][18]
			#features_martix[source_index][24]
			if float(row[1]) < features_martix[source_index][46] or features_martix[source_index][46] == 0:
				features_martix[source_index][46] = float(row[1])
			if float(row[1]) > features_martix[source_index][47]:
				features_martix[source_index][47] = float(row[1])

			features_martix[aim_index][2] += float(row[11])
			features_martix[aim_index][3] += float(row[10])
			features_martix[aim_index][7] += 1
			features_martix[aim_index][8] += 1
			features_martix[aim_index][9] += 1
			if row[6] in features_dic:
				features_martix[aim_index][features_dic[row[6]]] += float(row[11])
			#features_martix[aim_index][36]
			#features_martix[aim_index][41]
			if float(row[1]) < features_martix[aim_index][48] or features_martix[aim_index][48] == 0:
				features_martix[aim_index][48] = float(row[1])
			if float(row[1]) > features_martix[aim_index][49]:
				features_martix[aim_index][49] = float(row[1])
			row_cout += 1
			if row_cout%100 == 0:
				sys.stdout.write("%d rows done"%row_cout)
				sys.stdout.write("\r")
				sys.stdout.flush()
		sys.stdout.write("%d rows done"%row_cout)
		sys.stdout.flush()
	with open(features_save_filename,'w+') as f:
		pkl.dump(features_martix,f)
		print("done")

#build_features(ip_num,50)

def build_one_hot_labels(ip_num):
	#前600万条只有3类，background, blacklist, anomaly-spam(稀少)（干脆去掉做二分类）
	#为了测试图卷积，每一类都只抽取一部分，剩下作验证集或测试集
	one_hot_labels = np.zeros((ip_num,3))
	ip_dic = {}
	class_dir = {'background':0, 'blacklist':1, 'anomaly-spam':2}

	with open(ipdic_save_filename,'r') as f:
		ip_dic = pkl.load(f)
		print("ip_dic done load")

	row_cout = 0
	no_key_cout = 0
	with open(save_filename,'r') as f:
		csv_file = csv.reader(f)
		for row in csv_file:
			source_index = ip_dic[row[2]]
			if row[12] in class_dir:
				#确保one-hot
				if row[12] != 'background':
					one_hot_labels[source_index][0] = 0
					one_hot_labels[source_index][1] = 0
					one_hot_labels[source_index][2] = 0
				if one_hot_labels[source_index][2] == 0 and one_hot_labels[source_index][1] == 0:
					one_hot_labels[source_index][class_dir[row[12]]] = 1
			else:
				no_key_cout += 1
			row_cout += 1
			if row_cout%100 == 0: 
				sys.stdout.write("%d rows done"%row_cout)
				sys.stdout.write("\r")
				sys.stdout.flush()
		sys.stdout.write("%d rows done"%row_cout)
		sys.stdout.flush()
	print("no key num :%d"%no_key_cout)
	with open(labels_save_filename,'w+') as f:
		pkl.dump(one_hot_labels,f)
		print("done")

#build_one_hot_labels(560441)

def build_one_hot_labels_for_test(ip_num):
	one_hot_labels_for_test = np.zeros((ip_num,3))
	background_test_num = 500
	background_test_cout = 0

	blacklist_test_num = 500
	blacklist_test_cout = 0

	spam_test_num = 5
	spam_test_cout = 0

	row_cout = 0
 	with open(labels_save_filename,'r') as f:
 		labels = pkl.load(f)
 		for i in xrange(len(labels)):
 			if labels[i][0] == 1 and background_test_cout < background_test_num:
 				one_hot_labels_for_test[i][0] = 1
 				background_test_cout += 1
 			if labels[i][1] == 1 and blacklist_test_cout < blacklist_test_num:
 				one_hot_labels_for_test[i][1] = 1
 				blacklist_test_cout += 1
 			if labels[i][2] == 1 and spam_test_cout < spam_test_num:
 				one_hot_labels_for_test[i][2] = 1
 				spam_test_cout += 1
 			row_cout += 1
 			if row_cout%100 == 0:
 				sys.stdout.write("%d rows done"%row_cout)
 				sys.stdout.write("\r")
 				sys.stdout.flush()
 		sys.stdout.write("%d rows done"%row_cout)
 		sys.stdout.flush()
 	with open(labels_for_test_save_filename,'w+') as f:
 		pkl.dump(one_hot_labels_for_test,f)
 		print("done")

#build_one_hot_labels_for_test(560441)

def normalize_data(ip_num, features_num):
	n_features_martix = features_martix = np.zeros((ip_num,features_num))
	with open(features_save_filename,'r') as f:
		n_features_martix = pkl.load(f)
	mean_v = np.mean(n_features_martix, 0)
	std_v = np.std(n_features_martix, 1)
	for i in range(n_features_martix.shape[1]):
		n_features_martix[:,i] = (n_features_martix[:,i]-mean_v[i])/std_v[i]
		sys.stdout.write("%d comp"%i)
		sys.stdout.write("\r")
		sys.stdout.flush()
	with open(n_features_save_filename, "w+") as f:
		pkl.dump(n_features_martix,f)
		print("done")

#normalize_data(560441,50)










