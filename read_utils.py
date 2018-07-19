#-*- coding: utf-8 -*-
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
ipset_save_filename = "ip_set"
ipdic_save_filename = "ip_dic"
features_save_filename = "features_martix"

def pre_process(filname):
	ip_set = set([])

	cout = 0
	
	with open(filname,'r') as f:
		load_csv_file = csv.reader(f)
		for row in load_csv_file:
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
	print("?--------------------have:%f------------------------------"%pre)

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
	
	return sparse_m
#下面已经用过一次了
#pre_process(loadpath+load_filename)

#sparse_m = build_graph(save_filename)


'''
with open(save_filename,"r") as f:
	csv_file = csv.reader(f)
	for row in csv_file:
		print(row)
'''
def build_features(ip_num, features_num):
	features_martix = np.zeros((ip_num,features_num))
	ip_dic = {}
	port_dic = {}
	features_dic = {'IPv6':10, 'RSVP':11, 'GRE':12, 'ICMP':13, 'TCP':14, 'UDP':15, 'IPIP':16, 'ESP':17}
	
	with open(ipdic_save_filename,'r') as f:
		ip_dic = pkl.load(f)
		print("ip_dic done load")
	
	with open(sparse_save_filename,'r') as f:
		sparse_martix = pkl.load(f)
		print("sparse_martix done load")
	
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


build_features(15884000,50)


def build_one_hot_labels():
	pass











