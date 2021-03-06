#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import cPickle as pkl
import csv
import scipy.sparse as sp
import sys
import heapq


#控制所有数不小于0的正规化！试一试

loadpath = "../"
load_filename = "march.week3.csv.uniqblacklistremoved"
load_filename_2 = "july.week5.csv.uniqblacklistremoved"

savepath = ""
save_filename = "clear_data.csv"

sparse_save_filename = "sparse_martix"
scale_sparse_save_filename = "scale_sparse"

ipset_save_filename = "ip_set"
ipdic_save_filename = "ip_dic"

features_save_filename = "features_martix"
features_save_filename_without_port = "features_martix_without_port"
n_features_save_filename = "n_features_martix"
n_features_save_filename_without_port = "n_features_martix_without_port"

labels_save_filename = "labers"
labels_for_test_save_filename = "labels_for_test"

source_port_dic_save_filename = "source_port_dic"
aim_port_dic_save_filename = "aim_port_dic"
source_port_dic_2_save_filename = "source_port_dic_2"
aim_port_dic_2_save_filename = "aim_port_dic_2"

source_port_list_save_filename = "source_port_list"
source_port_list_2_save_filename = "source_port_list_2"
aim_port_list_save_filename = "aim_port_list"
aim_port_list_2_save_filename = "aim_port_list_2"

base_num = 65000000



def pre_process(filname):
	ip_set = set([])

	cout = 0

	source_ip_num = set([])
	aim_ip_num = set([])
	
	with open(filname,'r') as f:
		load_csv_file = csv.reader(f)
		for row in load_csv_file:

			if cout < base_num:
				cout += 1
				if cout % 100000 == 0:
					sys.stdout.write("%d"%cout)
					sys.stdout.write("\r")
					sys.stdout.flush()
				continue
			if cout > base_num+10000000:
				break
			try:
				ip_set.add(row[2])
				cout += 1
			except:
				print("warning: nothing add once")
			finally:
				if (cout-base_num)%100000 == 0:
					sys.stdout.write("already add:%d"%(cout-base_num))
					sys.stdout.write("\r")
					sys.stdout.flush()
	print("!---------------have:%d------------------------------"%(cout-base_num))
	sumn = cout-base_num
	cout = 0
	cout_sumn = 0
	with open(filname,'r') as f:
		load_csv_file = csv.reader(f)
		save_csv_file = open(save_filename,"w+")
		try:
			writer = csv.writer(save_csv_file)
			for row in load_csv_file:
				
				if cout_sumn < base_num:
					cout_sumn += 1
					if cout_sumn%100000 == 0:
						sys.stdout.write("%d"%cout_sumn)
						sys.stdout.write("\r")
						sys.stdout.flush()
					continue
				if cout_sumn > base_num+10000000:
					break
				try:
					if row[3] in ip_set:
						writer.writerow(row)
						cout += 1
						source_ip_num.add(row[2])
						aim_ip_num.add(row[3])
				except:
					print("warning:nothing write once")
				finally:
					cout_sumn += 1
					if (cout_sumn-base_num)%100000 == 0:
						pre = float(cout)*100/float(sumn)
						pre_sumn = float(cout_sumn-base_num)*100/float(sumn)
						sys.stdout.write("%.4f%%______________%.4f%%"%(pre, pre_sumn))
						sys.stdout.write("\r")
						sys.stdout.flush()
		finally:
			save_csv_file.close()
	print("?--------------------have:%f%%------------------------------"%pre)
	print("source ip num:%d"%len(source_ip_num))
	print("aim ip num:%d"%len(aim_ip_num))
	print("-----------------------------------")

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
'''
ip_num = build_graph(save_filename)

with open("ip_num",'w+') as f:
	pkl.dump(ip_num,f)
print("ip_num save")
'''
def build_scale_graph():
	with open(sparse_save_filename,'r') as f:
		sparse_m = pkl.load(f)
	m = sparse_m.max()
	sparse_m = sparse_m/m
	with open(scale_sparse_save_filename, 'w+') as f:
		pkl.dump(sparse_m,f)
	print("scale done")

#build_scale_graph()

def build_port_flow(ip_num):
	source_port_dic = {}
	source_port_dic_2 = {}
	aim_port_dic = {}
	aim_port_dic_2 = {}

	ip_dic = {}
	with open(ipdic_save_filename,'r') as f:
		ip_dic = pkl.load(f)
		print("\nip_dic done load...(build_port_flow)")

	with open(save_filename,'r') as f:
		csv_file = csv.reader(f)
		print("build port flow begin!")
		cout = 0
		for row in csv_file:
			if row[2] in source_port_dic:
				if row[4] in source_port_dic[row[2]]:
					source_port_dic[row[2]][row[4]] += float(row[11])
				else:
					source_port_dic[row[2]][row[4]] = float(row[11])
			else:
				dic = {}
				dic[row[4]] = float(row[11])
				source_port_dic[row[2]] = dic
		
			if row[2] in source_port_dic_2:
				if row[5] in source_port_dic_2[row[2]]:
					source_port_dic_2[row[2]][row[5]] += float(row[11])
				else:
					source_port_dic_2[row[2]][row[5]] = float(row[11])
			else:
				dic = {}
				dic[row[5]] = float(row[11])
				source_port_dic_2[row[2]] = dic
		
			if row[3] in aim_port_dic:
				if row[5] in aim_port_dic[row[3]]:
					aim_port_dic[row[3]][row[5]] += float(row[11])
				else:
					aim_port_dic[row[3]][row[5]] = float(row[11])
			else:
				dic = {}
				dic[row[5]] = float(row[11])
				aim_port_dic[row[3]] = dic

			if row[3] in aim_port_dic_2:
				if row[4] in aim_port_dic_2[row[3]]:
					aim_port_dic_2[row[3]][row[4]] += float(row[11])
				else:
					aim_port_dic_2[row[3]][row[4]] = float(row[11])
			else:
				dic = {}
				dic[row[4]] = float(row[11])
				aim_port_dic_2[row[3]] = dic
			cout += 1
			if cout%10000 == 0:
				sys.stdout.write("%d rows write"%cout)
				sys.stdout.write("\r")
				sys.stdout.flush()
		sys.stdout.write("%d rows write\n"%cout)
		sys.stdout.flush()
	'''
	cout = 0
	add_dic = {-1:0,-2:0,-3:0,-4:0,-5:0}
	for key in source_port_dic:
		source_port_dic[key].update(add_dic)
		cout += 1
		if cout%1000 == 0:
			sys.stdout.write("%d/%d rows write"%(cout,len(source_port_dic)))
			sys.stdout.write("\r")
			sys.stdout.flush()
	print("add 1 is done...")
	cout = 0
	for key in source_port_dic_2:
		source_port_dic_2[key].update(add_dic)
		cout += 1
		if cout % 1000 == 0:
			sys.stdout.write("%d/%d rows write"%(cout,len(source_port_dic_2)))
			sys.stdout.write("\r")
			sys.stdout.flush()
	print("add 2 is done...")
	cout = 0
	for key in aim_port_dic:
		aim_port_dic[key].update(add_dic)
		cout += 1
		if cout % 1000 == 0:
			sys.stdout.write("%d/%d rows write"%(cout,len(aim_port_dic)))
			sys.stdout.write("\r")
			sys.stdout.flush()
	print("add 3 is done...")
	cout = 0
	for key in aim_port_dic_2:
		aim_port_dic_2[key].update(add_dic)
		cout += 1
		if cout % 1000 == 0:
			sys.stdout.write("%d/%d rows write"%(cout,len(aim_port_dic_2)))
			sys.stdout.write("\r")
			sys.stdout.flush()
	print("add 4 is done...")
	'''
	'''
	with open(source_port_dic_save_filename,'w+') as f:
		cPickle.dump(source_port_dic,f)
	print("save source_port_dic...")
	with open(aim_port_dic_save_filename,'w+') as f:
		cPickle.dump(aim_port_dic,f)
	print("save aim_port_dic...")
	with open(source_port_dic_2_save_filename,'w+') as f:
		cPickle.dump(source_port_dic_2,f)
	print("save source_port_dic_2...")
	with open(aim_port_dic_2_save_filename,'w+') as f:
		cPickle.dump(aim_port_dic_2,f)
	print("save aim_port_dic_2...")
	'''
	sort_source_list = np.zeros((ip_num,5))
	sort_aim_list = np.zeros((ip_num,5))
	sort_source_list_2 = np.zeros((ip_num,5))
	sort_aim_list_2 =np.zeros((ip_num,5))
	
	cout = 0
	for key in source_port_dic:
		largest_5 = heapq.nlargest(5,np.pad(source_port_dic[key].values(),(0,5),'constant',constant_values=0))
		sort_source_list[ip_dic[key]] = largest_5
		cout += 1
		sys.stdout.write("%d key write done/%d"%(cout,len(source_port_dic)))
		sys.stdout.write("\r")
		sys.stdout.flush()
	print("1 is done...\n")

	cout = 0
	for key in aim_port_dic:
		largest_5 = heapq.nlargest(5,np.pad(aim_port_dic[key].values(),(0,5),'constant',constant_values=0))
		sort_aim_list[ip_dic[key]] = largest_5
		cout += 1
		sys.stdout.write("%d key write done/%d"%(cout,len(aim_port_dic)))
		sys.stdout.write("\r")
		sys.stdout.flush()
	print("2 is done...\n")

	cout = 0
	for key in source_port_dic_2:
		largest_5 = heapq.nlargest(5,np.pad(source_port_dic_2[key].values(),(0,5),'constant',constant_values=0))
		sort_source_list_2[ip_dic[key]] = largest_5
		cout += 1
		sys.stdout.write("%d key write done/%d"%(cout,len(source_port_dic_2)))
		sys.stdout.write("\r")
		sys.stdout.flush()
	print("3 is done...\n")

	cout = 0
	for key in aim_port_dic_2:
		largest_5 = heapq.nlargest(5,np.pad(aim_port_dic_2[key].values(),(0,5),'constant',constant_values=0))
		sort_aim_list_2[ip_dic[key]] = largest_5
		cout += 1
		sys.stdout.write("%d key write done/%d"%(cout,len(aim_port_dic_2)))
		sys.stdout.write("\r")
		sys.stdout.flush()
	print("4 is done...\n")

	with open(source_port_list_save_filename,'w+') as f:
		pkl.dump(sort_source_list,f)
		print("sort source list save!")
	with open(aim_port_list_save_filename,'w+') as f:
		pkl.dump(sort_aim_list,f)
		print("sort aim list save")
	with open(source_port_list_2_save_filename,'w+') as f:
		pkl.dump(sort_source_list_2,f)
		print("sort source list 2 save!")
	with open(aim_port_list_2_save_filename,'w+') as f:
		pkl.dump(sort_aim_list_2,f)
		print("sort aim list 2 save!")

with open("ip_num",'r') as f:
	ip_num = pkl.load(f)

#build_port_flow(ip_num)
#----------------------------------
def build_features(ip_num, features_num):
	features_martix = np.zeros((ip_num,features_num))
	ip_dic = {}
	features_dic = {'IPv6':10, 'RSVP':11, 'GRE':12, 'ICMP':13, 'TCP':14, 'UDP':15, 'IPIP':16, 'ESP':17}
	
	with open(ipdic_save_filename,'r') as f:
		ip_dic = pkl.load(f)
		print("ip_dic done load...(build_features)")
	'''
	with open(source_port_list_save_filename,'r') as f:
		source_port_list = pkl.load(f)
		print("source port list done load...")
	with open(aim_port_list_save_filename,'r') as f:
		aim_port_list = pkl.load(f)
		print("aim port list done load...")
	with open(source_port_list_2_save_filename,'r') as f:
		source_port_list_2 = pkl.load(f)
		print("source port list 2 done load...")
	with open(aim_port_list_2_save_filename,'r') as f:
		aim_port_list_2 = pkl.load(f)
		print("aim port list 2 done load...")
	
	for i in xrange(len(source_port_list)):
		for j in range(5):   
			features_martix[i][18+j] = source_port_list[i][j]
			features_martix[i][23+j] = source_port_list_2[i][j]
			features_martix[i][36+j] = aim_port_list_2[i][j]
			features_martix[i][41+j] = aim_port_list[i][j]
	print("port flow set done!")
	'''
	row_cout = 0
	with open(save_filename,'r') as f:
		csv_file = csv.reader(f)
		print("begin")
		try:
			for row in csv_file:
				source_index = ip_dic[row[2]]
				aim_index = ip_dic[row[3]]

				features_martix[source_index][0] += float(row[11])
				features_martix[source_index][1] += float(row[10])
				features_martix[source_index][4] += 1  
				features_martix[source_index][5] += 1
				features_martix[source_index][6] += 1
				if row[6] in features_dic:
					features_martix[source_index][features_dic[row[6]]] += float(row[11])
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
					features_martix[aim_index][features_dic[row[6]]+18] += float(row[11])
				#features_martix[aim_index][36]
				#features_martix[aim_index][41]
				if float(row[1]) < features_martix[aim_index][48] or features_martix[aim_index][48] == 0:
					features_martix[aim_index][48] = float(row[1])
				if float(row[1]) > features_martix[aim_index][49]:
					features_martix[aim_index][49] = float(row[1])
				row_cout += 1
				if row_cout%10000 == 0:
					sys.stdout.write("%d rows done"%row_cout)
					sys.stdout.write("\r")
					sys.stdout.flush()
			sys.stdout.write("%d rows done(build features)"%row_cout)
			sys.stdout.flush()
		except:
			sys.stdout.write("%d rows done"%row_cout)
			sys.stdout.flush()

	summ = np.sum(features_martix[:,10:18],axis=1)
	for i in xrange(len(features_martix)):
		features_martix[i,10:18] = features_martix[i,10:18]/(summ[i]+1.0)

	summ = np.sum(features_martix[:,28:36],axis=1)
	for i in xrange(len(features_martix)):
		features_martix[i,28:36] = features_martix[i,28:36]/(summ[i]+1.0)
		#有可能该ip（第i个）没有当过目的ip，使得除数为0
	'''
	with open(features_save_filename,'w+') as f:
		pkl.dump(features_martix,f)
	'''
	with open(features_save_filename_without_port,'w+') as f:
		pkl.dump(features_martix,f)
	print("\nfeatures martix bulid done")

#build_features(ip_num,50)

def build_one_hot_labels(ip_num):
	#前600万条只有3类，background, blacklist, anomaly-spam(稀少)（干脆去掉做二分类）
	#为了测试图卷积，每一类都只抽取一部分，剩下作验证集或测试集
	one_hot_labels = np.zeros((ip_num,3))
	ip_dic = {}
	class_dir = {'background':0, 'blacklist':1, 'anomaly-spam':2}
	print("one hot label build begin")
	with open(ipdic_save_filename,'r') as f:
		ip_dic = pkl.load(f)
		print("ip_dic done load")

	row_cout = 0
	no_key_cout = 0
	abnormal_cout = 0
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
					abnormal_cout += 1.  #异常记录的条数，大于异常ip的数量
				if one_hot_labels[source_index][2] == 0 and one_hot_labels[source_index][1] == 0:
					one_hot_labels[source_index][class_dir[row[12]]] = 1
			else:
				#所有非blacklist放到2
				one_hot_labels[source_index][0] = 0
				one_hot_labels[source_index][1] = 0
				one_hot_labels[source_index][2] = 1
				no_key_cout += 1
			row_cout += 1
			if row_cout%10000 == 0: 
				sys.stdout.write("%d rows done"%row_cout)
				sys.stdout.write("\r")
				sys.stdout.flush()
		sys.stdout.write("%d rows done"%row_cout)
		sys.stdout.flush()
	print("\nno key num :%d"%no_key_cout)
	with open(labels_save_filename,'w+') as f:
		pkl.dump(one_hot_labels,f)
		print("done(one hot labels)")
	print("abnormal cout: %d"%abnormal_cout)
	with open("abnormal_cout",'w+') as f:
		pkl.dump(abnormal_cout,f)
		print("abnormal_cout save...")

#build_one_hot_labels(ip_num)

def build_one_hot_labels_for_test(ip_num):
	one_hot_labels_for_test = np.zeros((ip_num,3))
	background_test_num = 0
	background_test_cout = 0

	blacklist_test_num = 200
	blacklist_test_cout = 0

	#实际是所有非blacklist的异常流量ip
	spam_test_num = 0
	spam_test_cout = 0

	row_cout = 0
	save_path_of_labels = "fin_pre/" 
	file_cout = 0
 	with open(labels_save_filename,'r') as f:
 		labels = pkl.load(f)
 		for i in xrange(len(labels)):
 			'''
 			if labels[i][0] == 1 and background_test_cout < background_test_num:
 				one_hot_labels_for_test[i][0] = 1
 				background_test_cout += 1
 			'''
 			if labels[i][1] == 1 and blacklist_test_cout < blacklist_test_num:
 				one_hot_labels_for_test[i][1] = 1
 				blacklist_test_cout += 1
 			'''
 			elif blacklist_test_cout >= blacklist_test_num:
 				with open(save_path_of_labels+labels_for_test_save_filename+str(file_cout),'w+') as f:
 					pkl.dump(one_hot_labels_for_test,f)
 					print("\nlabels for testn num:%d save..."%file_cout)
 				file_cout += 1
 				one_hot_labels_for_test = np.zeros((ip_num,3))
 				blacklist_test_cout = 0
 			'''
 			'''
 			if labels[i][2] == 1 and spam_test_cout < spam_test_num:
 				one_hot_labels_for_test[i][2] = 1
 				spam_test_cout += 1
 			'''
 			row_cout += 1
 			if row_cout%10000 == 0:
 				sys.stdout.write("%d rows done"%row_cout)
 				sys.stdout.write("\r")
 				sys.stdout.flush()
 		sys.stdout.write("%d rows done"%row_cout)
 		sys.stdout.flush()
 	
 	with open(labels_for_test_save_filename,'w+') as f:
 		pkl.dump(one_hot_labels_for_test,f)
 		print("\ndone(labels for test)")
 	

 	with open("blacklist_cout",'w+') as f:
 		pkl.dump(blacklist_test_cout,f)
 		print("blacklist_cout save...(%d)"%blacklist_test_cout)
 	with open("spam_cout",'w+') as f:
 		pkl.dump(spam_test_cout,f)
 		print("spam_cout save...(%d)"%spam_test_cout)
 	with open("test_num",'w+') as f:
 		pkl.dump((background_test_num,blacklist_test_num),f)
 		print("test num save....(%d, %d)"%(background_test_num,blacklist_test_num))

build_one_hot_labels_for_test(ip_num)

def normalize_data(ip_num, features_num):

	non_zero_add = 0.0001

	n_features_martix = features_martix = np.zeros((ip_num,features_num))
	with open(features_save_filename,'r') as f:
		n_features_martix = pkl.load(f)
	mean_v = np.mean(n_features_martix, 0)
	std_v = np.std(n_features_martix, 0)
	for i in range(n_features_martix.shape[1]):
		if i < 10 or (i > 17 and i < 28) or i > 35: 
			#有无效值或零值在分母？
			n_features_martix[:,i] = (n_features_martix[:,i]-mean_v[i])/(std_v[i]+non_zero_add)
			sys.stdout.write("%d completed"%i)
			sys.stdout.write("\r")
			sys.stdout.flush()
	'''
	with open(n_features_save_filename, 'w+') as f:
		pkl.dump(n_features_martix,f)
	'''
	with open(n_features_save_filename_without_port,'w+') as f:
		pkl.dump(n_features_martix,f)
	print("\ndone(normalize data)")

#normalize_data(ip_num,50)











