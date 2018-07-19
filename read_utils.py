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
						pre = float(cout)/float(sumn)
						pre_sumn = float(cout_sumn)/float(sumn)
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
			if key not in sparse:
				sparse[key] = int(row[11])
			elif key in sparse:
				sparse[key] += int(row[11])
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

	return sparse_m

pre_process(loadpath+load_filename)
with open("save_filename","r") as f:
	csv_file = csv.reader(f)
	for row in csv_file:
		print(row)

def build_features():
	pass

def build_one_hot_labels():
	pass