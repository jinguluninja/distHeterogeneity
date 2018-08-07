import os
import sys
import random

random.seed(517)

path_train = sys.argv[1]
path_label = sys.argv[2]
path_output = sys.argv[3]
num_inst = int(sys.argv[4])
inst_train_samples = [int(i) for i in sys.argv[5:5+num_inst]]
inst_pos_label_prop = [float(i) for i in sys.argv[5+num_inst:5+2*num_inst]]

train_files = set([file for file in os.listdir(path_train) if file[-3:] == 'npy'])
label_files = {0: [], 1: []}
lines = [line.strip().split(',') for line in open(path_label)]
for line in lines:
	if line[0] in train_files:
		label_files[int(line[1])].append(line[0])
random.shuffle(label_files[0])
random.shuffle(label_files[1])

inst_num_pos_samples = [round(inst_train_samples[i]*inst_pos_label_prop[i]) for i in range(num_inst)]
inst_num_neg_samples = [inst_train_samples[i] - inst_num_pos_samples[i] for i in range(num_inst)]

with open(path_output, 'w') as f:
	for i in range(num_inst):
		for ps in range(sum(inst_num_pos_samples[:i]), sum(inst_num_pos_samples[:i+1])):
			f.write('%s,%s\n' % (label_files[1][ps], i))		
		for ns in range(sum(inst_num_neg_samples[:i]), sum(inst_num_neg_samples[:i+1])):
			f.write('%s,%s\n' % (label_files[0][ns], i))
