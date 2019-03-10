#!/usr/bin/python
# coding:utf-8
import numpy as np
import scipy.sparse as ss
from util.constants import base_metapath_path, base_embedding_path, dir_path
import os
import subprocess


class Deepwalk:

    def __init__(self, enum, dnum, dtnum, tnum):

        self.enum = enum + 1
        self.dnum = dnum + 1
        self.dtnum = dtnum + 1
        self.tnum = tnum + 1

    def run(self):

        # 生成元路径，相关元路径的邻接矩阵相乘
        self.generate_metapath(dir_path + 'ed.txt', base_metapath_path + 'ede_dpwk.txt', self.enum, self.dnum)
        self.generate_metapath(dir_path + 'edt.txt', base_metapath_path + 'edte_dpwk.txt', self.enum, self.dtnum)
        self.generate_metapath(dir_path + 'et.txt', base_metapath_path + 'ete_dpwk.txt', self.enum, self.tnum)

        self.gen_embedding()

    def generate_metapath(self, filename, targetfile, row_num, col_num):
        print('EDE adjacency matrix multiplication ...')
        ed = self.matrix_init(filename, row_num, col_num)
        print(ed.shape)
        ee = ed.dot(ed.T)
        print('writing to file...')
        self.save(targetfile, ee.toarray())

    def matrix_init(self, file, row_num, colomn_num):
        matrix = np.zeros((row_num, colomn_num))
        with open(file, 'r') as infile:
            for line in infile.readlines():
                m, d = line.strip().split(',')
                matrix[int(m)][int(d)] = 1
        sparse_matrix = ss.csc_matrix(matrix)
        return sparse_matrix

    def save(self, targetfile, matrix):
        total = 0
        with open(targetfile, 'w') as outfile:
            rows, cols, data = ss.find(matrix)
            for i in range(len(rows)):
                outfile.write(str(rows[i]) + '\t' + str(cols[i]) + '\t' + str(data[i]) + '\n')
                total += 1
        print('total = ', total)

    def gen_embedding(self):
        dim = 128
        walk_len = 10
        win_size = 3
        num_walk = 5

        metapaths = ['ede', 'edte', 'ete']

        for metapath in metapaths:
            metapath = metapath + '_dpwk.txt'
            input_file = base_metapath_path + metapath
            output_file = base_embedding_path + metapath

            cmd = 'deepwalk --format edgelist --input ' + input_file + \
                  ' --output ' + output_file + \
                  ' --walk-length ' + str(walk_len) + ' --window-size ' + str(win_size) + ' --number-walks ' \
                  + str(num_walk) + ' --representation-size ' + str(dim)

            # subprocess.call(cmd)
            print(cmd)


if __name__ == '__main__':
    Deepwalk(enum=21011, dnum=1347, dtnum=25, tnum=40).run()
