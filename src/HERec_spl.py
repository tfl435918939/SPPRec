#!/usr/bin/python
# encoding=utf-8

import numpy as np

from src.data_process import get_metapaths
from src.metapath2vec import Metapath2vec
from src.deepwalk import Deepwalk
import multiprocessing
import time


class HNERec:
    def __init__(self, unum, userdim, embedding):
        print(' userdim: ', userdim)
        print(' embedding method: ', embedding)

        self.unum = unum
        self.userdim = userdim
        self.embedding = embedding
        self.X = None
        self.user_metapathdims = None
        self.item_metapathdims = None
        self.fusion_matrix = None
        self.user_metapaths, self.user_metapathnum = get_metapaths(embedding)

    def run(self):
        #
        # if self.embedding is 'dpwk':
        #     Deepwalk(enum=self.unum, pnum=self.inum, dnum=47, dtnum=511, train_rate=self.train_rate).run()
        #
        # if self.embedding is 'mp2vec':
        #     Metapath2vec(self.train_rate).run()
        print('Start load embedding.')
        self.X, self.user_metapathdims = self.load_embedding(self.user_metapaths, self.unum)
        self.initialize()
        print('Load embedding finished.')

        self.fusion_matrix = self.get_fusion_embedding()
        # todo 检查是否有推荐结果文件存在
        pool = self.multi_thread_cal()
        pool.close()
        pool.join()

    def load_embedding(self, metapaths, num):
        X = {}
        for i in range(num):
            X[i] = {}
        metapath_dims = []

        ctn = 0
        for metapath in metapaths:
            sourcefile = '../data/test/embedding/' + metapath
            print('Loading embedding data, location: %s' % sourcefile)
            with open(sourcefile) as infile:

                k = int(infile.readline().strip().split(' ')[1])
                print('Metapath: %s. The dim of metapath embedding: %d' % (metapath, k))
                metapath_dims.append(k)

                # 根据不同的元路径，创建一个二维数组.
                # 数组的第二维度为 Expert/Project 的特征空间的表示 row=Expert/Project col=feature(1,..,k)
                for i in range(num):
                    # 第i个Expert/Project，在当前metapath下的特征空间的表示
                    X[i][ctn] = np.zeros(k)

                for line in infile.readlines():
                    # 获取特征空间向量中每个维度的值
                    arr = line.strip().split(' ')
                    # 将序号转成index
                    i = int(arr[0]) - 1
                    # 将每个维度值附给 X[i][ctn][j]
                    for j in range(k):
                        X[i][ctn][j] = float(arr[j + 1])
            ctn += 1
        return X, metapath_dims

    def initialize(self):
        # unum X 3
        self.pu = np.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum

        self.Wu = {}
        self.bu = {}
        for k in range(self.user_metapathnum):
            # userdim X 128
            self.Wu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            # userdim X 1
            self.bu[k] = np.random.randn(self.userdim) * 0.1

    def sigmod(self, x):
        # Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间
        return 1 / (1 + np.exp(-x))

    def nonlinear_fusion(self, i):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            # 将生成的 userdim X 128（由正态分布的值填充）与某专家的embedding 128 X 1 做点乘，再加上 userdim X 1（由正态分布填充的矩阵）
            # s3 最后的维度是 userdim X 1
            # TODO 为何要使用Sigmoid 函数？
            s3 = self.sigmod(self.Wu[k].dot(self.X[i][k]) + self.bu[k])
            # ui 最终为 s3 乘以每条元路径的权重，通过这样的方式，将所有元路径的embedding融合在一起？
            ui += self.pu[i][k] * s3
        return self.sigmod(ui)

    def linear_fusion(self, i):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            s3 = self.Wu[k].dot(self.X[i][k]) + self.bu[k]
            ui += self.pu[i][k] * s3
        return ui

    def get_fusion_embedding(self):
        print('embedding fusion...')
        fusion_matrix = np.zeros((self.unum, self.userdim))
        for i in range(self.unum):
            fusion_matrix[i] = self.linear_fusion(i)
        return fusion_matrix

    def cal_similarity(self, idx):
        sims = []
        base_vec = self.fusion_matrix[idx]
        for j in range(self.unum):
            vec = self.fusion_matrix[j]
            # 欧式距离
            # euclidean_distance = np.linalg.norm(base_vec-vec)
            # 余弦相似
            cos_sim = np.dot(base_vec, vec) / (np.linalg.norm(base_vec) * np.linalg.norm(vec))
            sims.append([cos_sim, j + 1])
        sims.sort(reverse=True)
        sims = sims[:31]
        rec_res = []
        for sim in sims:
            rec_res.append(sim[1])

        with open('../data/result.txt', 'a+') as simfile:
            simfile.write(str(idx + 1) + '\t' + str(rec_res) + '\n')

        time.sleep(0.1)

    def multi_thread_cal(self):
        p = multiprocessing.Pool()
        p.map_async(self.cal_similarity, iterable=range(self.unum))
        return p


if __name__ == "__main__":
    deepwalk = 'dpwk'
    mp2vec = 'mp2vec'

    hnrec = HNERec(unum=21021, userdim=30, embedding=deepwalk)
    hnrec.run()
