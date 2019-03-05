#!/usr/bin/python
# encoding=utf-8

import numpy as np

from src.data_process import get_metapaths
from src.metapath2vec import Metapath2vec
from src.deepwalk import Deepwalk
import os
from util.constants import base_embedding_path, dir_path
import multiprocessing
import time


class HNERec:
    def __init__(self, unum, userdim, embedding, res_num):
        print(' userdim: ', userdim)
        print(' embedding method: ', embedding)

        self.unum = unum
        self.userdim = userdim
        self.embedding = embedding
        self.X = None
        self.res = []
        self.user_metapathdims = None
        self.item_metapathdims = None
        self.fusion_matrix = None
        self.user_metapaths, self.user_metapathnum = get_metapaths(embedding)
        self.res_num = res_num

    def run(self):
        #
        # if self.embedding is 'dpwk':
        #     Deepwalk(enum=self.unum, pnum=self.inum, dnum=47, dtnum=511, train_rate=self.train_rate).run()
        #
        # if self.embedding is 'mp2vec':
        #     Metapath2vec(self.train_rate).run()
        self.X, self.user_metapathdims = self.load_embedding(self.user_metapaths, self.unum)
        self.initialize()

        if os.path.exists(dir_path + 'result.npy'):
            os.remove(dir_path + 'result.npy')
            print('the old version of result file has been deleted!')
        self.fusion_matrix = self.get_fusion_embedding()

        # 计算推荐结果
        pool = self.multi_process_call()
        pool.close()
        pool.join()

        # 生成data_dic
        data_dic = self.gen_data_dic()
        # 计算每位人员的相关人员人数，并更新data_dic
        self.cal_n2(data_dic)
        f1, precision, recall = self.cal_f1(data_dic, self.res_num)
        print('f1 = ', f1, 'precision = ', precision, 'recall = ', recall)

    def load_embedding(self, metapaths, num):
        print('Start load embedding.')
        X = {}
        for i in range(num):
            X[i] = {}
        metapath_dims = []

        ctn = 0
        for metapath in metapaths:
            sourcefile = base_embedding_path + metapath
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
        print('Load embedding finished.')
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
        tmp = []
        for sim in sims:
            tmp.append(sim[1])
        self.res.append(tmp.insert(0, idx + 1))

        # with open(dir_path + 'result.txt', 'a+') as simfile:
        #     simfile.write(str(idx + 1) + '\t' + str(tmp) + '\n')

    def multi_process_call(self):
        p = multiprocessing.Pool()
        p.map_async(self.cal_similarity, iterable=range(self.unum))
        return p

    def gen_data_dic(self):
        # 利用gen_id_relationship生成的对象-对象ID的文件
        # 根据源数据文件，生成每个申报者的相关信息
        def gen_dic(filename):
            o_dic = {}
            with open(filename, 'r') as infile:
                for line in infile:
                    o, oid = line.strip().split(',')
                    o_dic[o] = oid
            return o_dic

        e_dic = gen_dic(dir_path + 'eid.txt')
        d_dic = gen_dic(dir_path + 'did.txt')
        dt_dic = gen_dic(dir_path + 'dtid.txt')
        t_dic = gen_dic(dir_path + 'tid.txt')

        data_dic = {}

        with open(dir_path + '导出.csv', 'r', encoding='UTF-8') as infile:
            for line in infile.readlines():
                e, d, dt, aboard, t, md = line.strip().replace('"', '').replace(' ', '').split(',')

                pdids = []
                d = d.split(';') + md.split(';')
                for d in d:
                    if d == '':
                        continue
                    pdids.append(d_dic[d])

                aboard = (aboard == '') and 0 or 1

                # 非法格式检测
                if len(pdids) == 0 or e_dic[e] is None or dt_dic[dt] is None or t_dic[t] is None:
                    print(e)
                # 处理申报过多个项目的情况
                if e_dic[e] in data_dic:
                    tmp = data_dic[e_dic[e]]
                    tmp[0] = list(set(tmp[0]).union(set(pdids)))
                    if tmp[1].count(dt_dic[dt]) == 0:
                        tmp[1].append(dt_dic[dt])
                    data_dic[e_dic[e]] = tmp
                else:
                    data_dic[e_dic[e]] = [pdids, [dt_dic[dt]], aboard, t_dic[t]]

        return data_dic

    def cal_n2(self, data_dic):
        # 从data_dic取出一位人员的数据，遍历data_dic中所有人员，查询所有与其相关的人数，并减1
        for key in data_dic:
            base_rec = data_dic[key]
            n = 0
            for k in data_dic:
                data = data_dic[k]
                if len(list(set(data[0]).union(base_rec[0]))) > 1 \
                        or len(list(set(data[1]).union(base_rec[1]))) \
                        or data[2] == base_rec[2]:
                    n += 1
            data_dic[key] = base_rec.append(n-1)

    def cal_f1(self, data_dic, res_num):
        # 抽样人数
        # 准确率 list
        precision = 0
        recall = 0
        for d in self.res:
            # eid 为被推荐人
            eid = d[0]
            # base_rec 被推荐人信息 [dids, dtids, aboard, title, n2]
            base_rec = data_dic[eid]
            d = d[2:]
            # n1 为推荐人数
            n1 = len(d)
            precision_tmp = 0
            recall_tmp = 0
            for e in d:
                n = 0
                data = data_dic[e]
                if len(list(set(data[0]).union(base_rec[0]))) > 1 \
                        or len(list(set(data[1]).union(base_rec[1]))) \
                        or data[2] == base_rec[2]:
                    n += 1
                precision_tmp += n / n1
                # base_rec[3] 与被推荐人相关的人人数
                recall_tmp = n / base_rec[3]
            recall += recall_tmp / n1
            precision += precision_tmp / n1
        precision = precision / res_num
        recall = recall / res_num
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall


if __name__ == "__main__":
    deepwalk = 'dpwk'
    mp2vec = 'mp2vec'

    hnrec = HNERec(unum=21021, userdim=30, embedding=deepwalk)
    hnrec.run()
