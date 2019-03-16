#!/usr/bin/python
# encoding=utf-8

import numpy as np
import random
from copy import deepcopy

from src.data_process import get_metapaths
from src.metapath2vec import Metapath2vec
from src.deepwalk import Deepwalk
from util.constants import base_embedding_path, dir_path
import multiprocessing
import time


class HNERec:
    def __init__(self, unum, userdim, embedding):
        print(' userdim: ', userdim)
        print(' embedding method: ', embedding)
        self.unum = unum
        self.userdim = userdim
        self.embedding = embedding
        self.res_num = 0

        # 生成推荐人序列
        self.recommender = random.sample(range(0, self.unum), 500)
        # 加载元路径以及各元路径的embedding信息
        self.user_metapaths, self.user_metapathnum = get_metapaths(embedding)
        self.X, self.user_metapathdims = self.load_embedding(self.user_metapaths, self.unum)
        # unum X 3
        # self.pu = [[1 / 3, 1 / 3, 1 / 3]]
        self.pu = [[1.0, 0.0, 0.0]]
        self.Wu = {}
        self.bu = {}
        for k in range(self.user_metapathnum):
            # userdim X 128
            self.Wu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            # userdim X 1
            self.bu[k] = np.random.randn(self.userdim) * 0.1

        # 生成data_dic
        self.data_dic = self.gen_data_dic()
        self.data_copy = deepcopy(self.data_dic)
        self.d_dic = self.gen_dic(dir_path + 'did.txt')
        # 计算融合embedding数据
        self.fusion_emb = None

    def gen_dic(self, filename):
        o_dic = {}
        with open(filename, 'r') as infile:
            for line in infile:
                o, oid = line.strip().split(',')
                o_dic[oid] = o
        return o_dic

    def recommend(self):
        #
        # if self.embedding is 'dpwk':
        #     Deepwalk(enum=self.unum, pnum=self.inum, dnum=47, dtnum=511, train_rate=self.train_rate).run()
        #
        # if self.embedding is 'mp2vec':
        #     Metapath2vec(self.train_rate).run()
        start = time.time()
        rm = self.matrix_init('../data/ede_dpwk.txt', self.unum, self.unum)
        # rm = rm + self.matrix_init('../data/edte_dpwk.txt', self.unum, self.unum)
        # rm = rm + self.matrix_init('../data/ete_dpwk.txt', self.unum, self.unum)
        # rm = rm / 3
        end = time.time()
        print('func M matrix calculation cost %d secs' % (end - start))
        # for j in range(10, 60, 10):
        #     self.res_num = j

        # 计算推荐结果
        for weight in self.pu:
            self.fusion_emb = self.get_fusion_embedding(weight=weight)
            print('weight:', weight)
            for j in range(10, 60, 10):
                self.res_num = j

                res = self.multi_process_cal_smilarity()
                lpr = res[0:-1:2]
                # nlpr = res[1:: 2]

                # 计算 线性融合函数和非线性融合函数 推荐指标
                l_prec_recl = self.cal_f1(lpr)
                # nl_prec_recl = self.cal_f1(nlpr)
                self.cal_rec_quota(l_prec_recl, self.res_num, 'linear')
                # self.cal_rec_quota(nl_prec_recl, self.res_num, 'non-linear')
                res = []
                self.cal_pathsim(res, rm)
                pr = self.cal_f1(res)
                self.cal_rec_quota(pr, self.res_num, 'pathsim')

                # 统计与被推荐人各属性相关的人员数
                self.statistics(lpr, 'linear')
                # self.statistics(nlpr, 'non-linear')
                self.statistics(res, 'pathsim')

    def cal_pathsim(self, sim_res, rm):
        for idx in self.recommender:
            sim = []
            for j in range(self.unum):
                fm = rm[idx][idx] + rm[j][j]
                fz = 2 * rm[idx][j]
                sim.append([fz / fm, str(j + 1)])
            sim.sort(reverse=True)
            sim = [s[1] for s in sim[:self.res_num + 1]]
            sim.insert(0, str(idx + 1))
            sim_res.append(sim)

    def matrix_init(self, file, row_num, colomn_num):
        matrix = np.zeros((row_num, colomn_num))
        with open(file, 'r') as infile:
            for line in infile.readlines():
                m, d, r = line.strip().split('\t')
                matrix[int(m)][int(d)] = r
        return matrix

    def cal_rec_quota(self, quotas, res_num, method):
        prec = 0
        recl = 0
        for i in quotas:
            prec += i[0]
            recl += i[1]
        prec = prec / len(quotas)
        recl = recl / len(quotas)
        print('method:', method, 'precision:', prec, 'recall:', recl, 'f1:',
              2 * prec * recl / (prec + recl), 'res_num:', res_num)

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

    def sigmod(self, x):
        # Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间
        return 1 / (1 + np.exp(-x))

    def nonlinear_fusion(self, i, weight):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            # 将生成的 userdim X 128（由正态分布的值填充）与某专家的embedding 128 X 1 做点乘，再加上 userdim X 1（由正态分布填充的矩阵）
            # s3 最后的维度是 userdim X 1
            # TODO 为何要使用Sigmoid 函数？
            s3 = self.sigmod(self.Wu[k].dot(self.X[i][k]) + self.bu[k])
            # ui 最终为 s3 乘以每条元路径的权重，通过这样的方式，将所有元路径的embedding融合在一起？
            ui += weight[k] * s3
        return ui

    def linear_fusion(self, i, weight):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            # s3 = self.Wu[k].dot(self.X[i][k]) + self.bu[k]
            ui += weight[k] * self.X[i][k]
        return ui

    def get_fusion_embedding(self, weight):
        print('embedding fusion...')
        fusion_matrix = np.zeros((2, self.unum, self.userdim))
        for i in range(self.unum):
            fusion_matrix[0][i] = self.linear_fusion(i, weight)
            fusion_matrix[1][i] = self.nonlinear_fusion(i, weight)
        return fusion_matrix

    def cal_similarity(self, idx, sim_res):
        for femb in self.fusion_emb:
            sim = []
            rec_emb = femb[idx]
            for eid in range(self.unum):
                emb = femb[eid]
                cos = np.dot(rec_emb, emb) / (np.linalg.norm(rec_emb) * np.linalg.norm(emb))
                sim.append([cos, str(eid + 1)])
            sim.sort(reverse=True)
            sim = [s[1] for s in sim[:self.res_num + 1]]
            sim.insert(0, str(idx + 1))
            sim_res.append(sim)

    def multi_process_cal_smilarity(self):
        mg = multiprocessing.Manager()
        ls = mg.list([])
        pool = multiprocessing.Pool(4)
        for i in self.recommender:
            pool.apply_async(self.cal_similarity, args=(i, ls))
        pool.close()
        pool.join()
        return ls

    def gen_data_dic(self):
        data_dic = {}
        print('generate data dictionaty...')

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

        with open(dir_path + 'out.csv', 'r') as infile:
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

    def cal_f1(self, res):
        # 抽样人数
        # 准确率 list
        quota = []
        for rec_list in res:
            # eid 为被推荐人
            rec_id = rec_list[0]
            # base_rec 被推荐人信息 [dids, dtids, aboard, title, n2]
            base_rec = self.data_dic[rec_id]
            res_set = set()
            for eid in self.data_dic:
                data = self.data_dic[eid]
                if len(set(data[0]).intersection(set(base_rec[0]))) >= 1 :
                    res_set.add(eid)
            # 如果跟被推荐人相关的人员数为0 计算召回率会报错
            if len(res_set) == 0:
                continue
            n = len(set(rec_list[2:]).intersection(res_set))
            if n / self.res_num > 1 or n / len(res_set) > 1:
                print('数据异常，召回率:', n / len(res_set), '准确率:', n / self.res_num, '人员id:', rec_id)
            quota.append([n / self.res_num, n / len(res_set)])
        return quota

    def statistics(self, res, method):
        d, dt, t = 0, 0, 0
        for rec_list in res:
            rec_id = rec_list[0]
            base_rec = self.data_dic[rec_id]
            rec_list = rec_list[2:]
            for eid in rec_list:
                data = self.data_dic[eid]
                base_d = base_rec[0]
                compare_d = data[0]

                tmp_d1, tmp_d2 = 0, 0
                for i in base_d:
                    d1 = self.d_dic[i]
                    for j in compare_d:

                        d2 = self.d_dic[j]
                        if d1 == d2:
                            tmp_d1 += 1
                        elif d1[0:3] == d2[0:3]:
                            tmp_d2 += 1

                if tmp_d1 > 0:
                    d += 1
                if tmp_d2 > 0:
                    dt += 1

                # if len(set(data[0]).intersection(set(base_rec[0]))) >= 1:
                #     d += 1
                # if len(list(set(data[1]).intersection(set(base_rec[1])))) >= 1:
                #     dt += 1
                # if data[3] == base_rec[3]:
                #     t += 1
        t = d + dt
        print('method:', method, 'd-num:', d / len(res), 'dt-num:', dt / len(res), 't-num:', t / len(res), 'res_num:',
              self.res_num)


if __name__ == "__main__":
    deepwalk = 'dpwk'
    mp2vec = 'mp2vec'

    hnrec = HNERec(unum=21011, userdim=128, embedding=deepwalk)
    hnrec.recommend()
