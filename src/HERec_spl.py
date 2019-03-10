#!/usr/bin/python
# encoding=utf-8

import numpy as np
import random
from copy import deepcopy

from src.data_process import get_metapaths
from src.metapath2vec import Metapath2vec
from src.deepwalk import Deepwalk
import os
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
        self.X = None
        self.res = []
        self.user_metapathdims = None
        self.item_metapathdims = None
        self.fusion_matrix = None
        self.user_metapaths, self.user_metapathnum = get_metapaths(embedding)
        self.recommender = None
        # self.recommender = range(self.unum)
        self.res_num = 0
        self.data_dic = {}
        self.data_copy = {}
        self.f1, self.precision, self.recall = 0, 0, 0
        self.nf1, self.nprecision, self.nrecall = 0, 0, 0
        self.d, self.dt, self.t = 0, 0, 0

    def run(self):
        #
        # if self.embedding is 'dpwk':
        #     Deepwalk(enum=self.unum, pnum=self.inum, dnum=47, dtnum=511, train_rate=self.train_rate).run()
        #
        # if self.embedding is 'mp2vec':
        #     Metapath2vec(self.train_rate).run()
        self.X, self.user_metapathdims = self.load_embedding(self.user_metapaths, self.unum)
        self.initialize()

        self.fusion_matrix = self.get_fusion_embedding()
        # 生成data_dic
        self.gen_data_dic()
        self.data_copy = deepcopy(self.data_dic)

        # 计算推荐结果
        iter_num = 50
        for j in range(10, 60, 10):
            self.res_num, self.res, self.f1, self.precision, self.recall = j, [], 0, 0, 0
            self.d, self.dt, self.t = 0, 0, 0
            self.nf1, self.nprecision, self.nrecall = 0, 0, 0
            for i in range(iter_num):
                # start = time.time()
                self.res = self.multi_process_cal_smilarity(self.cal_similarity)

                # result = self.multiprocess_cal_f1()
                # self.cal_res(result)

                f1, precision, recall, nf1, nprecision, nrecall = self.cal_f1()
                self.f1 += f1
                self.precision += precision
                self.recall += recall
                self.nf1 += nf1
                self.nprecision += nprecision
                self.nrecall += nrecall

                # end = time.time()
                # print('iter:', i, 'f1 = ', f1, 'precision = ', precision, 'recall = ', recall, 'cost:', end - start)
                # print('iter:', i, 'nf1 = ', nf1, 'nprecision = ', nprecision, 'nrecall = ', nrecall, 'cost:', end - start)
            print('result: f1 = ', self.f1/iter_num, 'precision = ', self.precision/iter_num, 'recall = ', self.recall/iter_num, 'ren-num:', self.res_num)
            print('result: nf1 = ', self.nf1/iter_num, 'nprecision = ', self.nprecision/iter_num, 'nrecall = ', self.nrecall/iter_num, 'ren-num:', self.res_num)
            # print('result: d = ', self.d / iter_num, 'dt = ', self.dt / iter_num, 't = ', self.t / iter_num, 'ren-num:',
            #       self.res_num)

    def cal_res(self, result_list):
        for res in result_list:
            self.f1 += res[0]
            self.precision += res[1]
            self.recall += res[2]

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
        return ui

    def linear_fusion(self, i):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            s3 = self.Wu[k].dot(self.X[i][k]) + self.bu[k]
            ui += self.pu[i][k] * s3
        return ui

    # def get_fusion_embedding(self):
    #     print('embedding fusion...')
    #     fusion_matrix = np.zeros((self.unum, self.userdim))
    #     for i in range(self.unum):
    #         fusion_matrix[i] = self.nonlinear_fusion(i)
    #     return fusion_matrix

    def get_fusion_embedding(self):
        print('embedding fusion...')
        fusion_matrix = np.zeros((2, self.unum, self.userdim))
        for i in range(self.unum):
            fusion_matrix[0][i] = self.linear_fusion(i)
            fusion_matrix[1][i] = self.nonlinear_fusion(i)
        return fusion_matrix

    # def cal_similarity(self, idx, sim_res):
    #     sims = []
    #     base_vec = self.fusion_matrix[idx]
    #     for j in range(self.unum):
    #         vec = self.fusion_matrix[j]
    #         # 欧式距离
    #         # euclidean_distance = np.linalg.norm(base_vec-vec)
    #         # 余弦相似
    #         cos_sim = np.dot(base_vec, vec) / (np.linalg.norm(base_vec) * np.linalg.norm(vec))
    #         sims.append([cos_sim, str(j + 1)])
    #     sims.sort(reverse=True)
    #     sims = sims[:self.res_num + 1]
    #     tmp = []
    #     for sim in sims:
    #         tmp.append(sim[1])
    #     tmp.insert(0, str(idx + 1))
    #     sim_res.append(tmp)

    def cal_similarity(self, idx, sim_res):
        linear_sims = []
        non_linear_sims = []
        linear_base_vec = self.fusion_matrix[0][idx]
        non_linear_base_vec = self.fusion_matrix[1][idx]
        for j in range(self.unum):
            linear_vec = self.fusion_matrix[0][j]
            non_linear_vec = self.fusion_matrix[1][j]
            # 欧式距离
            # euclidean_distance = np.linalg.norm(base_vec-vec)
            # 余弦相似
            linear_cos_sim = np.dot(linear_base_vec, linear_vec) / (
                        np.linalg.norm(linear_base_vec) * np.linalg.norm(linear_vec))
            non_linear_cos_sim = np.dot(non_linear_base_vec, non_linear_vec) / (
                        np.linalg.norm(non_linear_base_vec) * np.linalg.norm(non_linear_vec))
            linear_sims.append([linear_cos_sim, str(j + 1)])
            non_linear_sims.append([non_linear_cos_sim, str(j + 1)])
        linear_sims.sort(reverse=True)
        linear_sims = linear_sims[:self.res_num + 1]
        non_linear_sims.sort(reverse=True)
        non_linear_sims = non_linear_sims[:self.res_num + 1]
        linear_tmp = []
        for sim in linear_sims:
            linear_tmp.append(sim[1])
        linear_tmp.insert(0, str(idx + 1))
        non_linear_tmp = []
        for sim in non_linear_sims:
            non_linear_tmp.append(sim[1])
        non_linear_tmp.insert(0, str(idx + 1))
        sim_res.append(linear_tmp)
        sim_res.append(non_linear_tmp)

    def multi_process_cal_smilarity(self, func):
        self.recommender = random.sample(range(1, self.unum), self.res_num)
        mg = multiprocessing.Manager()
        ls = mg.list([])
        pool = multiprocessing.Pool(4)
        for i in self.recommender:
            pool.apply_async(func, args=(i, ls))
        pool.close()
        pool.join()
        return ls

    def multiprocess_cal_f1(self):
        mg = multiprocessing.Manager()
        ls = mg.list([])
        pool = multiprocessing.Pool(4)
        for i in self.res:
            pool.apply_async(self.cal_f1, args=(i, ls))
        pool.close()
        pool.join()
        return ls

    def gen_data_dic(self):
        start = time.time()
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
                if e_dic[e] in self.data_dic:
                    tmp = self.data_dic[e_dic[e]]
                    tmp[0] = list(set(tmp[0]).union(set(pdids)))
                    if tmp[1].count(dt_dic[dt]) == 0:
                        tmp[1].append(dt_dic[dt])
                        self.data_dic[e_dic[e]] = tmp
                else:
                    self.data_dic[e_dic[e]] = [pdids, [dt_dic[dt]], aboard, t_dic[t]]
        end = time.time()
        print('func gen_data_dic cost %d secs' % (end - start))

    def cal_n2(self):
        start = time.time()
        # 从data_dic取出一位人员的数据，遍历data_dic中所有人员，查询所有与其相关的人数，并减1
        for key in self.data_dic:
            base_rec = self.data_dic[key]
            n = 0
            for k in self.data_copy:
                data = self.data_copy[k]
                try:
                    if len(set(data[0]).intersection(set(base_rec[0]))) >= 1 and len(
                            list(set(data[1]).intersection(set(base_rec[1])))) >= 1 and data[3] == base_rec[3]:
                        n += 1
                except TypeError as e:
                    print(k)
            if n == 1:
                print('key:', key, 'n2 --> 0')
            base_rec.append(n - 1)
        end = time.time()
        print('func cal_n2 cost %d secs' % (end - start))

    # def cal_f1(self, input_list, res_list):
    #     # 抽样人数
    #     # 准确率 list
    #     precision = 0
    #     recall = 0
    #     for d in input_list:
    #         # eid 为被推荐人
    #         eid = d[0]
    #         # base_rec 被推荐人信息 [dids, dtids, aboard, title, n2]
    #         base_rec = self.data_dic[eid]
    #         res_set = set()
    #         for key in self.data_dic:
    #             data = self.data_dic[key]
    #             if len(set(data[0]).intersection(set(base_rec[0]))) >= 1 \
    #                     and len(list(set(data[1]).intersection(set(base_rec[1])))) >= 1 \
    #                     and data[3] == base_rec[3]:
    #                 res_set.add(key)
    #         # 如果跟被推荐人相关的人员数为0 计算召回率会报错
    #         if len(res_set) == 0:
    #             continue
    #         n = len(set(d[2:]).intersection(res_set))
    #         if n / self.res_num > 1 or n / len(res_set) > 1:
    #             print('数据异常，召回率:', n / len(res_set), '准确率:', n / self.res_num, '人员id:', eid)
    #         precision += n / self.res_num
    #         recall += n / len(res_set)
    #
    #     precision = precision / self.res_num
    #     recall = recall / self.res_num
    #     f1 = 2 * precision * recall / (precision + recall)
    #     res_list.append([f1, precision, recall])

    # def cal_f1(self):
    #     # 抽样人数
    #     # 准确率 list
    #     precision = 0
    #     recall = 0
    #     for d in self.res:
    #         # eid 为被推荐人
    #         eid = d[0]
    #         # base_rec 被推荐人信息 [dids, dtids, aboard, title, n2]
    #         base_rec = self.data_dic[eid]
    #         res_set = set()
    #         for key in self.data_dic:
    #             data = self.data_dic[key]
    #             if len(set(data[0]).intersection(set(base_rec[0]))) >= 1 \
    #                     and len(list(set(data[1]).intersection(set(base_rec[1])))) >= 1 \
    #                     and data[3] == base_rec[3]:
    #                 res_set.add(key)
    #         # 如果跟被推荐人相关的人员数为0 计算召回率会报错
    #         if len(res_set) == 0:
    #             continue
    #         n = len(set(d[2:]).intersection(res_set))
    #         if n / self.res_num > 1 or n / len(res_set) > 1:
    #             print('数据异常，召回率:', n / len(res_set), '准确率:', n / self.res_num, '人员id:', eid)
    #         precision += n / self.res_num
    #         recall += n / len(res_set)
    #
    #     precision = precision / self.res_num
    #     recall = recall / self.res_num
    #     f1 = 2 * precision * recall / (precision + recall)
    #     return f1, precision, recall

    def cal_f1(self):
        # 抽样人数
        # 准确率 list
        linear_precision = 0
        linear_recall = 0
        non_linear_precision = 0
        non_linear_recall = 0
        for d in range(len(self.res)):
            # eid 为被推荐人
            item = self.res[d]
            eid = item[0]
            # base_rec 被推荐人信息 [dids, dtids, aboard, title, n2]
            base_rec = self.data_dic[eid]
            res_set = set()
            for key in self.data_dic:
                data = self.data_dic[key]
                if len(set(data[0]).intersection(set(base_rec[0]))) >= 1 \
                        and len(list(set(data[1]).intersection(set(base_rec[1])))) >= 1 \
                        and data[3] == base_rec[3]:
                    res_set.add(key)
            # 如果跟被推荐人相关的人员数为0 计算召回率会报错
            if len(res_set) == 0:
                continue
            n = len(set(item[2:]).intersection(res_set))
            if n / self.res_num > 1 or n / len(res_set) > 1:
                print('数据异常，召回率:', n / len(res_set), '准确率:', n / self.res_num, '人员id:', eid)
            if d % 2 == 0:
                linear_precision += n / self.res_num
                linear_recall += n / len(res_set)
            else:
                non_linear_precision += n / self.res_num
                non_linear_recall += n / len(res_set)

        linear_precision = linear_precision / self.res_num
        linear_recall = linear_recall / self.res_num
        non_linear_precision = non_linear_precision / self.res_num
        non_linear_recall = non_linear_recall / self.res_num
        return 2 * linear_precision * linear_recall / (linear_precision + linear_recall), linear_precision, \
               linear_recall, 2 * non_linear_precision * non_linear_recall / (non_linear_precision + non_linear_recall), \
               non_linear_precision, non_linear_recall

    def cal_e_num(self):
        discipline, discipline_type, title = 0, 0, 0
        for d in self.res:
            eid = d[0]
            base_rec = self.data_dic[eid]

            d = d[2:]
            for e in d:
                data = self.data_dic[e]
                if len(set(data[0]).intersection(set(base_rec[0]))) >= 1:
                    discipline += 1
                if len(list(set(data[1]).intersection(set(base_rec[1])))) >= 1:
                    discipline_type += 1
                if data[3] == base_rec[3]:
                    title += 1
        print(discipline / self.res_num, discipline_type / self.res_num, title / self.res_num)
        return discipline / self.res_num, discipline_type / self.res_num, title / self.res_num


if __name__ == "__main__":
    deepwalk = 'dpwk'
    mp2vec = 'mp2vec'

    hnrec = HNERec(unum=21011, userdim=30, embedding=deepwalk)
    hnrec.run()
