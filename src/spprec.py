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
        self.recommender = random.sample(range(1, self.unum), 1000)  # 1 ~ unum 取 1000 个
        # 加载元路径以及各元路径的embedding信息
        self.user_metapaths, self.user_metapathnum = get_metapaths(embedding)
        self.X, self.user_metapathdims = self.load_embedding(self.user_metapaths, self.unum)
        # unum X 3
        self.pu = [[0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [0.4, 0.3, 0.3], [1/3, 1/3, 1/3], [0.2, 0.4, 0.4]]
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
        # 计算融合embedding数据
        self.fusion_emb = None

    def recommend(self):
        #
        # if self.embedding is 'dpwk':
        #     Deepwalk(enum=self.unum, pnum=self.inum, dnum=47, dtnum=511, train_rate=self.train_rate).run()
        #
        # if self.embedding is 'mp2vec':
        #     Metapath2vec(self.train_rate).run()

        # 计算推荐结果
        for weight in self.pu:
            self.fusion_emb = self.get_fusion_embedding(weight=weight)
            print('weight:', weight)
            for j in range(10, 60, 10):
                self.res_num = j

                res = self.multi_process_cal_smilarity()
                lpr = res[0:-1:2]
                nlpr = res[1:: 2]

                # 计算 线性融合函数和非线性融合函数 推荐指标
                l_prec_recl = self.cal_f1(lpr)
                nl_prec_recl = self.cal_f1(nlpr)
                self.cal_rec_quota(l_prec_recl, self.res_num, 'linear')
                self.cal_rec_quota(nl_prec_recl, self.res_num, 'non-linear')

                # 统计与被推荐人各属性相关的人员数
                self.statistics(lpr, 'linear')
                self.statistics(nlpr, 'non-linear')

    def cal_rec_quota(self, quotas, res_num, method):
        """
        对召回率、准确率, 在不同推荐人个数下的平均, 并进行打印
        :param quotas: 召回率, 准确率列表
        :param res_num: 统计个数(取多少个推荐人)
        :param method: 融合方法
        :return:
        """
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
        """
        从 embedding 文件中加载 num 对应所有元路径类型对应向量
        :param metapaths: 所有元路径的类型字符串, ex: ede, ete 等
        :param num: 专家数 e 的个数
        :return: X: X[i][j][k]: i -> unum, j -> 第几个 metapath, k -> 读取文件每一行向量表示中的第 k 列
                 metapath_dims: 生成的元路径向量维度, 这里是 128 维
        """
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
                        X[i][ctn][j] = float(arr[j + 1])  # 每行 idx = 0 位置是被推荐人的序号, 也就是每行有 129 个数
            ctn += 1
        print('Load embedding finished.')
        return X, metapath_dims

    def sigmod(self, x):
        # Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间
        return 1 / (1 + np.exp(-x))

    def nonlinear_fusion(self, i, weight):
        """
        对 i 对应的用户向量, 根据不同的 weight 权重矩阵进行聚合, 生成非线性聚合后的用户向量
        :param i: i对应用户的序号
        :param weight: 权重
        :return: 用户向量
        """
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
        """
        对 i 对应的用户向量, 根据不同的 weight 权重矩阵进行聚合, 生成线性聚合后的用户向量
        :param i: i对应用户的序号
        :param weight: 权重
        :return: 用户向量
        """
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            s3 = self.Wu[k].dot(self.X[i][k]) + self.bu[k]
            ui += weight[k] * s3
        return ui

    def get_fusion_embedding(self, weight):
        """
        根据权重矩阵对向量进行线性与非线性聚合
        :param weight: 权重矩阵, ex:[0.8, 0.1, 0.1]
        :return: 经过聚合后的向量构成的矩阵
        """
        print('embedding fusion...')
        fusion_matrix = np.zeros((2, self.unum, self.userdim))
        for i in range(self.unum):
            fusion_matrix[0][i] = self.linear_fusion(i, weight)
            fusion_matrix[1][i] = self.nonlinear_fusion(i, weight)
        return fusion_matrix

    def cal_similarity(self, idx, sim_res):
        """
        计算每个向量的 cos 值, 将被推荐人idx , 以及 cos 值从大到小的被推荐人 idx 列表
        :param idx: 被推荐人 idx
        :param sim_res: 用于存放多进程对于每个被推荐人的推荐结果的列表
        :return:
        """
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
        """
        计算推荐人列表的多进程调用
        :return:
        """
        mg = multiprocessing.Manager()
        ls = mg.list([])
        pool = multiprocessing.Pool(4)
        for i in self.recommender:
            pool.apply_async(self.cal_similarity, args=(i, ls))
        pool.close()
        pool.join()
        return ls

    def gen_data_dic(self):
        """
        构建 eid: {pdids, dt对应列表, aboard, t对应列表} 这样一个字典
        :return:
        """
        data_dic = {}
        start = time.time()
        print('generate data dictionary...')

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
        end = time.time()
        print('func gen_data_dic cost %d secs' % (end - start))
        return data_dic

    def cal_f1(self, res):
        """
        对于得到的推荐人列表, 每一个生成一行[准确率, 召回率], 最后得到一个准确率, 召回率的列表
        :param res: 推荐人列表
        :return:
        """
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
                if len(set(data[0]).intersection(set(base_rec[0]))) >= 1 \
                        and len(list(set(data[1]).intersection(set(base_rec[1])))) >= 1 \
                        and data[3] == base_rec[3]:
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
        """
        统计与被推荐人各属性相关的人员数, 打印相关属性数的均值
        :param res: 推荐人列表
        :param method: 融合方式
        :return:
        """
        d, dt, t = 0, 0, 0
        for rec_list in res:
            rec_id = rec_list[0]
            base_rec = self.data_dic[rec_id]
            rec_list = rec_list[2:]
            for eid in rec_list:
                data = self.data_dic[eid]
                if len(set(data[0]).intersection(set(base_rec[0]))) >= 1:
                    d += 1
                if len(list(set(data[1]).intersection(set(base_rec[1])))) >= 1:
                    dt += 1
                if data[3] == base_rec[3]:
                    t += 1
        print('method:', method, 'd-num:', d / len(res), 'dt-num:', dt / len(res), 't-num:', t / len(res), 'res_num:',
              self.res_num)


if __name__ == "__main__":
    deepwalk = 'dpwk'
    mp2vec = 'mp2vec'

    hnrec = HNERec(unum=21011, userdim=30, embedding=deepwalk)
    hnrec.recommend()
