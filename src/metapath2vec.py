from stellargraph import StellarGraph
import networkx as nx
import os
from util.constants import graph_infos, metapaths, dir_path
import stellargraph.data.loader as loader


class Metapath2vec:

    def __init__(self):

        self.graph_infos = graph_infos

        self.metapaths = metapaths
        self.g_nx = None
        self.rw = None

    def run(self):
        self.g_nx, self.rw = self.load4graph()
        self.gen4embedding(self.metapaths, self.g_nx, self.rw)

    def load4graph(self):
        """
        将所有边和顶点信息装入图，并生成游走路径rw
        :return:
        """
        g_nx = self.load_dataset_SMDB(dir_path, self.graph_infos)
        print("Number of nodes {} and number  of edges {} in graph.".format(g_nx.number_of_nodes(),
                                                                            g_nx.number_of_edges()))
        from stellargraph.data import UniformRandomMetaPathWalk
        rw = UniformRandomMetaPathWalk(StellarGraph(g_nx))
        return g_nx, rw

    def gen4embedding(self, metapaths, g_nx, rw):
        """
        根据元路径生成embedding数据，并保存到指定路径
        :param metapaths: 元路径数组
        :param g_nx: 装载了所有边和顶点的图
        :param rw: 随机游走模型
        :return:
        """
        for metapath in metapaths:
            walks = rw.run(nodes=list(g_nx.nodes()),  # root nodes
                           length=100,  # maximum length of a random walk
                           n=5,  # number of random walks per root node
                           metapaths=metapath  # the metapaths
                           )
            print("Number of random walks: {}".format(len(walks)))

            from gensim.models import Word2Vec
            model = Word2Vec(walks, size=128, window=5, min_count=0, sg=1, workers=4, iter=5)

            filename = ''
            for tp in metapath[0]:
                filename += str(tp)
            filepath = dir_path + 'embedding/' + filename + '_mp2vec.txt'
            print(filepath)
            self.save4embedding(filepath, model.wv, metapath[0][0])

    def save4embedding(self, targetfile, word_vec, e_name):
        """
        生成embedding数据文件
        :param e_name:
        :param targetfile: 保存文件路径
        :param word_vec: embedding数据
        :return:
        """
        with open(targetfile, 'w+') as outfile:
            outfile.writelines(str(len(word_vec.vectors)) + ' 128\n')
            for entity in word_vec.index2entity:
                if str(entity)[:1] != e_name:
                    continue
                string = str(word_vec.get_vector(entity))
                string = string.replace('[', '').replace(']', '').replace('\n', '').replace('  ', ' ')
                if string[:1] is '-':
                    string = ' ' + string
                string = string.replace('  ', ' ').replace('  ', ' ')
                entity = str(entity).replace(e_name, '').replace('\n', '')
                outfile.writelines(entity + string + '\n')

    def load_dataset_SMDB(self, location, infos, weighted=False):
        location = os.path.expanduser(location)
        if not os.path.isdir(location):
            print("The location {} is not a directory.".format(location))
            exit(0)
        g_nx = nx.Graph()  # create the graph
        print('init the graph and load data...')
        for info in infos:
            # add nodes with labels.
            if info[1] is 'node':
                # load the raw data
                ids = []
                with open(os.path.join(location, info[0]), 'r') as file:
                    for line in file.readlines():
                        content, nid = line.strip().split(',')
                        ids.append(info[2] + nid)
                g_nx.add_nodes_from(ids, label=info[2])

            # add the edges with labels
            if info[1] is 'edge':
                # load the raw data
                edges = []
                with open(os.path.join(location, info[0]), 'r') as file:
                    for line in file.readlines():
                        from_node, to_node = line.strip().split(',')
                        if weighted:
                            edges.append((info[3] + str(from_node), info[4] + str(to_node), 1))
                        else:
                            edges.append((info[3] + str(from_node), info[4] + str(to_node), 1))
                g_nx.add_weighted_edges_from(edges, label=info[2])

        print('All data has been loaded.')
        return g_nx


if __name__ == '__main__':
    Metapath2vec().run()
