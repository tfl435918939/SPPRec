#!/usr/bin/python
# encoding=utf-8

dir_path = '../data/'
base_metapath_path = dir_path + 'metapath/'
base_embedding_path = dir_path + 'embedding/'

# metapath2vec
graph_infos = [['eid.txt', 'node', 'e'],
               ['tid.txt', 'node', 't'],
               ['did.txt', 'node', 'd'],
               ['dtid.txt', 'node', 'dt'],
               ['ed.txt', 'edge', 'e-d', 'e', 'd'],
               ['edt.txt', 'edge', 'e-dt', 'e', 'dt'],
               ['et.txt', 'edge', 'e-t', 'e', 't']]
metapaths = [
            [['e', 'd', 'e']],
            [['e', 'dt', 'e']],
            [['e', 't', 'e']]
        ]

if __name__ == '__main__':
    print('dir_path:', dir_path)
    print('base_metapath_path:', base_metapath_path)
    print('base_embedding_path:', base_embedding_path)
