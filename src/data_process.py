#!/usr/bin/python
# encoding=utf-8

from util.constants import dir_path


def get_metapaths(embedding):
    u_mps = ['ede', 'edte', 'ete']
    for i in range(len(u_mps)):
        u_mps[i] += '_' + embedding + '.txt'

    return u_mps, len(u_mps)


def gen_id_relationship():
    e_set = set()
    title_set = set()
    discipline_type_set = set()
    discipline_set = set()
    with open(dir_path + 'out.csv', 'r', encoding='gbk') as infile:
        for line in infile.readlines():
            e, pd, pdt, aboard, title, md = line.strip().replace('"', '').replace(' ', '').split(',')
            e_set.add(e)
            title_set.add(title)

            pd = pd.split(';') + md.split(';')
            for discipline in pd:
                if discipline == '':
                    continue
                discipline_set.add(discipline)
            discipline_type_set.add(pdt)

    def object_process(objects, filename):
        i = 1
        object_dict = {}

        for o in objects:
            object_dict[o] = i
            i += 1

        with open(dir_path + filename + '.txt', 'w') as titlefile:
            for o in object_dict:
                titlefile.write(o + ',' + str(object_dict[o]) + '\n')

    object_process(e_set, 'eid')
    object_process(discipline_set, 'did')
    object_process(discipline_type_set, 'dtid')
    object_process(title_set, 'tid')


def gen_meta_info(data_dic):
    ed = []
    edt = []
    ea = []
    et = []

    for key in data_dic:
        data = data_dic[key]
        if len(data[0]) != 0:
            for pdid in data[0]:
                ed.append([key, pdid])
        else:
            print(str(key) + '---> discipline len is 0')
            continue

        if len(data[1]) != 0:
            for pdtid in data[1]:
                edt.append([key, pdtid])
        else:
            print(str(key) + '---> discipline type len is 0')
            continue

        if data[2] == '':
            ea.append([key, 0])
        else:
            ea.append([key, 1])

        et.append([key, data[3]])

    def save(filename, data_list):
        with open(dir_path + filename, 'w') as outfile:
            for data in data_list:
                data = str(data).replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
                outfile.write(str(data) + '\n')

    save('ed.txt', ed)
    save('edt.txt', edt)
    save('ea.txt', ea)
    save('et.txt', et)


def gen_data_dic():
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

    with open(dir_path + 'out.csv', 'r', encoding='gbk') as infile:
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


if __name__ == '__main__':
    gen_id_relationship()
    data_dic = gen_data_dic()
    gen_meta_info(data_dic)
