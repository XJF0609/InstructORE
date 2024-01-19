# *-* coding:utf-8 *-*
colors = ['red','blue','purple','green','orange','red','blue','purple','green','orange','red','blue','purple','green','orange']

import os
from stanfordcorenlp import StanfordCoreNLP
from xml.dom.minidom import parse
import xml.dom.minidom
import math
from copy import deepcopy
from graphviz import Digraph
all_error_sen = set()
# nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
nlp = StanfordCoreNLP(r'/home/code/ReMe/stanford-corenlp-4.5.5')

from shortest_path import 

from graphviz import Digraph
def kmp(mom_string, son_string):
    # 传入一个母串和一个子串
    # 返回子串匹配上的第一个位置，若没有匹配上返回-1
    # test = ''
    # if type(mom_string) != type(test) or type(son_string) != type(test):
    #     return -1
    if len(son_string) == 0:
        return 0
    if len(mom_string) == 0:
        return -1
    # 求next数组
    next = [-1] * len(son_string)
    if len(son_string) > 1:  # 这里加if是怕列表越界
        next[1] = 0
        i, j = 1, 0
        while i < len(son_string) - 1:  # 这里一定要-1，不然会像例子中出现next[8]会越界的
            if j == -1 or son_string[i] == son_string[j]:
                i += 1
                j += 1
                next[i] = j
            else:
                j = next[j]

    # kmp框架
    m = s = 0  # 母指针和子指针初始化为0
    while (s < len(son_string) and m < len(mom_string)):
        # 匹配成功,或者遍历完母串匹配失败退出
        if s == -1 or mom_string[m] == son_string[s]:
            m += 1
            s += 1
        else:
            s = next[s]

    if s == len(son_string):  # 匹配成功
        return m - s
    # 匹配失败
    return -1


class tool:
    @staticmethod
    def kmp(mom_string, son_string):
        # 传入一个母串和一个子串
        # 返回子串匹配上的第一个位置，若没有匹配上返回-1
        # test = ''
        # if type(mom_string) != type(test) or type(son_string) != type(test):
        #     return -1
        if len(son_string) == 0:
            return 0
        if len(mom_string) == 0:
            return -1
        # 求next数组
        next = [-1] * len(son_string)
        if len(son_string) > 1:  # 这里加if是怕列表越界
            next[1] = 0
            i, j = 1, 0
            while i < len(son_string) - 1:  # 这里一定要-1，不然会像例子中出现next[8]会越界的
                if j == -1 or son_string[i] == son_string[j]:
                    i += 1
                    j += 1
                    next[i] = j
                else:
                    j = next[j]

        # kmp框架
        m = s = 0  # 母指针和子指针初始化为0
        while (s < len(son_string) and m < len(mom_string)):
            # 匹配成功,或者遍历完母串匹配失败退出
            if s == -1 or mom_string[m] == son_string[s]:
                m += 1
                s += 1
            else:
                s = next[s]

        if s == len(son_string):  # 匹配成功
            return m - s
        # 匹配失败
        return -1

    @staticmethod
    def parseXML(data_path = './data/laptop/'):
        all_sentence_list = []
        DP_list = []
        # 使用minidom解析器打开 XML 文档
        DOMTree = xml.dom.minidom.parse("{}train.xml".format(data_path))
        collection = DOMTree.documentElement
        # 在集合中获取所有电影
        sentences = collection.getElementsByTagName("sentence")
        for sentence in sentences:
            current_sentence_dict = {}
            # 
            if sentence.hasAttribute("id"):
                current_sentence_dict["id"] = sentence.getAttribute("id")

            text = sentence.getElementsByTagName('text')[0]
            # 
            current_sentence_dict["text"] = text.childNodes[0].data
            sentence_words = nlp.word_tokenize(text.childNodes[0].data)

            current_sentence_dict["aspectTerm"] = []
            for term in sentence.childNodes:
                try:
                    if type(term) == xml.dom.minidom.Element:
                        if term.tagName == 'aspectTerms':
                            aspectTerms = term.getElementsByTagName("aspectTerm")
                            for aspectTerm in aspectTerms:
                                # 
                                term = aspectTerm.getAttribute('term')
                                current_sentence_dict["aspectTerm"].append(term)
                except:
                    continue
            tree = DPTree(current_sentence_dict['id'],current_sentence_dict['text'],current_sentence_dict['aspectTerm'])
            all_sentence_list.append(current_sentence_dict)
            DP_list.append(tree)
        # DPTree()
        return DP_list



class DPTree:
    def __init__(self,id='0',sentence = None,aspectTerms=[],orgin_text = '',max_modifying_dist = 6):
        self.sentence = sentence
        self.sentence_tree = self.generateTree()
        self.aspectTerms = aspectTerms
        self.id = id
        self.orgin_text = orgin_text
        self.sentence_words = nlp.word_tokenize(sentence)
        self.pos_list = nlp.pos_tag(sentence)
        self.word_dict = {i+1:value for i,value in enumerate(self.sentence_words)}
        self.word_dict[0] = '[pad]'
        self.pos_dict = {i+1:value[1] for i,value in enumerate(self.pos_list)}
        self.pos_dict[0] = 'ROOT'
        self.max_modifying_dist = max_modifying_dist


    def prase_dependency(self, g_ls, start, end):
        '''
        :param g_ls:
        :param start:
        :param end:
        :return: costs
                    trace
                    costs[end]
                    trace[end]
                    prase_present
        '''
        find_start = False
        find_end = False

        for tuple in g_ls:
            if start in tuple:
                find_start = True
            if end in tuple:
                find_end = True

        if find_start == False or find_end == False:#无穷大的情况
            result = []
            result.append([0 + self.max_modifying_dist])
            result.append([0])
            result.append(0 + self.max_modifying_dist)  # 起点到终点距离
            result.append([self.word_dict[i] for i in [0]])  # 起点到终点路径上所有点序号
            result.append(['undefined'])  # 起点到终点路径上所有语法关系组合
            result.append(['undefined'])  # 起点到终点路径上词性
            return result

        if start == end:#自身的情况
            result = []
            result.append([0])
            result.append([end])
            result.append(0)  # 起点到终点距离
            result.append([self.word_dict[i] for i in [end]])  # 起点到终点路径上所有点序号
            result.append(['self'])  # 起点到终点路径上所有语法关系组合
            result.append([self.pos_dict[i] for i in [end]])  # 起点到终点路径上词性
            return result

        dic = {}
        for si in g_ls:
            key = si[1]
            value = si[2]
            dic.setdefault(key, {})[value] = 1
            dic.setdefault(value, {})[key] = 1
        graph = dic

        costs = {}  # 记录start到其他所有点的距离
        trace = {start: [start]}  # 记录start到其他所有点的路径
        # 初始化costs
        for key in graph.keys():
            costs[key] = math.inf
        costs[start] = 0

        queue = [start]  # 初始化queue

        while len(queue) != 0:
            head = queue[0]  # 起始节点
            for key in graph[head].keys():  # 遍历起始节点的子节点
                dis = graph[head][key] + costs[head]
                if costs[key] > dis:
                    costs[key] = dis
                    temp = deepcopy(trace[head])
                    temp.append(key)
                    trace[key] = temp  # key节点的最优路径为起始节点最优路径+key
                    queue.append(key)

            queue.pop(0)  # 删除原来的起始节点
        if end > len(trace) or end < 0:
            return
        prase_present = []
        pos_present = []
        for i in range(len(trace[end]) - 1):
            s = trace[end][i]
            e = trace[end][i + 1]
            for ls in g_ls:
                start_index = ls[1]
                end_index = ls[2]
                if (s == ls[1] and e == ls[2]):
                    prase_present.append(ls[0])
                    pos_present.append(self.pos_list[end_index - 1][1])
                elif (s == ls[2] and e == ls[1]):
                    prase_present.append("-" + ls[0])
                    pos_present.append(self.pos_list[start_index - 1][1])
        result = []
        result.append(costs) # 起点到所有点的距离
        result.append(trace) # 起点到所有点的路径
        result.append(costs[end])  # 起点到终点距离
        # result.append([self.word_dict[i] for i in trace[end][:-1]])  # 起点到终点的路径的所有点序号
        result.append([self.word_dict[i] for i in trace[end]])
        result.append(prase_present + ['self'])  # 起点到终点的路径path
        # result.append([self.pos_dict[i] for i in trace[end][:-1]])# 起点到终点路径上所有点的词性
        result.append([self.pos_dict[i] for i in trace[end]])
        return result

    #绘图
    def drawDP(self, id='0', sentence='This French food tastes very well, but the restaurant has poor service.',aspectTerms=[]):
        origin_words = nlp.word_tokenize(sentence)
        all_aspect_words_list = [nlp.word_tokenize(aspect) for aspect in aspectTerms]

        all_colored_index = []
        color_index = 0
        dict_word = {}
        for aspect in all_aspect_words_list:
            aspect_from_index = kmp(origin_words, aspect) + 1
            aspect_to_index = aspect_from_index + len(aspect)
            all_colored_index.extend([i for i in range(aspect_from_index, aspect_to_index)])
            for j in range(aspect_from_index, aspect_to_index):
                dict_word[j] = colors[color_index]
            color_index += 1

        origin_words_add_root = ['root'] + origin_words

        words = []
        for (i, word) in enumerate(origin_words_add_root):
            words.append(str(i) + '_' + word)

        s_dependency_parse = nlp.dependency_parse(sentence)
        
        

        g = Digraph(id)

        g.node(name=sentence)

        for (i, word) in enumerate(words):
            word_id = int(word.split('_')[0])
            if (word_id in all_colored_index):
                g.node(name=word, _attributes={'color': dict_word[word_id], 'fontcolor': dict_word[word_id]})
            else:
                g.node(name=word, _attributes={'color': 'black', 'fontcolor': 'black'})

        for tripple in s_dependency_parse:
            g.edge(words[tripple[1]], words[tripple[2]], label=tripple[0])

        g.view()

    def generateTree(self):
        return nlp.dependency_parse(self.sentence)

    def getTermIndexInSentence(self,Term=''):
        aspectTerm_words = nlp.word_tokenize(Term)
        from_index = tool.kmp(self.sentence_words,aspectTerm_words) + 1
        to_index = from_index + len(aspectTerm_words)
        return [i for i in range(from_index,to_index)]

    # 计算任意两个节点的相对距离
    # sentence_tree表示Dependency树状结构，形如[('ROOT', 0, 3), ('det', 2, 1),...]
    # origin_singleToken_index表示起始位置集合,形如[3,4]
    # target_singleToken_index表示结束位置集合，形如[7,8]
    # 返回从origin_singleToken_index到target_singleToken_index的距离，以两个集合到达的最短距离
    def calcDist_for_index(self, origin_singleToken_index = [],target_singleToken_index = []):

        min_relative_dist = float('inf')
        flag = False
        for i in origin_singleToken_index:
            for j in target_singleToken_index:
                try:
                    return_res = self.prase_dependency(self.sentence_tree, i, j)
                    min_relative_dist = min(min_relative_dist, return_res[2])
                except:
                    flag = True
        if flag and self.sentence not in all_error_sen:
            
            
            all_error_sen.add(self.sentence)
            return 0
        return min_relative_dist

    # 计算任意两个节点的传递位置路径
    # sentence_tree表示Dependency树状结构，形如[('ROOT', 0, 3), ('det', 2, 1),...]
    # origin_singleToken_index表示起始位置集合,形如[3,4]
    # target_singleToken_index表示结束位置集合，形如[7,8]
    # 返回从origin_singleToken_index到target_singleToken_index的距离，以两个集合到达的最短路径，形如[ '-nsubj','det',...]
    def calcRoute_for_index(self, origin_singleToken_index = [],target_singleToken_index = []):
        path = []
        for i in origin_singleToken_index:
            for j in target_singleToken_index:
                return_res = self.prase_dependency(self.sentence_tree, i, j)
                path.append(return_res[4])
        leng = float('inf')
        res = []
        for p in path:
            leng = min(leng, len(p))
        for p in path:
            if len(p) == leng:
                res.append(p)
        return res


    # 计算任意两个节点的传递位置路径，包括依存路径的类型，path等
    def calcTriples_for_index(self, origin_singleToken_index = [],target_singleToken_index = []):
        triple = []

        for i in origin_singleToken_index:
            for j in target_singleToken_index:
                return_res = self.prase_dependency(self.sentence_tree, i, j)
                current_triple_dict = {
                    'modification_path':return_res[4],
                    'node_path':return_res[3],
                    'pos_path':return_res[5]
                }
                triple.append(current_triple_dict)
        leng = float('inf')
        res = []
        for p in triple:
            leng = min(leng, len(p['modification_path']))
        for p in triple:
            if len(p['modification_path']) == leng:
                res.append(p)
        return res


    def calcDist_for_two_singleToken(self,origin_singleToken='',target_singleToken=''):
        origin_singleToken_index = self.getTermIndexInSentence(origin_singleToken)
        target_singleToken_index = self.getTermIndexInSentence(target_singleToken)
        return self.calcDist_for_index(origin_singleToken_index,target_singleToken_index)

    def calcRoute_for_two_singleToken(self,origin_singleToken='',target_singleToken=''):
        origin_singleToken_index = self.getTermIndexInSentence(origin_singleToken)
        target_singleToken_index = self.getTermIndexInSentence(target_singleToken)
        return self.calcRoute_for_index(origin_singleToken_index,target_singleToken_index)


    def calcDist_for_aspect_singleToken(self,origin_singleToken=''):
        origin_singleToken_index = self.getTermIndexInSentence(origin_singleToken)
        aspect_index = []
        # aspect_start_index = nlp.word_tokenize(self.orgin_text).index('$') + 1

        aspect_start_index = -1
        ls = nlp.word_tokenize(self.orgin_text)
        for i in range(len(ls)-1):
            if ls[i] == '$' and ls[i + 1] == 'T$':
                aspect_start_index = i + 1
                break
        if aspect_start_index == -1:
            
            return

        for i in range(0, len(nlp.word_tokenize(self.aspectTerms[0]))):
            aspect_index.append(aspect_start_index + i)
        return self.calcDist_for_index(origin_singleToken_index,aspect_index)

    def calcRoute_for_aspect_singleToken(self,origin_singleToken=''):
        origin_singleToken_index = self.getTermIndexInSentence(origin_singleToken)
        aspect_index = []
        # aspect_start_index = nlp.word_tokenize(self.orgin_text).index('$') + 1

        aspect_start_index = -1
        ls = nlp.word_tokenize(self.orgin_text)
        for i in range(len(ls)-1):
            if ls[i] == '$' and ls[i + 1] == 'T$':
                aspect_start_index = i + 1
                break
        if aspect_start_index == -1:
            
            return


        for i in range(0, len(nlp.word_tokenize(self.aspectTerms[0]))):
            aspect_index.append(aspect_start_index + i)
        return self.calcRoute_for_index(origin_singleToken_index,aspect_index)

    def calcDist_for_aspect_singleIndex(self,origin_singleIndex=[]):
        aspect_index = []
        # aspect_start_index = nlp.word_tokenize(self.orgin_text).index('$') + 1

        aspect_start_index = -1
        ls = nlp.word_tokenize(self.orgin_text)
        for i in range(len(ls)-1):
            if ls[i] == '$' and ls[i + 1] == 'T$':
                aspect_start_index = i + 1
                break
        if aspect_start_index == -1:
            
            return


        for i in range(0, len(nlp.word_tokenize(self.aspectTerms[0]))):
            aspect_index.append(aspect_start_index + i)
        return self.calcDist_for_index(origin_singleIndex,aspect_index)

    def calcRoute_for_aspect_singleIndex(self,origin_singleIndex=[]):
        aspect_index = []
        aspect_start_index = -1
        ls = nlp.word_tokenize(self.orgin_text)
        for i in range(len(ls)-1):
            if ls[i] == '$' and ls[i + 1] == 'T$':
                aspect_start_index = i + 1
                break
        if aspect_start_index == -1:
            
            return

        for i in range(0, len(nlp.word_tokenize(self.aspectTerms[0]))):
            aspect_index.append(aspect_start_index + i)
        return self.calcRoute_for_index(origin_singleIndex,aspect_index)

    def calcTriples_for_aspect_singleIndex(self, origin_singleIndex=[]):
        aspect_index = []
        aspect_start_index = -1
        ls = nlp.word_tokenize(self.orgin_text)
        for i in range(len(ls)-1):
            if ls[i] == '$' and ls[i + 1] == 'T$':
                aspect_start_index = i + 1
                break
        if aspect_start_index == -1:
            
            return

        for i in range(0, len(nlp.word_tokenize(self.aspectTerms[0]))):
            aspect_index.append(aspect_start_index + i)
        return self.calcTriples_for_index(origin_singleIndex,aspect_index)[0]

    def get_tree_informations(self):
        all_struct_info = []
        all_words = self.sentence_words

        for index in range(1, len(all_words) + 1):
            triple = self.calcTriples_for_aspect_singleIndex([index])
            current_tree_info = {}
            current_tree_info['relative_dist'] = len(triple['node_path']) if len(triple['node_path']) < self.max_modifying_dist else self.max_modifying_dist
            current_tree_info['modification_path'] = triple['modification_path'] if len(triple['modification_path']) < self.max_modifying_dist else ['undefined']
            current_tree_info['node_path'] = triple['node_path'] if len(triple['node_path']) < self.max_modifying_dist else ['undefined']
            current_tree_info['pos_path'] = triple['pos_path'] if len(triple['pos_path']) < self.max_modifying_dist else ['undefined']

            current_tree_info['modification_path'] = current_tree_info['modification_path'] + ['undefined'] * (self.max_modifying_dist - 1 - len(current_tree_info['modification_path']))
            current_tree_info['node_path'] = current_tree_info['node_path'] + ['undefined'] * (self.max_modifying_dist - 1 - len(current_tree_info['node_path']))
            current_tree_info['pos_path'] = current_tree_info['pos_path'] + ['undefined'] * (self.max_modifying_dist - 1 - len(current_tree_info['pos_path']))
            all_struct_info.append(current_tree_info)

        return all_struct_info

if __name__ == '__main__':

    # tree = DPTree(id='0',sentence='My real problem with it ? The statement of 7 hour battery life is not just mere exaggeration -- it \'s a lie .',\
    #          aspectTerms=['battery life'],\
    #          orgin_text ='My real problem with it ? The statement of 7 hour $T$ is not just mere exaggeration -- it \'s a lie .')

    tree = DPTree(id='0',
                  sentence='Would n\'t you know it ?',
                  aspectTerms=['staff'],
                  orgin_text='But the $T$ was so horrible to us .'
                  )
    try:
        tree.drawDP(id='0',sentence = tree.sentence,aspectTerms=['staff'])
    except:
        pass

    # tree.get_tree_informations()
    #
    
    # 
    # 
    #
    # 
    nlp.close()