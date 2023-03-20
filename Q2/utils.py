"""
author: 
date: 2021/4/8 
func:
"""
import string
import time
import pandas as pd
import re
from collections import defaultdict
import networkx as nx
import numpy as np
from scipy.signal import find_peaks
from jieba.analyse import extract_tags
from langdetect import detect_langs
from nltk.chunk import ne_chunk,tree2conlltags
from nltk import pos_tag,word_tokenize
import random
import os

def cache(cache_name,conn,query,result):
    """缓存mongo查询结果"""
    # 如果存在则先删除
    print(os.path.exists(cache_name))
    if os.path.exists(cache_name):
        os.remove(cache_name)
    print(os.path.exists(cache_name))
    info = pd.DataFrame(list(conn.find(query,result)))
    print(info)
    info.to_csv(cache_name,index=1)

def find_cluster_num(n_posts):
    '''自动寻找合适的主题数量'''
    print(n_posts)
    n_cluster = 0
    if n_posts<=50:
        n_cluster = min(5,int(n_posts/5)+1)
    if n_posts>50 and n_posts<=100:
        n_cluster = 5 + int((n_posts-50)/10)
    if n_posts>100 and n_posts<=1000:
        n_cluster = 10 + int((n_posts-100)/60)
    if n_posts>1000:
        n_cluster = min(40,25 +int((n_posts-1000)/100))
    print("n_cluster",n_cluster)
    return n_cluster

def get_key_words(posts:list,pos_tags:dict):
    '''
    keywords with tf-idf
    :param posts: 句子列表
    :return: 画词云的词列表wordcloud  送入事件分析的关键词列表phrases, 语种
    '''
    pos_eng = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # 动 名 形
    pos_chn = ['a','ad','an','f','n','nr','ns','nt','nz','v','vg','vd','vn'] # 动 名 形
    jieba_posts = "".join(posts)  # 需要整理为一个字符串
    jieba_keywords = []
    wordcloud = defaultdict(int)
    top_words = extract_tags(sentence=jieba_posts, withWeight=True, topK=30)
    for word, weight in top_words:
        if len(jieba_keywords) == 10: break
        if word not in pos_tags: continue
        if pos_tags[word] in pos_eng + pos_chn: jieba_keywords.append(word)
    for word,weight in top_words[:20]:
        wordcloud[word] = int(weight*100)
    word_cloud = [{"name":word,"value":freq} for word, freq in wordcloud.items()]

    if len(jieba_keywords)==0: # 没有关键词析出
        return word_cloud, [], 'None'
    else:
        # 组合为词组
        phrases = []
        for p in posts:
            if len(phrases) == 10: break
            try:
                langauage = detect_langs(p)
                for l in langauage:
                    top_lang = l.lang
                    break
                if top_lang not in ["en", "zh-cn", "zh-tw", "zh-yue"]: continue  # 其他语种不需要拼接
            except Exception as e:
                pass
            p = p.split(' ')  # string -> list
            if len(p) < 2: continue
            for kw in jieba_keywords:
                try:
                    idx = p.index(kw)
                    if idx != 0 and p[idx - 1] in jieba_keywords:
                        if top_lang != "en":
                            phrases.append(p[idx - 1] + p[idx])
                        else:
                            phrases.append(p[idx - 1] + " " + p[idx])
                    elif idx != len(p) - 1 and p[idx + 1] in jieba_keywords:
                        if top_lang != "en":
                            phrases.append(p[idx] + p[idx + 1])
                        else:
                            phrases.append(p[idx] + " " + p[idx + 1])
                    else:
                        continue
                except ValueError as err:  # 列表中不存在元素
                    pass
        phrases = list(set(phrases))[:5]  # 最大记录5个

        # 应对不同情况的返回策略
        if len(phrases) > 0:
            return word_cloud, phrases, top_lang
        else:
            return word_cloud, jieba_keywords[:5], top_lang

def aggregate_timeline(timeline:list,gap="1D"):
    """
    aggregate pt time lines
    :param timeline: list of pts(10-bit)
    :return: {"yyyy-mm-dd":freq}
    """
    delta_days = (max(timeline)-min(timeline))/(24*3600)
    if delta_days<30: gap = "1D"
    elif delta_days<180: gap="1W"
    else: gap="1M"
    post_time = pd.Series([time.strftime("%Y-%m-%d", time.localtime(x)) for x in timeline])
    index = pd.Series([1] * (len(post_time)), index=pd.DatetimeIndex(post_time))  # 以时间为索引的series可以使用resample聚合
    timeSeries = index.resample(gap).sum()
    timeSeries = dict(timeSeries.reindex(timeSeries.index.astype(str)))  # 更改索引类型并把series转换为字典
    for date, freq in timeSeries.items():
        timeSeries[date] = int(freq)  # 更改值类型变为py内置int
    return timeSeries

def extractHashtag(cont:list):
    '''
    :param cont: 句子列表
    :return:  {hashtag:freq}
    '''
    hashtags= defaultdict(int)
    for x in cont:
        search = re.findall("#(.*?) ", x) + re.findall("【(.*?)】",x) + re.findall("[(.*?)]",x)
        for s in search:
            s = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', s)  # 删除无用url
            s = re.sub(r'\’s','',s) # 删除's
            s = re.sub(r'|(.*) ','',s) #删除 A|B这种tag后面的B
            tags = s.split('#')  # 二次处理#
            for t in tags:
                # 删除各种标点
                t_out = ''.join(ch for ch in t if ch not in list(string.punctuation)+['…','’','”'])
                # 限制长度
                t_out = t_out[:15]
                hashtags[t_out] += 1
    hashtags_sorted = sorted(hashtags.items(),key=lambda x:x[1], reverse=True)
    top_hashtags = {x[0]:x[1] for x in hashtags_sorted}  # 返回排好序的hashtag但不限制数量
    return top_hashtags

def extractGlobalHashtag(cont:pd.DataFrame):
    hashtags = defaultdict(int)
    hashtag_time = {}
    hashtag_mid = {}
    def _get_tags(x):
        try:
            search = re.findall("#(.*?)[ |】]", str(x['cont']))
            for s in search:
                s = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', s)  # 删除无用url
                s = re.sub(r'\’s', '', s)  # 删除's
                s = re.sub(r'|(.*) ', '', s)  # 删除 A|B这种tag后面的B
                tags = s.split('#')  # 二次处理#
                for t in tags:
                    # 删除各种标点
                    t_out = ''.join(ch for ch in t if ch not in list(string.punctuation) + ['…', '’', '”'])
                    # 限制长度
                    t_out = t_out[:15]
                    hashtags[t_out] += 1
                    if t_out not in hashtag_time:
                        hashtag_time[t_out] = [int(x['pt']/1000)]
                    else:
                        hashtag_time[t_out].append(int(x['pt']/1000))
                    if t_out not in hashtag_mid:
                        hashtag_mid[t_out] = [x['mid']]
                    else:
                        hashtag_mid[t_out].append(x['mid'])
        except:
            pass
    cont.apply(_get_tags,axis=1)
    for tag,timeline in hashtag_time.items():
        hashtag_time.update({tag:aggregate_timeline(timeline)})

    return hashtags, hashtag_time, hashtag_mid

def extractGlobalLocation(lang,posts,ht):
    """
    对输入的dataframe抽取地名
    :param lang: 语种
    :param posts: dataframe
    :param ht:  harvesttext 类
    :return: [{name:xx, value:xxx},{}]
    """
    location = defaultdict(int)
    for p in list(posts['cont']):
        if lang in ["zh-cn", 'zh-tw', 'zh-yue']:
            res = ht.named_entity_recognition(p)
            for term, ner in res.items():
                if ner == "地名": location[term] += 1
        elif lang == "en":
            tokens = word_tokenize(p)
            tagged = pos_tag(tokens)
            entities = tree2conlltags(ne_chunk(tagged))  # [(word,O,NE),(),()]
            for word in entities:
                if word[2] == "B-GPE": location[word[0]] += 1
        else:
            pass
    sorted_location = sorted(location.items(), key=lambda x: x[1], reverse=True)[:10]
    top_location = []
    for l in sorted_location:
        top_location.append({"name": l[0], "value": l[1]})
    return top_location

def extractGloablSent(lang,posts,ht):
    sent_type = {"正面": 0, "中性": 0, "负面": 0}
    sent_type_ratio = {"正面": 0., "中性": 0., "负面": 0.}
    try:
        if lang == "en":
            from harvesttext.resources import get_english_senti_lexicon
            sent_lexicon = get_english_senti_lexicon()
            sent_dict = ht.build_sent_dict(list(posts['cont']), pos_seeds=sent_lexicon["pos"], neg_seeds=sent_lexicon["neg"],
                                           scale="+-1")
        else:
            sent_dict = ht.build_sent_dict(list(posts['cont']), min_times=1, scale="+-1")
        total_seeds = 0
        for p in list(posts['cont']):
            if ht.analyse_sent(p) > 0:
                sent_type['正面'] += 1
                total_seeds += 1
            elif ht.analyse_sent(p) == 0:
                sent_type['中性'] += 1
                total_seeds += 1
            else:
                sent_type['负面'] += 1
                total_seeds += 1
        # 转换为比例
        for k,v in sent_type.items():
            sent_type_ratio.update({k:round(v/total_seeds,3)})
        sorted_ratio = sorted(sent_type_ratio.items(), key=lambda x: x[1], reverse=True)
        if sorted_ratio[0][1] + sorted_ratio[1][1] > 1:  # 最大的两个加和超过1，此时第三个肯定是0，那么削减第二个
            sent_type_ratio.update({sorted_ratio[1][0]: 1 - sorted_ratio[0][1]})
        elif sorted_ratio[0][1] + sorted_ratio[1][1] + sorted_ratio[2][1] > 1:  # 三个加和超过1，削减第三个
            sent_type_ratio.update({sorted_ratio[2][0]: 1 - sorted_ratio[0][1] - sorted_ratio[1][1]})
        else:
            pass
        return {"sentiment": sorted(sent_type.items(), key=lambda x: x[1], reverse=True)[0][0], "sentiment_score_num": sent_type,"sentiment_score_ratio":sent_type_ratio}
    except Exception as err:  # 文本中不包含情感种子
        return {"sentiment": "中性", "sentiment_score_num": sent_type,"sentiment_score_ratio":{"正面": 0.333, "中性": 0.334, "负面": 0.333}}


def aggregate(record:pd.DataFrame, gap="1W",find_peak=False):
    """在时间线上聚合贴文并发现峰值"""
    # 13位时间戳
    post_time = record['pt'].map(lambda x: int(x / 1000))
    post_time = post_time.map(lambda x: time.strftime("%Y-%m-%d", time.localtime(x)))
    index = pd.Series([1] * (len(post_time)), index=pd.DatetimeIndex(post_time))  # 以时间为索引的series可以使用resample聚合
    timeSeries = index.resample(gap).sum()
    ts = np.array(timeSeries)
    if find_peak:
        # 寻找峰值和峰值区间
        if gap=="1W": delta=7
        elif gap=="1M": delta=30
        else: delta=1
        duration = (list(record['pt'].values)[-1] - list(record['pt'].values)[0])/(3600*24*delta)
        if len(record)>1:
            avg_postnum = max(1,int(len(record)/duration))  # 平均每个gap的活跃度
        else:
            avg_postnum = 0
        peaks, _ = find_peaks(x=ts, height=3*avg_postnum)
        peaks = sorted([x[0] for x in sorted(dict(zip(peaks,ts[peaks])).items(),key=lambda x:x[1],reverse=True)][:10])
        time_index = [str(x)[:10] for x in list(timeSeries.index)]  # yyyy-mm-dd格式
        # 时间序列格式修改
        timeSeries = dict(timeSeries.reindex(timeSeries.index.astype(str)))  # 更改索引类型并把series转换为字典
        for date, freq in timeSeries.items():
            timeSeries[date] = int(freq)  # 更改值类型变为py内置int
        return timeSeries, [time_index[idx] for idx in peaks]
    else:
        timeSeries = dict(timeSeries.reindex(timeSeries.index.astype(str)))  # 更改索引类型并把series转换为字典
        for date, freq in timeSeries.items():
            timeSeries[date] = int(freq)  # 更改值类型变为py内置int
        return timeSeries


def createGraph(edges,n_edges):
    """建图算法"""
    # build the Graph
    GM = nx.DiGraph()
    lookup = {}
    r_lookup = {}
    cnt = 0
    n_edge = 0
    edges = sorted(edges.items(),key=lambda x:x[1], reverse=True)
    for edge, freq in edges:
        if n_edge>n_edges: break
        out_node, in_node = edge.split('>')
        # 避免A到A的边 和 重复节点计入
        if out_node != in_node and ((out_node not in lookup) or (in_node not in lookup)):
            GM.add_edge(out_node, in_node, freq=freq)
            n_edge += 1
        if out_node not in lookup:
            lookup[out_node] = cnt
            r_lookup[cnt] = out_node
            cnt += 1
        if in_node not in lookup:
            lookup[in_node] = cnt
            r_lookup[cnt]  = in_node
            cnt += 1
    print(len(edges))
    return GM,lookup,r_lookup

def add_profile(profile:dict,sname:str):
    profile.pop("_id")
    profile.update({"photo": f"https://twitter.com/{sname}/photo"})
    return profile

def mediaField(wordcloud,lang):
    """
    生成所有行业的属性；求得每个行业的属性，进行归一化
    :param media_dict:
    :return:
    """
    # 语种的判断；因为只有汉语的词典；
    media_field = []
    if lang not in ['zh','zh-cn','zh-tw','zh-yue'] :
        # 非中文，随机返回五个领域值
        field_dim = {}
        field_dim["军事"] = random.uniform(0.,100.)
        field_dim["娱乐"] = random.uniform(0.,100.)
        field_dim["政治"] = random.uniform(0.,100.)
        field_dim["法律"] = random.uniform(0.,100.)
        field_dim["经济"] = random.uniform(0.,100.)
        media_field = [{"name":key,"value":round(value,2)} for key,value in field_dim.items()]
    else:
        #处理传过来的词,输入的高频词是[{name:xxx, value:xxx}]形式的
        word_dict = {}
        for pair in wordcloud:
            word_dict.update({pair['name']:max(int(pair['value']*10),1)})
        militaryDict = generate_dict("./worddict/军事词汇.txt", encode='utf-8')
        militaryNum = max(single_attri(militaryDict, word_dict),1)
        amusementDict = generate_dict("./worddict/娱乐词汇.txt")
        amusementNum = max(single_attri(amusementDict, word_dict),1)
        politicsDict = generate_dict("./worddict/政治词汇.txt")
        politicsNum = max(single_attri(politicsDict, word_dict),1)
        lawDict = generate_dict("./worddict/法律词汇.txt")
        lawNum = max(single_attri(lawDict, word_dict), 1)
        economicsDict = generate_dict("./worddict/财经词汇.txt")
        economicsNum = max(single_attri(economicsDict, word_dict), 1)
        #归一化
        nums = militaryNum + amusementNum + politicsNum + lawNum + economicsNum
        field_dim = {}
        field_dim["军事"] = max(round((militaryNum / nums )* 100, 2), 1.)
        field_dim["娱乐"] = max(round((amusementNum / nums) * 100, 2), 1.)
        field_dim["政治"] = max(round((politicsNum / nums) * 100, 2), 1.)
        field_dim["法律"] = max(round((lawNum / nums) * 100, 2), 1.)
        field_dim["经济"] = max(round((economicsNum / nums) * 100, 2), 1.)
        media_field = [{"name": key, "value": value} for key, value in field_dim.items()]
    return media_field


def single_attri(attri_dict, word_dict):
    """
    由传入的行业属性词典和媒体的关键词词典，生成媒体属性中这个行业的占比；
    :param attri_dict:
    :return:
    """
    nums = 0
    for word,weight in word_dict.items():
        if word in attri_dict:
            nums += weight
    return nums

def generate_dict(path,encode = 'gbk'):
    field_words = []
    with open(path,'r',encoding=encode) as f:
        cont = f.read().splitlines()
        for word in cont:
            field_words.append(word.strip())
    return field_words

