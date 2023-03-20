"""
author:
date: 2021/4/6
func: 媒体画像
"""
import re
import pandas as pd
import pymongo
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD
import time
import warnings
warnings.filterwarnings("ignore")
from utils import *
from langdetect import detect_langs
from harvesttext import HarvestText
from nltk import pos_tag,word_tokenize
from nltk.chunk import ne_chunk,tree2conlltags
from jieba.analyse import extract_tags
import pickle
from WordCloudUtils import *

class MediaProfileModel():
    def __init__(self,uid, svd_dim, kmeans_iter, merge_iter):
        self.uid = uid
        print(f"=======================processing {uid}==================================")
        self.svd_dim = svd_dim
        self.kmeans_iter = kmeans_iter
        self.merge_iter = merge_iter
        self.info = {}  # 全部的分析结果
        self.global_tmp = {}  # 存储每个成员函数需要提供给别的函数的值。不允许每个值都单独注册到全局
        self.skip = False # 如果content中根本找不到信息，那也不需要对这个人画像了


    def dataFetchAndLoad(self):
        """访问数据库，获得所需数据并缓存下来。减少多次与数据库通信的时间"""
        print("Loading Data: posts, profiles from media and its audience\n")

        def _process_pt(x):
            if x.isdigit():
                return int(x)
            else:
                timeArray = time.strptime(x, "%Y-%m-%d %H:%M:%S")
                timestamp = time.mktime(timeArray)
                return timestamp * 1000
        # 加载该账号的推文
        self.posts = pd.read_csv(f"cache/{self.uid}_posts.csv", lineterminator='\n', dtype=object,index_col=0)
        self.posts['pt'] = self.posts['pt'].apply(_process_pt)
        self.mids = list(self.posts['mid'])
        self.rmids = list(self.posts['r_mid'].dropna())
        self.posts['cont'] = self.posts['cont'].fillna("")

        # 加载follower的推文
        self.audience_posts = pd.read_csv(f"cache/{self.uid}_audience_posts.csv",lineterminator='\n',dtype=object,index_col=0)
        self.audience_posts['cont'] = self.audience_posts['cont'].fillna("")
        self.audience_posts['pt'] = self.audience_posts['pt'].apply(_process_pt)
        self.audience_uids = list(set(list(self.audience_posts['uid'])))
        # 加载follower的个人资料
        self.audience_profiles = pd.read_csv(f"cache/{self.uid}_audience_profiles.csv", lineterminator='\n',dtype=object,index_col=0)
        self.audience_profiles['nfans']  = self.audience_profiles['nfans'].fillna(0)
        self.audience_profiles['nfans'] = self.audience_profiles['nfans'].apply(lambda x: int(x))
        self.uid2sname = dict(
            zip(list(self.audience_profiles['uid']), list(self.audience_profiles['sname'])))
        self.uid2uname = dict(
            zip(list(self.audience_profiles['uid']), list(self.audience_profiles['uname'])))
        self.sname2nfans = dict(
            zip(list(self.audience_profiles['sname']), list(self.audience_profiles['nfans'])))

        # 加载 follow的推文
        self.focus_posts = pd.read_csv(f"cache/{self.uid}_focus_posts.csv",lineterminator='\n',dtype=object,index_col=0)
        self.focus_uids = list(set(list(self.focus_posts['uid'])))
        self.focus_posts['pt'] = self.focus_posts['pt'].apply(_process_pt)
        self.focus_posts['cont'] = self.focus_posts['cont'].fillna("")
        # 加载 follow的个人资料
        self.focus_profiles = pd.read_csv(f"cache/{self.uid}_focus_profiles.csv", lineterminator='\n', dtype=object, index_col=0)
        self.focus_profiles['nfans'] = self.focus_profiles['nfans'].fillna(0)
        self.focus_profiles['nfans'] = self.focus_profiles['nfans'].apply(lambda x: int(x))
        self.uid2sname.update(
            dict(zip(list(self.focus_profiles['uid']), list(self.focus_profiles['sname']))))
        self.uid2uname.update(
            dict(zip(list(self.focus_profiles['uid']), list(self.focus_profiles['uname']))))
        self.sname2nfans.update(
            dict(zip(list(self.focus_profiles['sname']), list(self.focus_profiles['nfans']))))

        # 加载该账号的个人资料
        self.profile = pd.read_csv(f"cache/{self.uid}_profile.csv", lineterminator='\n', dtype=object,index_col=0)
        self.profile['nfans'] = self.profile['nfans'].apply(lambda x: int(x))
        self.sname = self.profile['sname'].values[0]
        self.uid2sname.update({self.uid: self.sname})
        self.uname = self.profile['uname'].values[0]
        self.uid2uname.update({self.uid:self.uname})
        self.sname2nfans.update({self.profile['sname'].values[0]:max(int(self.profile['nfans'].values[0]),1)})

        self.sname2uid = {v:k for k,v in self.uid2sname.items()}
        # 加载媒体账号的sname列表
        self.total_media = pd.concat([pd.read_csv(f"media_account/HKMedia.csv", dtype=object,index_col=0),pd.read_csv(f"media_account/TWMedia.csv", dtype=object,index_col=0)],ignore_index=True)
        self.media_snames = self.total_media['sname'].tolist()  # 参与媒体的sname列表

        # 形成画像对象自己的基础资料
        self.uid_profile = list(self.profile.T.to_dict().values())[0]
        self.uid_profile = add_profile(self.uid_profile, self.uid_profile['sname'])
        self.uid_profile.update(self.uid_profile)
        print(self.uid_profile)
        try:
            self.uid_profile.update({"npost": max(len(self.posts),int(self.uid_profile['nmedia']),int(self.uid_profile['ntweet']))})
        except:
            self.uid_profile.update({"npost": len(self.posts)})
        self.uid_profile.update({"is_media": self.sname in self.media_snames})
        if 'nmedia' in self.uid_profile: self.uid_profile.pop("nmedia")
        if 'ntweet' in self.uid_profile: self.uid_profile.pop("ntweet")
        print("All data is ready !!!")


    def event_cluster(self):
        '''执行二分kmeans算法'''
        if not os.path.exists(f"cache/{self.uid}_postags.pickle"):
            _common_texts,total_mids,total_postags = cut_docs(self.posts, get_postag=True)
            with open(f"cache/{self.uid}_postags.pickle","wb") as f:
                pickle.dump(total_postags,f)
        else:
            _common_texts, total_mids, _ = cut_docs(self.posts)
            with open(f"cache/{self.uid}_postags.pickle","rb") as f:
                total_postags = pickle.load(f)
            print("loading pos tags from cache")

        self.total_mids = total_mids
        self.total_postags = total_postags
        common_texts = []
        for text in _common_texts:
            common_texts.append(' '.join(text))
        self.tfidf_skip = 0
        try:
            tfidf_transformer = TfidfVectorizer()
            input_feat = tfidf_transformer.fit_transform(common_texts)
        except ValueError as exception:  # 不支持的语言，或者全是表情符号,事件这里是空
            print(exception,"tfidf")
            self.cluster_post = []
            self.tfidf_skip = 1
            return 1
        # TODO  SVD在服务器上运行会报段错误，只能改成内置算法为arpack，相当于调用PCA,所以维度要求小于min(A.shape)
        try:
            input_feat_svd = TruncatedSVD(n_components=min(self.svd_dim, min(input_feat.shape) - 1), algorithm='arpack').fit_transform(input_feat)
            print(f"svd dim: ", input_feat_svd.shape[1])
        except:
            # 什么方法都做不了降维，直接把所有贴文看为一类
            print("svd  wrong")
            self.cluster_post = []
            event = {}
            event['event']=1
            event['event_id']=self.uid+'-1'
            event['volume'] = len(self.posts)
            event['mids'] = self.posts['mid'].tolist()
            event['posts'] = self.posts['cont'].tolist()
            event['time'] = int(self.posts['pt'].mean())
            self.cluster_post.append(event)
            return 0
        print("dim reduction done")
        # TODO: 测试时可以减少cluster数量
        n_cluster = min(len(self.posts),find_cluster_num(len(self.posts)))
        #n_cluster = min(5,len(self.posts))
        print(n_cluster)

        text_vec_map = []  # 文档和文档向量
        vecs = [] # 存储文档向量的序列
        for i in range(len(common_texts)):
            text_vec_map.append((''.join(common_texts[i]), input_feat_svd[i]))
            vecs.append(list(input_feat_svd[i]))

        ncluster_gridsearch   = [] # [n-45, n+10]
        ncluster_gridsearch += [n_cluster+5, n_cluster+10, n_cluster+15, n_cluster+20]
        for i in range(8):
            if n_cluster>i*5+2: ncluster_gridsearch.append(n_cluster-i*5)


        #ncluster_gridsearch = [n_cluster + 5, n_cluster + 10]
        # TODO: 削减搜索功能提高速度
        ncluster_gridsearch = [n_cluster]
        gridsearch_cluster_vec = {} # {epoch:processing_cluster_vec}
        gridsearch_CH = {}
        for epoch in range(len(ncluster_gridsearch)):
            processing_cluster_vec = {}
            processing_cluster_sse = {}
            processing_cluster_vol = {}
            counter = 0
            processing_cluster_vec[counter] = input_feat_svd
            processing_cluster_sse[counter] = -1
            processing_cluster_vol[counter] = len(input_feat_svd)
            next = counter
            counter += 1
            ncluster_in_epoch = ncluster_gridsearch[epoch]
            while(len(processing_cluster_vec)<ncluster_in_epoch):
                iter_input_feat = processing_cluster_vec[next]
                processing_cluster_vec.pop(next)
                processing_cluster_sse.pop(next)
                processing_cluster_vol.pop(next)
                iter_total_sse = {}
                iter_vec = []
                iter_sse = {}
                for iter in range(self.kmeans_iter):
                    if len(common_texts)>5000:
                        ks = MiniBatchKMeans(n_clusters=2, random_state=np.random.randint(0,10000), max_iter=1000)
                    else:
                        ks = KMeans(n_clusters=2, random_state=np.random.randint(0,10000), n_init=100, max_iter=1000, n_jobs=4)
                    labels = np.array(ks.fit_predict(iter_input_feat))
                    label_0 = (labels==0)
                    label_1 = (labels==1)
                    label_0_feat = iter_input_feat[label_0]
                    label_1_feat = iter_input_feat[label_1]
                    label_0_num = len(label_0)
                    label_1_num = len(label_1)
                    label_0_ctr = ks.cluster_centers_[0].reshape(1,-1)
                    label_1_ctr = ks.cluster_centers_[1].reshape(1,-1)
                    if len(label_0)==1: label_0_sse = 0
                    else:label_0_sse = np.sum(np.square(label_0_feat - np.tile(label_0_ctr,(len(label_0_feat),1))))
                    if len(label_1)==1: label_1_sse = 0
                    else:label_1_sse = np.sum(np.square(label_1_feat - np.tile(label_1_ctr, (len(label_1_feat),1))))
                    iter_total_sse[iter] = label_0_sse + label_1_sse
                    iter_sse[iter] = (label_0_sse,label_1_sse)
                    iter_vec.append((label_0_feat,label_1_feat))

                # 决定本轮放哪两个子簇进来:sse之和最小
                min_sse_subclusters = min(iter_total_sse,key=iter_total_sse.get)
                processing_cluster_vec[counter] = iter_vec[min_sse_subclusters][0]
                processing_cluster_sse[counter] = iter_sse[min_sse_subclusters][0]
                processing_cluster_vol[counter] = len(processing_cluster_vec[counter])
                counter += 1
                processing_cluster_vec[counter] = iter_vec[min_sse_subclusters][1]
                processing_cluster_sse[counter] = iter_sse[min_sse_subclusters][1]
                processing_cluster_vol[counter] = len(processing_cluster_vec[counter])
                counter += 1
                next_strategy = "sse"
                if next_strategy == "sse":
                    # 下一轮拆sse最大的簇
                    next_candidate = sorted(processing_cluster_sse.items(), key=lambda x: x[1], reverse=True)
                    for e in next_candidate:
                        idx = e[0]
                        if processing_cluster_vec[idx].shape[0] >= 2:
                            next = idx
                            break
                        else:
                            continue
                elif next_strategy == "vol":
                    # 下一轮拆容量最大的簇
                    next_candidate = sorted(processing_cluster_vol.items(), key=lambda x: x[1], reverse=True)
                    next = next_candidate[0][0]
                else:
                    pass
            # 计算CH指数和本轮聚类效果
            gridsearch_cluster_vec[epoch] = processing_cluster_vec # 存储本轮聚类后的效果
            #within_idx  = sum(processing_cluster_sse.values())/(len(self.total_mids)-n_cluster)
            #samples_avg = np.sum(input_feat_svd,axis=0)
            #between_idx = 0
            #for idx, vec in processing_cluster_vec.items():
            #    center = np.mean(vec,axis=0)
            #    between_idx += center.shape[0]*np.sum(np.square(center - samples_avg))
            #between_idx /= (n_cluster-1)
            #CH_idx = between_idx / within_idx
            #gridsearch_CH[epoch] = CH_idx
            gridsearch_CH[epoch] = 1

        # 初步聚类完成，形成新的事件列表
        best_epoch = max(gridsearch_CH, key=gridsearch_CH.get)
        best_cluster_vec = gridsearch_cluster_vec[best_epoch]
        self.cluster_post = []
        event_counter = 1
        for idx, vec in best_cluster_vec.items():
            if vec.shape[0]==0:continue
            p = {}
            p['event'] = event_counter
            p['event_id'] = str(self.uid) +"-"+str(event_counter)
            event_counter += 1
            posts = []
            mids = []
            for j in range(vec.shape[0]):
                pos = vecs.index(list(vec[j])) # 用向量寻找第几个文档
                posts.append(text_vec_map[pos][0])
                mids.append(total_mids[pos])
            old_len = len(posts)
            old_mids = mids
            posts = list(set(posts))
            mids = list(set(mids))
            p['posts'] = posts
            p['mids'] = mids
            # TODO: assert有问题，聚类后对应文档的查找这里有问题
            #assert len(posts) == len(mids)
            p['volume'] = len(p['posts'])
            p['merge_keywords'] = extract_tags("".join(posts),topK=30) # 做簇融合后处理时需要的字段
            p['time'] = int(self.posts[self.posts['mid'].isin(mids)]['pt'].mean())
            self.cluster_post.append(p)

        # 后处理： 对两个具有40%以上相同的关键词的簇合并，最多合并次数为merge_iter控制
        '''
        for _ in range(self.merge_iter):
            cluster_kw = {}
            for p in self.cluster_post:
                cluster_kw[p['event']] = p['merge_keywords']
            merge_i = []
            merge_j = []
            for i in range(len(cluster_kw.values())-1):
                for j in range(i+1,len(cluster_kw.values())):
                    if i in merge_i+merge_j or j in merge_i+merge_j: continue
                    sim = len(set(list(cluster_kw.values())[i]).intersection(set(list(cluster_kw.values())[j])))/len(list(cluster_kw.values())[i])
                    if sim>=0.4: # 经验系数
                        merge_i.append(i)
                        merge_j.append(j)
            merge_pairs = zip(merge_i, merge_j)
            if len([x for x in zip(merge_i, merge_j)])==0: break
            new_clusters = []
            for i,j in merge_pairs:
                p = {}
                p['event'] = min(i,j)
                p['event_id'] = self.uid+'-'+str(min(i,j))
                p['posts'] = self.cluster_post[i]['posts'] + self.cluster_post[j]['posts']
                p['mids'] = self.cluster_post[i]['mids'] + self.cluster_post[j]['mids']
                p['volume'] = self.cluster_post[i]['volume'] + self.cluster_post[j]['volume']
                p['time'] = self.cluster_post[i]['time'] + self.cluster_post[j]['time']
                p['merge_keywords'] = self.cluster_post[i]['merge_keywords'] + self.cluster_post[j]['merge_keywords']
                new_clusters.append(p)
            for idx in sorted(merge_i+merge_j,reverse=True):  # 倒序删除
                del self.cluster_post[idx]
            self.cluster_post += new_clusters
        '''
        return 0

    def event_base(self):
        '''对聚类得到的事件进一步增加分析内容'''

        # 统计hashtag，keywords在每个簇中的出现情况
        hashtag_in_cluster = defaultdict(int)
        keyword_in_cluster = defaultdict(int)

        for event in self.cluster_post:
            # 得到高质量的topK贴文
            top_lang = "en"
            try:
                langauage = detect_langs("".join(event['posts']))
                for l in langauage:
                    top_lang = l.lang
                    break
            except:
                pass
            if top_lang == "en":
                ht = HarvestText(language="en")
            else:
                ht = HarvestText()

            original_posts = self.posts[self.posts['mid'].isin(event['mids'])]['cont'].tolist()
            try:
                highquality_posts = ht.get_summary(event['posts'], topK=min(30, event['volume']),avoid_repeat=True)  # list of sentences
            except:
                highquality_posts = event['posts']
            hashtags_dict = extractHashtag(original_posts)
            event['wordcloud'], event['keywords'], event['lang'] = get_key_words(highquality_posts, self.total_postags)
            for tag in hashtags_dict.keys():
                hashtag_in_cluster[tag] += 1
            for word in event['keywords']:
                keyword_in_cluster[word] += 1
            event.update({"hashtags": hashtags_dict})
        # 计算出现在不同簇多次的tag
        drop_tags = []
        for tag, freq in sorted(hashtag_in_cluster.items(), key=lambda x: x[1], reverse=True):
            if freq >= 3 and len(drop_tags) < min(15, int(len(hashtag_in_cluster) / 5)):  # 根据hashtag总数自适应drop number
                drop_tags.append(tag)
        # 计算出现在不同簇多次的关键词（组）
        drop_keywords = []
        for word, freq in sorted(keyword_in_cluster.items(), key=lambda x: x[1], reverse=True):
            if freq >= 3 and len(drop_keywords) < min(15,
                                                      int(len(keyword_in_cluster) / 5)):  # 根据hashtag总数自适应drop number
                drop_keywords.append(word)

        for event in self.cluster_post:
            # 过滤hashtag，keywords,整合为最终的hashtag(list of dicts), keywords(Array)字段,保证hashtag和关键词与事件强相关
            for tag in drop_tags:
                if tag in event['hashtags']: event['hashtags'].pop(tag)  # 更新每个簇的hashtag
            final_hashtags = []
            for k,v in event['hashtags'].items():
                if len(final_hashtags)==10: break
                final_hashtags.append({"name":k,"value":v })
            event['hashtags'] = final_hashtags
            for word in drop_keywords:
                if word in event['keywords']: event['keywords'].remove(word)  # 更新每个簇的keywords
            if len(event['hashtags']) != 0:
                event.update({"keywords": list(set(event['keywords']))})

            # 找到阅读量最高的topK贴文
            mids = event['mids']
            records = self.posts[self.posts['mid'].isin(mids)]
            records.sort_values("nfwd", ascending=False, inplace=True)
            highquality_posts = records[:10]
            top_posts = []
            for _, row in highquality_posts.iterrows():
                top_posts.append(
                    {"cont": row['cont'], "read": row['nfwd'], "pt": row['pt'], "uname": self.uid2uname[row["uid"]],
                     "sname": self.uid2sname[row['uid']], "photo": f"twitter.com/{self.uid2sname[row['uid']]}/photo"})
            event.update({"top_posts": top_posts})

            # 初始化文本分析工具类
            if event['lang'] == "en":
                ht = HarvestText(language="en")
            else:
                ht = HarvestText()
            # 确定标题
            sentences = []
            for post in event['top_posts']:
                sentences.append(post['cont'])
            if len(event['top_posts']) > 1:  # 只有一个句子没法用textrank
                # 确定最匹配簇主题的贴文作为标题
                best_sent = list(ht.get_summary(sentences, topK=min(3, len(sentences)), avoid_repeat=True, with_importance=True))[0][0]
            else:
                best_sent = sentences[0]
            best_title = re.findall(r"【(.*?)】", best_sent)
            if len(best_title) > 0:
                event.update({"title": best_title[0]})
            else:
                best_sent = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', best_sent)  # 清洗
                stop1 = best_sent.find("。")
                stop2 = best_sent.find(".")
                if stop1 != -1 or stop2 != -1:
                    stop = min(max(stop1, stop2), 20)
                else:
                    stop = 20
                event.update({"title": best_sent[:stop]})

            # 情感分析
            sent_type = {"正面": 0, "中性": 0, "负面": 0}
            try:
                if event['lang'] == "en":
                    from harvesttext.resources import get_english_senti_lexicon
                    sent_lexicon = get_english_senti_lexicon()
                    sent_dict = ht.build_sent_dict(event['posts'], pos_seeds=sent_lexicon["pos"],
                                                   neg_seeds=sent_lexicon["neg"], scale="+-1")
                else:
                    sent_dict = ht.build_sent_dict(event['posts'], min_times=1, scale="+-1")
                for p in event['posts']:
                    if ht.analyse_sent(p) > 0:
                        sent_type['正面'] += 1
                    elif ht.analyse_sent(p) == 0:
                        sent_type['中性'] += 1
                    else:
                        sent_type['负面'] += 1
                event.update({"sentiment": sorted(sent_type.items(), key=lambda x: x[1], reverse=True)[0][0]})
                event.update({"sentiment_score": sent_type})
            except Exception as err:  # 文本中不包含情感种子
                event.update({"sentiment": "中性"})
                event.update({"sentiment_score": sent_type})

            # 地名实体识别,媒体关注的地域
            location = defaultdict(int)
            for p in event['top_posts']:
                if event['lang'] in ["zh-cn", 'zh-tw', 'zh-yue']:
                    res = ht.named_entity_recognition(p['cont'])
                    for term, ner in res.items():
                        if ner == "地名": location[term] += 1
                elif event['lang'] == "en":
                    tokens = word_tokenize(p['cont'])
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
            event.update({"top_focus_loc": top_location})

    def generate_event(self):
        status = self.event_cluster()
        # 如果没有出错
        if status==0:
            self.event_base()

    def run(self):
        """
        只要有一个功能出错，这个账号的信息就不会存储到数据库中
        :return:
        """
        self.dataFetchAndLoad()
        print("data access done !!!")
        if self.skip: return 1
        # 每个功能都可能带来skip
        self.generate_event()
        print("event profiling done!!!")
        return 0






if __name__ == '__main__':
    model = MediaProfileModel(uid="11113", svd_dim=100, kmeans_iter=20, merge_iter=5)
    model.run()
    print(model.cluster_post)


