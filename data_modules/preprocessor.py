import argparse
from collections import defaultdict
import pickle
import json
import os
from data_modules.retriever import AtomicRetriever, ConceptNetRetriever
from data_modules.utils import sentence_encode
from sklearn.model_selection import KFold, train_test_split
import tqdm
from data_modules.datapoint_formats import get_datapoint
from data_modules.readers import cat_xml_reader, ctb_cat_reader, tml_reader, tsvx_reader
import random
import numpy as np


class Preprocessor(object):
    def __init__(self, dataset, datapoint, intra=True, inter=False):
        self.dataset = dataset
        self.intra = intra
        self.inter = inter
        self.datapoint = datapoint
        self.register_reader(self.dataset)
        self.atomic_retriever = AtomicRetriever('datasets/atomic2020_data-feb2021/atomic2020_data-feb2021/')
        self.conceptnet_retriever = ConceptNetRetriever()

    def register_reader(self, dataset):
        if self.dataset == 'HiEve' or self.dataset == 'IC':
            self.reader = tsvx_reader
        elif dataset == 'ESL':
            self.reader = cat_xml_reader
        elif 'MATRES' in dataset:
            self.reader = tml_reader
        elif dataset == 'Causal-TB':
            self.reader = ctb_cat_reader
        else:
            raise ValueError("We have not supported this dataset {} yet!".format(self.dataset))

    def retrieve_knowledge_sentences(self, my_dict):
        for eid, ev in tqdm.tqdm(my_dict['event_dict'].items()):
            sid = ev['sent_id']
            knowledge_sentences = []
            sent = my_dict['sentences'][sid]['tokens']
            knowledge_sentences = knowledge_sentences +  [sent[0] for sent in self.conceptnet_retriever.retrieve_from_conceptnet(sent, ev['token_id'], top_k=5)]
            knowledge_sentences = knowledge_sentences +  [sent[0] for sent in self.atomic_retriever.retrive_from_atomic(sent, ev['token_id'], top_k=3)]
            knowledge_sentences = list(set(knowledge_sentences))
            knowledge_sentences_emb = sentence_encode(knowledge_sentences)
            my_dict['event_dict'][eid]['knowledge_sentences'] = {kg_sent:knowledge_sentences_emb[i] for i, kg_sent in enumerate(knowledge_sentences)}
        print(f"my_dict after add knowledge sentences: {my_dict}")
        return my_dict

    def load_dataset(self, dir_name):
        corpus = []
        if self.dataset == 'ESL':
            topic_folders = [t for t in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, t))]
            for topic in tqdm.tqdm(topic_folders):
                topic_folder = os.path.join(dir_name, topic)
                onlyfiles = [f for f in os.listdir(topic_folder) if os.path.isfile(os.path.join(topic_folder, f))]
                for file_name in onlyfiles:
                    file_name = os.path.join(topic, file_name)
                    if file_name.endswith('.xml'):
                        cache_dir = dir_name+f'hoteer_{self.inter}_{self.intra}'
                        try:
                            os.mkdir(dir_name+f'hoteer_{self.inter}_{self.intra}')
                        except FileExistsError:
                            pass
                        cache_file = cache_dir + f'/{file_name}.pkl'
                        if os.path.exists(cache_file):
                            with open(cache_file, 'rb') as f:
                                my_dict = pickle.load(f)
                                corpus.append(my_dict)
                        else:
                            my_dict = self.reader(dir_name, file_name, inter=self.inter, intra=self.intra)
                            if my_dict != None:
                                doc_presentation = sentence_encode([sent['content'] for sent in my_dict['sentences']])
                                my_dict['doc_presentation'] = doc_presentation
                                my_dict = self.retrieve_knowledge_sentences(my_dict)
                                with open(cache_file, 'wb') as f:
                                    pickle.dump(my_dict, f, pickle.HIGHEST_PROTOCOL)
                                corpus.append(my_dict)
                            # print(my_dict)
        else:
            onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
            i = 0
            for file_name in tqdm.tqdm(onlyfiles):
                # if i == 11:
                #     break
                # i = i + 1
                cache_dir = dir_name+f'hoteer_{self.inter}_{self.intra}'
                try:
                    os.mkdir(dir_name+f'hoteer_{self.inter}_{self.intra}')
                except FileExistsError:
                    pass
                cache_file = cache_dir + f'/{file_name}.pkl'
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        my_dict = pickle.load(f)
                        corpus.append(my_dict)
                else:
                    my_dict = self.reader(dir_name, file_name)
                    if my_dict != None:
                        doc_presentation = sentence_encode([sent['content'] for sent in my_dict['sentences']])
                        my_dict['doc_presentation'] = doc_presentation
                        my_dict = self.retrieve_knowledge_sentences(my_dict)
                        with open(cache_file, 'wb') as f:
                            pickle.dump(my_dict, f, pickle.HIGHEST_PROTOCOL)
                        corpus.append(my_dict)
        
        return corpus
    
    def process_and_save(self, corpus, save_path=None, save_cache=False):
        if type(corpus) == list:
            processed_corpus = []
            for my_dict in tqdm.tqdm(corpus):
                len_doc = sum([len(sentence['tokens']) for sentence in my_dict['sentences']])
                doc_info = True
                processed_corpus.extend(get_datapoint(self.datapoint, my_dict))
            if save_path != None and save_cache == True:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_corpus, f, indent=6)
        else:
            processed_corpus = defaultdict(list)
            for key, topic in corpus.items():
                for my_dict in tqdm.tqdm(topic):
                    len_doc = sum([len(sentence['tokens']) for sentence in my_dict['sentences']])
                    doc_info = True
                    processed_corpus[key].extend(get_datapoint(self.datapoint, my_dict))
            if save_path != None and save_cache == True:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_corpus, f, indent=6)

        return processed_corpus


def load(dataset: str, save_cache=False):
    seed = 7890
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    if dataset == 'HiEve':
        datapoint = 'hoteere_data_point'
        corpus_dir = 'datasets/hievents_v2/processed/'
        processor = Preprocessor(dataset, datapoint)
        corpus = processor.load_dataset(corpus_dir)
        corpus = list(sorted(corpus, key=lambda x: x['doc_id']))
        train, test = train_test_split(corpus, train_size=100.0/120, test_size=20.0/120, random_state=seed)
        train, validate = train_test_split(train, train_size=80.0/100., test_size=20.0/100, random_state=seed)

        processed_path = 'datasets/hievents_v2/train.json'
        train = processor.process_and_save(train, processed_path, save_cache)

        processed_path = 'datasets/hievents_v2/val.json'
        val = processor.process_and_save(validate, processed_path, save_cache)

        processed_path = 'datasets/hievents_v2/test.json'
        test = processor.process_and_save(test, processed_path, save_cache)
        return {0: [train, val, test]}
    
    if dataset == 'IC':
        datapoint = 'hoteere_data_point'
        corpus_dir = 'datasets/IC/IC_Processed/'
        onlyfiles = [f for f in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, f)) and f[-4:] == "tsvx"]
        onlyfiles.sort()
        train_doc_id = [f.replace(".tsvx", "") for f in onlyfiles[0:60]]
        val_doc_id = [f.replace(".tsvx", "") for f in onlyfiles[60:80]]
        test_doc_id = [f.replace(".tsvx", "") for f in onlyfiles[80:]]
        processor = Preprocessor(dataset, datapoint)
        corpus = processor.load_dataset(corpus_dir)
        corpus = list(sorted(corpus, key=lambda x: x['doc_id']))
        train = [doc for doc in corpus if doc['doc_id'] in train_doc_id]
        test = [doc for doc in corpus if doc['doc_id'] in test_doc_id]
        validate = [doc for doc in corpus if doc['doc_id'] in val_doc_id]

        processed_path = 'datasets/IC/train.json'
        train = processor.process_and_save(train, processed_path, save_cache)

        processed_path = 'datasets/IC/val.json'
        val = processor.process_and_save(validate, processed_path, save_cache)

        processed_path = 'datasets/IC/test.json'
        test = processor.process_and_save(test, processed_path, save_cache)
        return {0: [train, val, test]}
   
    if dataset == 'ESL':
        datapoint = 'hoteere_data_point'
        kfold = KFold(n_splits=5)
        processor = Preprocessor(dataset, datapoint, intra=True, inter=False)
        corpus_dir = './datasets/EventStoryLine/annotated_data/v0.9/'
        corpus = processor.load_dataset(corpus_dir)

        _train, test = [], []
        data = defaultdict(list)
        for my_dict in corpus:
            topic = my_dict['doc_id'].split('/')[0]
            data[topic].append(my_dict)

            if '37/' in my_dict['doc_id'] or '41/' in my_dict['doc_id']:
                test.append(my_dict)
            else:
                _train.append(my_dict)

        # print()
        # processed_path = f"./datasets/EventStoryLine/intra_data.json"
        # processed_data = processor.process_and_save(processed_path, data)

        random.shuffle(_train)
        folds = {}
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(_train)):
            try:
                os.mkdir(f"./datasets/EventStoryLine/{fold}")
            except FileExistsError:
                pass

            train = [_train[id] for id in train_ids]
            # print(train[0])
            validate = [_train[id] for id in valid_ids]
        
            processed_path = f"./datasets/EventStoryLine/{fold}/train.json"
            train = processor.process_and_save(train, processed_path, save_cache)

            processed_path = f"./datasets/EventStoryLine/{fold}/test.json"
            val = processor.process_and_save(validate, processed_path, save_cache)
            
            processed_path = f"./datasets/EventStoryLine/{fold}/val.json"
            test = processor.process_and_save(test, processed_path, save_cache)
            folds[fold] = [train, val, test]
        return folds
    
    if dataset == 'Causal-TB':
        datapoint = 'hoteere_data_point'
        kfold = KFold(n_splits=10)
        processor = Preprocessor(dataset, datapoint)
        corpus_dir = './datasets/Causal-TimeBank/Causal-TimeBank-CAT/'
        corpus = processor.load_dataset(corpus_dir)

        random.shuffle(corpus)
        folds = {}
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(corpus)):
            try:
                os.mkdir(f"./datasets/EventStoryLine/{fold}")
            except FileExistsError:
                pass

            train = [corpus[id] for id in train_ids]
            # print(train[0])
            validate = [corpus[id] for id in valid_ids]
        
            processed_path = f"./datasets/EventStoryLine/{fold}/train.json"
            train = processor.process_and_save(train, processed_path, save_cache)

            processed_path = f"./datasets/EventStoryLine/{fold}/test.json"
            val = processor.process_and_save(validate, processed_path, save_cache)
            
            folds[fold] = [train, val, val]
        return folds

    if dataset == 'MATRES':
        datapoint = 'hoteere_data_point'
        aquaint_dir_name = "./datasets/MATRES/TBAQ-cleaned/AQUAINT/"
        timebank_dir_name = "./datasets/MATRES/TBAQ-cleaned/TimeBank/"
        platinum_dir_name = "./datasets/MATRES/te3-platinum/"
        processor = Preprocessor(dataset, datapoint)
        validate = Preprocessor.load_dataset(aquaint_dir_name)
        train = Preprocessor.load_dataset(timebank_dir_name)
        test = Preprocessor.load_dataset(platinum_dir_name)

        processed_path = 'datasets/MATRES/train.json'
        train = processor.process_and_save(train, processed_path, save_cache)

        processed_path = 'datasets/MATRES/val.json'
        val = processor.process_and_save(validate, processed_path, save_cache)

        processed_path = 'datasets/MATRES/test.json'
        test = processor.process_and_save(test, processed_path, save_cache)
        return {0: [train, val, test]}

    print(f"We have not supported {dataset} dataset!")
    return None
    

if __name__=='__main__':
    dataset = 'ESL'
    load(dataset=dataset)
