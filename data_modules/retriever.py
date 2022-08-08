from collections import defaultdict
import re
import requests
from typing import List
from trankit import Pipeline
import networkx as nx
from sentence_transformers import SentenceTransformer, util



class AtomicRetriever(object):
    def __init__(self,
                kb_path) -> None:
        self.p = Pipeline('english', cache_dir='./trankit')
        self.sim_evaluator = SentenceTransformer('all-MiniLM-L12-v1')
        self.choosen_rel = ['CapableOf', 'Causes', 'CausesDesire', 'Desires', 'HasA', 'HasSubEvent',
                            'HinderedBy', 'InstanceOf', 'isAfter', 'isBefore', 'MotivatedByGoal', 'NotDesires',
                            'UsedFor', 'oEffect', 'ReceivesAction', 'PartOf', 'xEffect', 'xIntent,','xReason',]
        self.rel_to_text = {
            'Causes': 'causes', 
            'CausesDesire': 'makes someone want', 
            'Desires': 'desires', 
            'HasA': 'has', 
            'HasSubEvent': 'includes the event',
            'HinderedBy': 'can be hindered by', 
            'isAfter': 'happens after', 
            'isBefore': 'happens before', 
            'MotivatedByGoal': 'is a step towards accomplishing the goal', 
            'NotDesires': 'do not desire',
            'UsedFor': 'uses for', 
            'oEffect': ', as a result, Y or others will', 
            'ReceivesAction': 'can receive or be affected by the action', 
            'PartOf': 'is a part of', 
            'xEffect': ', as a result, PersonX will', 
            'xIntent': 'because PersonX wanted',
            'xReason': 'because', 
        }
        self.kb = self.load_kb(kb_path)
        self.event_to_concept = defaultdict(list)
        
    def load_kb(self, kb_path):
        train = kb_path + 'train.tsv'
        dev = kb_path + 'dev.tsv'
        test = kb_path + 'test.tsv'
        kb_path = [train, dev, test]
        kb = []
        for split_path in kb_path:
            for line in open(split_path, encoding='UTF-8'):
                triples = line.split('\t')
                if triples[1] in self.choosen_rel and 'none' not in triples[2]:
                    kb.append(triples)
        return kb

    def retrive_from_atomic(self, input_seq: List[str], trigger_token_id: List[int], top_k: int=3):
        parsed_tokens = self.p.posdep(input_seq, is_sent=True)['tokens']
        heads = [token['head'] for token in parsed_tokens]
        dep_tree = nx.DiGraph()
        for head, tail in zip(heads, list(range(len(input_seq) + 1))):
            if head != tail:
                dep_tree.add_edge(head, tail)
        
        k_hop_tree = []
        for idx in trigger_token_id:
            k_hop_tree.extend(list(nx.dfs_tree(dep_tree, idx+1, depth_limit=2).nodes()))
        k_hop_seq = [node - 1 for node in k_hop_tree if node > 0]
        k_hop_seq.sort()
        k_hop_seq = ' '.join([input_seq[idx] for idx in k_hop_seq])
        event_mention = [input_seq[idx] for idx in trigger_token_id]
        lemmatized_mention = [t['lemma'] for t in self.p.lemmatize(event_mention, is_sent=True)['tokens']]

        knowledge_sents = []
        for m, lemmatized_m in zip(event_mention, lemmatized_mention):
            if self.event_to_concept.get((m, lemmatized_m)) == None:
                for triple in self.kb:
                    if m in triple[0] or m in triple[2] or lemmatized_m in triple[0] or lemmatized_m in triple[2]:
                        self.event_to_concept[(m, lemmatized_m)].append(triple)
                        knowledge_sent = ' '.join([triple[0], self.rel_to_text[triple[1]], triple[2]])
                        embeddings1 = self.sim_evaluator.encode([knowledge_sent], convert_to_tensor=True)
                        embeddings2 = self.sim_evaluator.encode([k_hop_seq], convert_to_tensor=True)
                        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
                        score = float(cosine_scores[0][0])
                        knowledge_sents.append((knowledge_sent, score))
            else:
                for triple in self.event_to_concept.get((m, lemmatized_m)):
                    if m in triple[0] or m in triple[2] or lemmatized_m in triple[0] or lemmatized_m in triple[2]:
                        self.event_to_concept[(m, lemmatized_m)] = triple
                        knowledge_sent = ' '.join([triple[0], self.rel_to_text[triple[1]], triple[2]])
                        embeddings1 = self.sim_evaluator.encode([knowledge_sent], convert_to_tensor=True)
                        embeddings2 = self.sim_evaluator.encode([k_hop_seq], convert_to_tensor=True)
                        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
                        score = float(cosine_scores[0][0])
                        knowledge_sents.append((knowledge_sent, score))
        
        knowledge_sents.sort(key=lambda x: x[1], reverse=True)
        return knowledge_sents[0:top_k]


class ConceptNetRetriever(object):
    def __init__(self) -> None:
        self.sim_evaluator = SentenceTransformer('all-MiniLM-L12-v1')
        self.chosen_rel = ['CapableOf', 'IsA', 'Causes', 'MannerOf', 'CausesDesire', 'UsedFor', 'HasSubevent', 'HasPrerequisite', 'NotDesires', 'PartOf', 'HasA', 'Entails', 'ReceivesAction', 'UsedFor', 'CreatedBy', 'MadeOf', 'Desires']
        self.rel_to_text  = {
            'CapableOf': 'is capable of', 
            'IsA': 'is a', 
            'Causes': 'causes', 
            'MannerOf': 'is a specific way to do', 
            'CausesDesire': 'makes someone want', 
            'UsedFor': 'uses for', 
            'HasSubevent': 'includes the event', 
            'HasPrerequisite': 'has a precondition of', 
            'NotDesires': 'do not desire', 
            'Entails': 'entails', 
            'ReceivesAction': 'can receive or be affected by the action', 
            'CreatedBy': 'is created by', 
            'Desires': 'desires'
        }
        self.p = Pipeline('english', cache_dir='./trankit')
    
    def retrieve_from_conceptnet(self, input_seq: List[str], trigger_token_id: List[int], top_k: int=3):
        parsed_tokens = self.p.posdep(input_seq, is_sent=True)['tokens']
        heads = [token['head'] for token in parsed_tokens]
        dep_tree = nx.DiGraph()
        for head, tail in zip(heads, list(range(len(input_seq) + 1))):
            if head != tail:
                dep_tree.add_edge(head, tail)
        
        k_hop_tree = []
        for idx in trigger_token_id:
            k_hop_tree.extend(list(nx.dfs_tree(dep_tree, idx+1, depth_limit=2).nodes()))
        k_hop_seq = [node - 1 for node in k_hop_tree if node > 0]
        k_hop_seq.sort()
        k_hop_seq = ' '.join([input_seq[idx] for idx in k_hop_seq])
        event_mention = [input_seq[idx] for idx in trigger_token_id]
        lemmatized_mention = [t['lemma'] for t in self.p.lemmatize(event_mention, is_sent=True)['tokens']]
        knowledge_sents = []
        for event in ['_'.join(event_mention), '_'.join(lemmatized_mention)]:
            obj = requests.get('http://api.conceptnet.io/c/en/' + event).json()
            for e in obj['edges']:
                # print(e)
                if e['start']['language'] == 'en' and e['end']['language'] == 'en':
                    if e['rel']['label'] in self.chosen_rel:
                        if e['surfaceText'] != None:
                            knowledge_sent = re.sub(r'[\[\]]','', e['surfaceText'])
                        else:
                            knowledge_sent = ' '.join([e['start']['label'], self.rel_to_text[e['rel']['label']], e['end']['label']])
                        embeddings1 = self.sim_evaluator.encode([knowledge_sent], convert_to_tensor=True)
                        embeddings2 = self.sim_evaluator.encode([k_hop_seq], convert_to_tensor=True)
                        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
                        score = float(cosine_scores[0][0])
                        knowledge_sents.append((knowledge_sent, score))
        
        knowledge_sents.sort(key=lambda x: x[1], reverse=True)
        return knowledge_sents[0:top_k]

                        
                

if __name__ == '__main__':

    sent = ["A", "woman", "has", "been", "arrested", "in", "connection", "with", "the", "murder", "of", "Ciaran", "Noonan."]

    atomic_retriever = AtomicRetriever('datasets/atomic2020_data-feb2021/atomic2020_data-feb2021/')
    atomic_sents = atomic_retriever.retrive_from_atomic(sent, trigger_token_id=[4], top_k=3)

    conceptnet_retriever = ConceptNetRetriever()
    conceptnet_sets = conceptnet_retriever.retrieve_from_conceptnet(sent,trigger_token_id=[4], top_k=5)

    print(f'atomic sents: {atomic_sents}')
    print(f"conceptnet sents: {conceptnet_sets}")


