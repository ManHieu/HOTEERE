
from typing import List
from sklearn.metrics import classification_report


def padding(sent, max_sent_len = 194, pad_tok=0):
    one_list = [pad_tok] * max_sent_len # none id 
    one_list[0:len(sent)] = sent
    return one_list


def compute_f1(predicted_seqs: List[str], gold_seqs: List[str], task: str):
        if task == 'ECI':
            n_predict = 0
            n_gold = 0
            tp = 0
            wrong_struct = 0
            for predict, gold in zip(predicted_seqs, gold_seqs):
                if predict.startswith('No')==False and gold.startswith('No')==False:
                    tp = tp + 1
                if predict.startswith('No')==False:
                    n_predict = n_predict + 1
                if gold.startswith('No')==False:
                    n_gold = n_gold + 1
                if predict.startswith('Yes')==False and predict.startswith('No')==False:
                    wrong_struct = wrong_struct + 1

            if wrong_struct == len(predicted_seqs):
                return 0.0, 0.0, 0.0, (0, 0, 0)
            elif n_predict==n_gold==0:
                return 0.1, 0.1, 0.1, (0.1, 0.1, 0.1)
            else:
                p = (tp + 1)/(n_predict + 1)
                r = (tp + 1)/(n_gold + 1)
                f1 = 2 * p * r / (p + r + 1e-9)
                return p, r, f1, (p, r, f1)
        
        elif task == 'TRE':
            label_idx={
            "before": 0, 
            "after": 1, 
            "same time": 2, 
            "no relation": 3
            }
        elif task == 'SRE':
            label_idx={
            "including": 0, 
            "part of": 1, 
            "coreference": 2, 
            "no relation": 3
            }
        n_predict = 0
        n_gold = 0
        tp = 0
        preds = []
        golds = []
        wrong_struct = 0
        
        for predict, gold in zip(predicted_seqs, gold_seqs):
            for key, item in label_idx.items():
                if predict.startswith(key):
                    preds.append(item)
                    if key not in ['no relation', 'coreference']:
                        n_predict = n_predict + 1

                if gold.startswith(key):
                    golds.append(item)
                    if key not in ['no relation', 'coreference']:
                        n_gold = n_gold + 1

                if predict.startswith(key) and gold.startswith(key) and key not in ['no relation', 'coreference']:
                    tp = tp + 1

            if not any([predict.startswith(k) for k in list(label_idx.keys())]):
                wrong_struct = wrong_struct + 1
                preds.append(len(list(label_idx.keys())))
        
        label_names = list(label_idx.keys()).append('wrongs structure')
        report = classification_report(golds, preds, target_names=label_names)
        # print(report)

        if wrong_struct == len(predicted_seqs):
            return 0.0, 0.0, 0.0, report
        elif n_predict==n_gold==0:
            return 0.1, 0.1, 0.1, report
        else:
            p = (tp + 1)/(n_predict + 1)
            r = (tp + 1)/(n_gold + 1)
            f1 = 2 * p * r / (p + r + 1e-9)
            return p, r, f1, report
    
