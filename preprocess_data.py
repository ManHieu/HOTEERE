from data_modules.preprocessor import load


datasets = ['causal_mulerx_en', 'causal_mulerx_da', 'causal_mulerx_es', 'causal_mulerx_tr', 'causal_mulerx_ur', 
            'subev_mulerx_en', 'subev_mulerx_da', 'subev_mulerx_es', 'subev_mulerx_tr', 'subev_mulerx_ur']
for dataset in datasets:
    load(dataset=dataset)