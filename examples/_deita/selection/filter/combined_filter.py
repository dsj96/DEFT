import torch
import logging
import numpy as np
from deita.pipeline.utils import sort_key_split
from deita.selection.filter.base import IterativeFilter
import pdb
logger = logging.getLogger(__name__)

def min_max_normalize(array_list, min_value, max_value):

    range_min_max = max_value - min_value


    normalized_list = (array_list - min_value) / range_min_max

    return normalized_list

def apply_decay(data_array, decay_rate=0.5):
    length = len(data_array)
    decay_factor = np.exp(-decay_rate * np.arange(length))
    decayed_array = data_array * decay_factor
    return decayed_array

class Combined_Filter(IterativeFilter):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

    def _normalize(self, df_data):
        """
        df=df ['embedding', 'idx', 'complexity_scores', 'quality_scores']
        """
        df_data['complexity_scores_max'] = df_data["complexity_scores"].apply(np.array).apply(lambda x: x.max())
        df_data['complexity_scores_min'] = df_data["complexity_scores"].apply(np.array).apply(lambda x: x.min())
        df_data['quality_scores_max'] = df_data["quality_scores"].apply(np.array).apply(lambda x: x.max())
        df_data['quality_scores_min'] = df_data["quality_scores"].apply(np.array).apply(lambda x: x.min())
        df_data['knowledge5_scores_max'] = df_data["knowledge5_scores"].apply(np.array).apply(lambda x: x.max())
        df_data['knowledge5_scores_min'] = df_data["knowledge5_scores"].apply(np.array).apply(lambda x: x.min())
        df_data['knowledge10_scores_max'] = df_data["knowledge10_scores"].apply(np.array).apply(lambda x: x.max())
        df_data['knowledge10_scores_min'] = df_data["knowledge10_scores"].apply(np.array).apply(lambda x: x.min())
        df_data['knowledge20_scores_max'] = df_data["knowledge20_scores"].apply(np.array).apply(lambda x: x.max())
        df_data['knowledge20_scores_min'] = df_data["knowledge20_scores"].apply(np.array).apply(lambda x: x.min())
        df_data['knowledge30_scores_max'] = df_data["knowledge30_scores"].apply(np.array).apply(lambda x: x.max())
        df_data['knowledge30_scores_min'] = df_data["knowledge30_scores"].apply(np.array).apply(lambda x: x.min())
        df_data['knowledge40_scores_max'] = df_data["knowledge40_scores"].apply(np.array).apply(lambda x: x.max())
        df_data['knowledge40_scores_min'] = df_data["knowledge40_scores"].apply(np.array).apply(lambda x: x.min())
        df_data['knowledge50_scores_max'] = df_data["knowledge50_scores"].apply(np.array).apply(lambda x: x.max())
        df_data['knowledge50_scores_min'] = df_data["knowledge50_scores"].apply(np.array).apply(lambda x: x.min())
        df_data['knowledge60_scores_max'] = df_data["knowledge60_scores"].apply(np.array).apply(lambda x: x.max())
        df_data['knowledge60_scores_min'] = df_data["knowledge60_scores"].apply(np.array).apply(lambda x: x.min())
        df_data['loss_max'] = df_data["loss"].apply(np.array).apply(lambda x: x.max())
        df_data['loss_min'] = df_data["loss"].apply(np.array).apply(lambda x: x.min())
        df_data['ppl_max'] = df_data["ppl"].apply(np.array).apply(lambda x: x.max())
        df_data['ppl_min'] = df_data["ppl"].apply(np.array).apply(lambda x: x.min())

        complexity_scores_max = max(df_data['complexity_scores_max'])
        complexity_scores_min = min(df_data['complexity_scores_min'])
        quality_scores_max = max(df_data['quality_scores_max'])
        quality_scores_min = min(df_data['quality_scores_min'])
        knowledge5_scores_max = max(df_data['knowledge5_scores_max'])
        knowledge5_scores_min = min(df_data['knowledge5_scores_min'])
        knowledge10_scores_max = max(df_data['knowledge10_scores_max'])
        knowledge10_scores_min = min(df_data['knowledge10_scores_min'])
        knowledge20_scores_max = max(df_data['knowledge20_scores_max'])
        knowledge20_scores_min = min(df_data['knowledge20_scores_min'])
        knowledge30_scores_max = max(df_data['knowledge30_scores_max'])
        knowledge30_scores_min = min(df_data['knowledge30_scores_min'])
        knowledge40_scores_max = max(df_data['knowledge40_scores_max'])
        knowledge40_scores_min = min(df_data['knowledge40_scores_min'])
        knowledge50_scores_max = max(df_data['knowledge50_scores_max'])
        knowledge50_scores_min = min(df_data['knowledge50_scores_min'])
        knowledge60_scores_max = max(df_data['knowledge60_scores_max'])
        knowledge60_scores_min = min(df_data['knowledge60_scores_min'])
        loss_max = max(df_data['loss_max'])
        loss_min = min(df_data['loss_min'])
        ppl_max = max(df_data['ppl_max'])
        ppl_min = min(df_data['ppl_min'])

        df_data = df_data.drop(columns=['complexity_scores_max', 'complexity_scores_min', 'quality_scores_max', 'quality_scores_min',\
                                        'knowledge5_scores_max', 'knowledge5_scores_min', 'knowledge10_scores_max', 'knowledge10_scores_min',\
                                        'knowledge20_scores_max', 'knowledge20_scores_min', 'knowledge30_scores_max', 'knowledge30_scores_min',\
                                        'knowledge40_scores_max', 'knowledge40_scores_min', 'knowledge50_scores_max', 'knowledge50_scores_min',\
                                        'knowledge60_scores_max', 'knowledge60_scores_min', 'loss_max', 'loss_min', 'ppl_max', 'ppl_min'\
                                            ])
        

        df_data["complexity_scores_normalized"] = df_data["complexity_scores"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=complexity_scores_min, max_value=complexity_scores_max))
        df_data["quality_scores_normalized"] = df_data["quality_scores"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=quality_scores_min, max_value=quality_scores_max))
        df_data["knowledge5_scores_normalized"] = df_data["knowledge5_scores"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=knowledge5_scores_min, max_value=knowledge5_scores_max))
        df_data["knowledge10_scores_normalized"] = df_data["knowledge10_scores"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=knowledge10_scores_min, max_value=knowledge10_scores_max))
        df_data["knowledge20_scores_normalized"] = df_data["knowledge20_scores"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=knowledge20_scores_min, max_value=knowledge20_scores_max))
        df_data["knowledge30_scores_normalized"] = df_data["knowledge30_scores"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=knowledge30_scores_min, max_value=knowledge30_scores_max))
        df_data["knowledge40_scores_normalized"] = df_data["knowledge40_scores"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=knowledge40_scores_min, max_value=knowledge40_scores_max))
        df_data["knowledge50_scores_normalized"] = df_data["knowledge50_scores"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=knowledge50_scores_min, max_value=knowledge50_scores_max))
        df_data["knowledge60_scores_normalized"] = df_data["knowledge60_scores"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=knowledge60_scores_min, max_value=knowledge60_scores_max))
        df_data["loss_normalized"] = df_data["loss"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=loss_min, max_value=loss_max))
        df_data["ppl_normalized"] = df_data["ppl"].apply(np.array).apply(lambda x: min_max_normalize(x, min_value=ppl_min, max_value=ppl_max))



        df_data = df_data.drop(columns=['complexity_scores', 'quality_scores', 'knowledge5_scores', 'knowledge10_scores', 'knowledge20_scores', 'knowledge30_scores',\
                                        'knowledge40_scores', 'knowledge50_scores', 'knowledge60_scores', 'loss', 'ppl'])
        columns_mapping = {'complexity_scores_normalized': 'complexity_scores', 'quality_scores_normalized': 'quality_scores',\
                        'knowledge5_scores_normalized' : 'knowledge5_scores', 'knowledge10_scores_normalized' : 'knowledge10_scores',\
                        'knowledge20_scores_normalized' : 'knowledge20_scores', 'knowledge30_scores_normalized' : 'knowledge30_scores',\
                        'knowledge40_scores_normalized' : 'knowledge40_scores', 'knowledge50_scores_normalized' : 'knowledge50_scores',\
                        'knowledge60_scores_normalized' : 'knowledge60_scores', "loss_normalized":"loss", "ppl_normalized":"ppl"}
        df_data = df_data.rename(columns=columns_mapping)
        return df_data

    def _sort(self, df):
        
        """
            Sort dataframe by given method
        """
        if isinstance(self.sort_key, list):
            all_sort_keys = self.sort_key
        else:
            all_sort_keys = sort_key_split(self.sort_key) # self.sort_key='complexity_scores,quality_scores,knowledge20_scores'
        # logger.info("Compute final score for each sample, consider {}".format("+".join(all_sort_keys)))
        for sk in all_sort_keys:
            df[sk] = df[sk].apply(np.array)
        df["final_score"] =  df["complexity_scores"] * df["quality_scores"]  + df["quality_scores"]*df[all_sort_keys[2]]

        # df["final_score"] = df["complexity_scores"] + df["quality_scores"] + df[all_sort_keys[2]]

        df["final_score"] = df["final_score"].apply(lambda x: apply_decay(x, decay_rate=self.decay_rate)).apply(lambda x: x.sum())
        df_sorted = df.sort_values(by = "final_score", ascending = False)
        
        return df_sorted
        

        # """        
        #     Sort dataframe by given method
        # """

        # if isinstance(self.sort_key, list):
        #     all_sort_keys = self.sort_key
        # else:
        #     all_sort_keys = sort_key_split(self.sort_key)
            
        # logger.info("Compute final score for each sample, consider {}".format("+".join(all_sort_keys))) # complexity_scores+quality_scores
        # for sk in all_sort_keys:
        #     df[sk] = df[sk].apply(np.array)
        
        # df["final_score"] = df[all_sort_keys[0]]
        # # pdb.set_trace()
        # for i in range(1, len(all_sort_keys)):
        #     df["final_score"] = df["final_score"] * df[all_sort_keys[i]]
            
        # df["final_score"] = df["final_score"].apply(lambda x: x.sum())
        # df_sorted = df.sort_values(by = "final_score", ascending = False)
        
        # return df_sorted