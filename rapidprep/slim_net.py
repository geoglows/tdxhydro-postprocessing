import json
import logging
import os
from itertools import chain
from multiprocessing import Pool

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def slim_streams_df(save_dir: str,
                    streams_df: pd.DataFrame = None,
                    id_field: str = 'LINKNO', ) -> pd.DataFrame:
    if streams_df is None:
        streams_df = pd.read_parquet(os.path.join(save_dir, 'rapid_inputs_master.parquet'), engine='fastparquet')

    # logger.info('\tDropping pruned shoots')
    # with open(os.path.join(save_dir, 'mod_prune_shoots.json'), 'r') as f:
    #     pruned_shoots = json.load(f)
    #     pruned_shoots = set([ids[-1] for _, ids in pruned_shoots.items()])
    #     streams_df = streams_df.loc[~streams_df[id_field].isin(pruned_shoots)]

    logger.info('\tDropping small trees')
    small_trees = pd.read_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv')).values.flatten()
    streams_df = streams_df.loc[~streams_df[id_field].isin(small_trees)].reset_index(drop=True)

    # # todo
    # with Pool(n_processes) as p:
    #     # Apply corrections based on the stream modification files
    #     logger.info('\tMerging head water stream segment rows')
    #     with open(os.path.join(save_dir, 'mod_dissolve_headwaters.json'), 'r') as f:
    #         diss_headwaters = json.load(f)
    #         all_merged_headwater = set(chain.from_iterable(
    #             [diss_headwaters[rivid] for rivid in diss_headwaters.keys()]))
    #     corrected_headwater_rows = p.starmap(
    #         _combine_routing_rows,
    #         [[streams_df, id_to_keep, ids_to_merge] for id_to_keep, ids_to_merge in diss_headwaters.items()]
    #     )
    #     logger.info('\tApplying corrected rows to gdf')
    #     streams_df = pd.concat([
    #         streams_df.loc[~streams_df[id_field].isin(all_merged_headwater)],
    #         *corrected_headwater_rows
    #     ])

    return streams_df


def slim_weight_table(save_dir: str, weight_table_path: str) -> None:
    wt_df = pd.read_csv(weight_table_path)
    logger.info('\tDropping small trees')
    small_trees = pd.read_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv')).values.flatten()
    wt_df = wt_df.dropna()
    wt_df['streamID'] = wt_df['streamID'].astype(int)
    wt_df = wt_df[~wt_df['streamID'].isin(small_trees)].reset_index(drop=True)
    wt_df.to_csv(weight_table_path.replace('_full.csv', '.csv'), index=False)
    return
