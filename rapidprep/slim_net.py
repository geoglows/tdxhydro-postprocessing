import json
import logging
import os
from itertools import chain
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .trace_streams import create_adjoint_json

__all__ = ['find_streams_to_slim', ]

logger = logging.getLogger(__name__)


def find_streams_to_slim(save_dir: str,
                         streams_df: pd.DataFrame = None,
                         id_field='LINKNO',
                         ds_id_field: str = 'DSLINKNO',
                         order_field: str = 'strmOrder', ) -> None:
    """
    Analyzes the connectivity of the streams to find errors and places that need to be dissolved

    Creates several files describing the network and how to modify it:
    - adjoint_tree.json: A json file that describes the network as an adjoint connections of stream
    - mod_zero_length.csv: A csv file that describes the stream segments that have a length of 0 in 3 cases
    - mod_drop_small_trees.csv: A csv list of all segments part of a drainage tree less than 75km2 at the outlet
    - mod_dissolve_headwaters.json: A json file that describes the headwaters streams to be dissolved together
    - mod_prune_shoots.json: A json file that describes the order 1 streams to be pruned from the drainage tree

    Args:
        streams_df (pd.DataFrame): streams master parquet file
        save_dir (str): Path to directory where dissolved network and catchments will be saved
        id_field (str, optional): Field name with to the unique id of each stream segment
        ds_id_field (str, optional): Field name with the unique downstream id of each stream segment
        order_field: Field in network file that corresponds to the strahler order of each stream segment

    Returns:
        None
    """
    logger.info('Looking for Places to Slim Network')
    if streams_df is None:
        streams_df = pd.read_parquet(os.path.join(save_dir, 'rapid_inputs_master.parquet'), engine='fastparquet')

    # trace network to create adjoint tree
    logger.info('\tTracing adjoint paths')
    adjoint_dict = create_adjoint_json(streams_df, id_field=id_field, ds_field=ds_id_field, order_field="strmOrder")
    with open(os.path.join(save_dir, 'adjoint_tree.json'), 'w') as f:
        json.dump(adjoint_dict, f)

    # Drop trees with small total length/area
    logger.info('\tFinding small trees')
    small_tree_outlet_ids = streams_df.loc[np.logical_and(
        streams_df[ds_id_field] == -1,
        streams_df['DSContArea'] < 100_000_000
    ), id_field].values
    small_tree_segments = set(chain.from_iterable([adjoint_dict[str(x)] for x in small_tree_outlet_ids]))
    (
        pd.DataFrame(small_tree_segments, columns=['drop'])
        .to_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv'), index=False)
    )

    # # Find headwater streams to be dissolved
    # adjoint_order_2_dict = create_adjoint_json(streams_df,
    #                                            id_field=id_field,
    #                                            ds_field=ds_id_field,
    #                                            order_field=order_field,
    #                                            order_filter=2)
    #
    # # list all ids that were merged, turn a list of lists into a flat list, remove duplicates by converting to a set
    # logger.info('\tFinding headwater streams to be dissolved')
    # top_order_2s = {str(value[-1]) for value in list(adjoint_order_2_dict.values())}
    # adjoint_order_2_dict = {key: adjoint_dict[key] for key in top_order_2s}
    # with open(os.path.join(save_dir, 'mod_dissolve_headwaters.json'), 'w') as f:
    #     json.dump(adjoint_order_2_dict, f)
    #
    # # Find order 1 streams that join an order 3+ stream
    # logger.info('\tLooking for order 1 streams that join order 3+')
    # order_1_branches = streams_df.loc[streams_df[order_field] == 1]
    # higher_order_trunks = streams_df.loc[streams_df[order_field] >= 3, id_field].values
    # ds_connections = order_1_branches.loc[order_1_branches[ds_id_field].isin(higher_order_trunks), ds_id_field].values
    #
    # pairs_to_prune = streams_df.loc[streams_df[id_field].isin(ds_connections), ['USLINKNO1', 'USLINKNO2']]
    # pairs_to_prune = {str(us1): [us1, us2] for us1, us2 in
    #                   zip(pairs_to_prune['USLINKNO1'], pairs_to_prune['USLINKNO2'])}
    #
    # with open(os.path.join(save_dir, 'mod_prune_shoots.json'), 'w') as f:
    #     json.dump(pairs_to_prune, f)
    return


def slim_streams_df(save_dir: str,
                    streams_df: pd.DataFrame = None,
                    id_field: str = 'LINKNO',
                    n_processes: int or None = None) -> pd.DataFrame:
    if streams_df is None:
        streams_df = pd.read_parquet(os.path.join(save_dir, 'rapid_inputs_master.parquet'), engine='fastparquet')

    # logger.info('\tDropping pruned shoots')
    # with open(os.path.join(save_dir, 'mod_prune_shoots.json'), 'r') as f:
    #     pruned_shoots = json.load(f)
    #     pruned_shoots = set([ids[-1] for _, ids in pruned_shoots.items()])
    #     streams_df = streams_df.loc[~streams_df[id_field].isin(pruned_shoots)]

    logger.info('\tDropping small trees')
    small_trees = pd.read_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv')).values.flatten()
    streams_df = streams_df.loc[~streams_df[id_field].isin(small_trees)]

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
