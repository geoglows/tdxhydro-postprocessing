import json
import os
from itertools import chain

import numpy as np
import geopandas as gpd
import pandas as pd
import dask.dataframe as dd
import logging

from .trace_streams import create_adjoint_json
from .correct_network import identify_0_length

__all__ = ['streams', ]

logger = logging.getLogger(__name__)


def streams(streams_gdf: str,
            save_dir: str,
            id_field='LINKNO',
            ds_field: str = 'DSLINKNO',
            len_field: str = 'Length',
            order_field: str = 'strmOrder',
            n_processes: int = None) -> None:
    """
    Analyzes the connectivity of the streams to find errors and places that need to be dissolved

    Creates several files describing the network and how to modify it:
    - adjoint_tree.json: A json file that describes the network as an adjoint connections of stream
    - mod_zero_length.csv: A csv file that describes the stream segments that have a length of 0 in 3 cases
    - mod_drop_small_trees.csv: A csv list of all segments part of a drainage tree less than 75km2 at the outlet
    - mod_dissolve_headwaters.json: A json file that describes the headwaters streams to be dissolved together
    - mod_prune_shoots.json: A json file that describes the order 1 streams to be pruned from the drainage tree

    Args:
        streams_gdf (str): Path to delineation network file
        save_dir (str): Path to directory where dissolved network and catchments will be saved
        id_field (str, optional): Field name with to the unique id of each stream segment
        ds_field (str, optional): Field name with the unique downstream id of each stream segment
        len_field (str, optional): Field in network file that corresponds to the length of each stream segment
        order_field: Field in network file that corresponds to the strahler order of each stream segment
        n_processes (int, optional): Number of processes to use for parallel processing

    Returns:
        None
    """
    logger.info('Reading streams')
    streams_df = gpd.read_file(streams_gdf, ignore_geometry=True)

    # trace network to create adjoint tree
    logger.info('Tracing network')
    adjoint_dict = create_adjoint_json(streams_df, id_field=id_field, ds_field=ds_field, order_field="strmOrder")
    with open(os.path.join(save_dir, 'adjoint_tree.json'), 'w') as f:
        json.dump(adjoint_dict, f)

    # Drop trees with small total length/area
    logger.info('Finding small trees')
    small_tree_outlet_ids = streams_df.loc[np.logical_and(
        streams_df[ds_field] == -1,
        streams_df['DSContArea'] < 75_000_000
    ), id_field].values
    small_tree_segments = set(chain.from_iterable([adjoint_dict[str(x)] for x in small_tree_outlet_ids]))
    pd.DataFrame(small_tree_segments).to_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv'), index=False)

    # Fix 0 length segments
    if 0 in streams_df[len_field].values:
        logger.info("Classifying length 0 segments")
        zero_length_fixes_df = identify_0_length(streams_df, id_field, ds_field, len_field)
        zero_length_fixes_df.to_csv(os.path.join(save_dir, 'mod_zero_length_streams.csv'), index=False)

    # Find headwater streams to be dissolved
    adjoint_order_2_dict = create_adjoint_json(streams_df, id_field=id_field, ds_field=ds_field,
                                               order_field=order_field, order_filter=2)

    # list all ids that were merged, turn a list of lists into a flat list, remove duplicates by converting to a set
    logger.info('Finding headwater streams to be dissolved')
    top_order_2s = {str(value[-1]) for value in list(adjoint_order_2_dict.values())}
    adjoint_order_2_dict = {key: adjoint_dict[key] for key in top_order_2s}
    with open(os.path.join(save_dir, 'mod_dissolve_headwaters.json'), 'w') as f:
        json.dump(adjoint_order_2_dict, f)

    # Find order 1 streams that join an order 3+ stream
    logger.info('Looking for order 1 streams that join order 3+')
    order_1_branches = streams_df.loc[streams_df[order_field] == 1]
    higher_order_trunks = streams_df.loc[streams_df[order_field] >= 3, id_field].values
    downstream_connections = order_1_branches.loc[order_1_branches[ds_field].isin(higher_order_trunks), ds_field].values

    pairs_to_prune = streams_df.loc[streams_df[id_field].isin(downstream_connections), ['USLINKNO1', 'USLINKNO2']]
    # pairs_to_prune['orderUS1'] = pairs_to_prune['USLINKNO1'].apply(
    #     lambda x: streams_df.loc[streams_df['LINKNO'] == x, 'strmOrder'].values[0])
    # pairs_to_prune['orderUS2'] = pairs_to_prune['USLINKNO2'].apply(
    #     lambda x: streams_df.loc[streams_df['LINKNO'] == x, 'strmOrder'].values[0])
    # if not all(pairs_to_prune['orderUS1'] > pairs_to_prune['orderUS2']):
    #     logger.info('All order 1 streams that join order 3+ streams have the same order upstream')
    #     raise RuntimeError('some streams have the US1 of lower or equal order to US2')
    pairs_to_prune = {str(us1): [us1, us2] for us1, us2 in
                      zip(pairs_to_prune['USLINKNO1'], pairs_to_prune['USLINKNO2'])}

    with open(os.path.join(save_dir, 'mod_prune_shoots.json'), 'w') as f:
        json.dump(pairs_to_prune, f)
    return
