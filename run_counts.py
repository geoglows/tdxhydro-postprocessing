import pandas as pd
import glob
import os

# todo put this in the validate directory
counts = []
for region in sorted(glob.glob('/Volumes/EB406_T7_2/TDXHydroRapid/*')):
    region_number = os.path.basename(region)
    comid_path = os.path.join(region, 'comid_lat_lon_z.csv')
    rivid_path = os.path.join(region, 'riv_bas_id.csv')
    if not os.path.exists(comid_path) or not os.path.exists(rivid_path):
        continue
    comid_df = pd.read_csv(comid_path).astype(int)
    rivid_df = pd.read_csv(rivid_path, header=None).astype(int)
    comid_count = comid_df.shape[0]
    rivid_count = rivid_df.shape[0]
    print(region_number, comid_count, rivid_count)
    if comid_count != rivid_count:
        print('---> Inconsistent comid and rivid counts')
    sum_comid = comid_df['LINKNO'].sum()
    sum_rivid = rivid_df.values.flatten().sum()
    print(sum_comid - sum_rivid)
    # missing_ids = [i for i in comid_df['LINKNO'].values if i not in rivid_df.values.flatten()]
    # missing_ids += [i for i in rivid_df.values.flatten() if i not in comid_df['LINKNO'].values]
    # missing_ids = set(missing_ids)
    if sum_comid - sum_rivid != 0:
        print(f'Sum of ID column not equal')
        # print(region_number, len(missing_ids))
        # print(missing_ids)
    counts.append([region_number, rivid_count])

    # if all([comid_df['LINKNO'].isin(rivid_df['LINKNO'].values[i] for i in range(comid_count)]):

    wt_sizes = [pd.read_csv(f).groupby('streamID').sum().shape[0] for f in glob.glob(os.path.join(region, 'weight_*0.csv'))]
    if not all([size == comid_count for size in wt_sizes]):
        print('WARNING: region {} has inconsistent weight table sizes'.format(region_number))
        print(region_number, rivid_count, wt_sizes)

df = pd.DataFrame(counts, columns=['region', 'count'])
# df.to_csv('network_data/stream_counts.csv', index=False)
print(df['count'].sum())
