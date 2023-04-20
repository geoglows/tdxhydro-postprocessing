import pandas as pd
import glob
import os

# todo put this in the validate directory
counts = []
for region in sorted(glob.glob('/Volumes/EB406_T7_2/TDXHydroRapid/*')):
    region_number = os.path.basename(region)
    comid_df = pd.read_csv(os.path.join(region, 'comid_lat_lon_z.csv'))
    rivid_df = pd.read_csv(os.path.join(region, 'riv_bas_id.csv'), header=None)
    comid_count = comid_df.shape[0]
    rivid_count = rivid_df.shape[0]
    if comid_count != rivid_count:
        print('WARNING: region {} has inconsistent comid and rivid counts'.format(region_number))
        print(region_number, comid_count, rivid_count)
        missing_ids = [i for i in comid_df['comid'].values if i not in rivid_df['comid'].values]
        missing_ids += [i for i in rivid_df['comid'].values if i not in comid_df['comid'].values]
    counts.append([region_number, rivid_count])

    wt_sizes = [pd.read_csv(f).groupby('streamID').sum().shape[0] for f in glob.glob(os.path.join(region, 'weight_*0.csv'))]
    if not all([size == comid_count for size in wt_sizes]):
        print('WARNING: region {} has inconsistent weight table sizes'.format(region_number))
        print(region_number, rivid_count, wt_sizes)

df = pd.DataFrame(counts, columns=['region', 'count'])
df.to_csv('network_data/stream_counts.csv', index=False)
print(df['count'].sum())
