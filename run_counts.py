import pandas as pd
import glob
import os

counts = []
for region in sorted(glob.glob('/Volumes/EB406_T7_2/TDXOutputsNew/*')):
    region_number = os.path.basename(region)
    count = pd.read_csv(os.path.join(region, 'comid_lat_lon_z.csv')).shape[0]
    counts.append([region_number, count])

df = pd.DataFrame(counts, columns=['region', 'count'])
df.to_csv('network_data/stream_counts.csv', index=False)
print(df['count'].sum())
