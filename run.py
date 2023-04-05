from RAPIDprep import preprocess_for_rapid
import glob
import os
import datetime


if __name__ == '__main__':
    rapid_outputs = '/tdxprocessed/rapidio'

    sample_grids = glob.glob('./era5_sample_grids/*.nc')

    numbers_to_skip = [

    ]

    for streams_gpkg, basins_gpkg in zip(
            sorted(glob.glob(f'/tdxhydro/TDX_streamnet*.gpkg')),
            sorted(glob.glob(f'/tdxhydro/TDX_streamreach_basins*.gpkg'))
    ):
        region_number = os.path.basename(streams_gpkg).split('_')[2]
        if region_number in numbers_to_skip:
            continue

        out_dir = os.path.join(rapid_outputs, region_number)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(streams_gpkg)
        print(basins_gpkg)
        print(region_number)
        print(out_dir)

        preprocess_for_rapid(
            streams_gpkg,
            basins_gpkg,
            sample_grids,
            out_dir,
            n_processes=24
        )
