from RAPIDprep import PreprocessForRAPID
import glob


if __name__ == '__main__':
    path_to_stream_network = '/Users/rchales/Downloads/TDX_streamnet_7020065090_01.gpkg'
    path_to_basins = '/Users/rchales/Downloads/TDX_streamreach_basins_7020065090_01.gpkg'

    out_dir = '/Users/rchales/Data/tdxhydro_rapid_files'
    sample_grids = glob.glob('/Volumes/EB406_T7_1/era5_sample_grids/*.nc')

    PreprocessForRAPID(path_to_stream_network, path_to_basins, sample_grids, out_dir)
