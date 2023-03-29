from RAPIDprep import PreprocessForRAPID

if __name__ == '__main__':
    path_to_stream_network = "C:\\User\\network.gpkg"
    path_to_basins = "C:\\User\\basins.gpkg"

    out_dir = "C:\\User\\rapid"
    nc = "C:\\User\\ncfile.nc"

    PreprocessForRAPID(path_to_stream_network, path_to_basins, [nc], out_dir)
