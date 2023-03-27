path_to_stream_network = "C:\\User\\network.gpkg"
path_to_basins = "C:\\User\\basins.gpkg"

out_dir = "C:\\User\\rapid"
nc = "C:\\User\\ncfile.nc"

if __name__ == '__main__':
    # from multiprocess_dissolve import main as processNetwork
    # processNetwork(path_to_stream_network, path_to_basins, model=True) # Leave model as True. Haven't corrected mapping yet
    # from rapid_preproccess import main as generateRAPID
    # generateRAPID(out_dir,nc)
    from RAPIDprep import PreprocessForRAPID
    PreprocessForRAPID(path_to_stream_network, path_to_basins, [nc], out_dir)