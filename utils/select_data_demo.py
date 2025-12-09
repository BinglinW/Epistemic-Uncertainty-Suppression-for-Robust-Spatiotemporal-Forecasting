from torch.utils.data import DataLoader


def select_data_demo(config):
    from algorithm.weather_preprocess import DataPreprocess, BoeingDataset

    file_path = "./data/weather/Hyperlocal_Temperature_Monitoring_20240928.csv"
    node_index_list = [
        ["Day", "Hour"],
        ["Longitude", "Latitude"],
        ["Install.Type", "Borough", "ntacode"],
        ["AirTemp"],
    ]

    # Use full length of the demo CSV
    data_range = [0, -1]

    config.data.num_vertices = 4
    config.data.num_features = 3
    config.data.points_per_hour = 1

    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features

    n_utils = [config.data.num_features, 16, 16]

    air_data = DataPreprocess(file_path, config.data.num_features, node_index_list)
    construct_data = air_data.get_data()
    dataset = BoeingDataset(construct_data, data_range, config)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    return config, n_utils, dataloader, air_data
