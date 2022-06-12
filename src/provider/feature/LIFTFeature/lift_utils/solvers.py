import importlib

def create_network(pathconf, param, train_data_in):
    network_module = importlib.import_module( f'src.provider.feature.LIFTFeature.lift_utils.networks.{param.dataset.dataType.lower()}_{param.model.modelType.lower()}')

    net_config = network_module.NetworkConfig()

    # ------------------------------------------------------------------------
    # Copy over all other attributes to Config, destroying the group
    for _group in param.__dict__.keys():
        for _key in getattr(param, _group).__dict__.keys():
            setattr(net_config, _key, getattr(getattr(param, _group), _key))

    # ------------------------------------------------------------------------
    # Config fields which need individual attention

    # directories
    net_config.save_dir = pathconf.result

    # dataset info (let's say these should be given in the data structure?)
    net_config.num_channel = train_data_in.num_channel
    net_config.patch_height = train_data_in.patch_height
    net_config.patch_width = train_data_in.patch_width
    net_config.out_dim = train_data_in.out_dim

    # ------------------------------------------------------------------------
    # Actual instantiation and setup
    net = network_module.Network(net_config)

    net.setup4Test()  # Setup Test
    net.compile4Test()  # Compile Test
    net.setupSpecific()  # Setup specific to runType
    net.compileSpecific()  # Compile specific to runType

    return net


def model(pathconf,
         param,
         test_data_in,
         test_mode=None,
         network_weights=None):

    net = create_network(pathconf, param, test_data_in)

    return net.runTest( test_data_in,
                        test_mode=test_mode,
                        deterministic=True,
                        model_epoch="",
                        verbose=True,
                        network_weights=network_weights)
