from absl import flags


def set_flags():
    FLAGS = flags.FLAGS
    flags.DEFINE_integer(
        'epochs', 200,
        'Number of epochs of training')
    flags.DEFINE_float(
        'stuggle_rate', 0.2, 'stuggle_rate')
    flags.DEFINE_string(
        'node_activation', 'elu',
        'Activation for node attention layers.')
    flags.DEFINE_string(
        'path_activation', 'relu',
        'Activation for path attention layers.')
    flags.DEFINE_bool('skip_connection', True,
                      'Whether use skip connection in multi layers')
    flags.DEFINE_integer(
        'T_asyn', 200,
        'tau')


    return FLAGS
