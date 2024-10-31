from utils.config_lib import set_flags
from process.process_ton import toniot_train_dec, toniot_test_dec
from torch.utils.data import DataLoader


def load_data():
    num_worker = 5
    batch_size = 16
    ft_size = 71
    nb_nodes = 43
    tr_dataiterator_list = []
    ts_dataiterator_list = []
    for n in range(num_worker):
        train_data = toniot_train_dec(n, repeat=None)
        test_data = toniot_test_dec(n, repeat=None)
        train_dataloder = DataLoader(train_data, batch_size=batch_size,
                                     num_workers=0, drop_last=True, shuffle=False)
        test_dataloder = DataLoader(test_data, batch_size=1,
                                    num_workers=0, drop_last=True, shuffle=False)
        tr_dataiterator = iter(train_dataloder)
        ts_dataiterator = iter(test_dataloder)
        tr_dataiterator_list.append(tr_dataiterator)
        ts_dataiterator_list.append(ts_dataiterator)
    return num_worker, ft_size, nb_nodes, tr_dataiterator_list, ts_dataiterator_list