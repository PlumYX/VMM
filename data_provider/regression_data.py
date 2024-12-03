import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Common_Dataset(Dataset):
    def __init__(self, root_path='.\datasets',
                 model_input_data_path='data.csv',
                 model_output_data_path=None,
                 input_index='all',
                 output_index='all',
                 flag='train',
                 data_file_suffix='csv',
                 scale=True):

        """ input_index & output_index: list or 'all' """

        assert flag in ['train', 'val', 'test', 'deploy']
        self.flag = flag
        self.root_path = root_path
        self.model_input_data_path = model_input_data_path
        self.model_output_data_path = model_output_data_path \
            if model_output_data_path else model_input_data_path
        self.input_index = input_index
        self.output_index = output_index
        self.scale = scale
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_hdf(os.path.join(self.root_path, self.model_input_data_path))

        x_columns = self.input_index
        y_columns = self.input_index
        data_columns = x_columns + y_columns
        df_data = df_raw[data_columns]

        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[0::2]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.flag == 'train':
            self.data_x, self.data_y = data[0::2, 0:10], data[0::2, 10:]
        elif self.flag == 'val':
            self.data_x, self.data_y = data[1::4, 0:10], data[1::4, 10:]
        elif self.flag == 'val':
            self.data_x, self.data_y = data[3::4, 0:10], data[3::4, 10:]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size

    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1

    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Common_Regression_Data(root_path=args.root_path, data_path=args.data_path, flag=flag)
    print(flag.capitalize(), 'dataset:', len(data_set))

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, 
                             num_workers=args.num_workers, drop_last=drop_last)

    return data_set, data_loader
