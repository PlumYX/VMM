import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Common_TimeSeries_Data(Dataset):
    def __init__(self, root_path='.\datasets',
                 input_data_path='input.csv',
                 output_data_path='output.csv',
                 input_index=None,
                 outpur_index=None,
                 flag='train',
                 scale=True):

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_hdf(os.path.join(self.root_path, self.data_path))

        x_columns = ['Time', 'x-coordinate', 'y-coordinate', 'z-coordinate', 
                     'Pump-mflowj', 'Rktpow', 'SG-in2-tempf', 'main-P', 'SG-in1-tempf', 'SG-in2-mflowj']

        y_columns = ['velocity-magnitude', 'temperature', 'phase-2-vof', 
                     'tav-outlet1&2(outlet_1)', 'tav-outlet1&2(outlet_2)']

        data_columns = x_columns + y_columns
        df_data = df_raw[data_columns]

        if self.scale:
            train_data = df_data[0::2]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        if self.set_type == 0:    # Train
            self.data_x, self.data_y = data[0::2, 0:10], data[0::2, 10:]
        elif self.set_type == 1:  # Val
            self.data_x, self.data_y = data[1::4, 0:10], data[1::4, 10:]
        else:                     # Test
            self.data_x, self.data_y = data[3::4, 0:10], data[3::4, 10:]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)