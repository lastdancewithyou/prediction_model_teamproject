import os
import numpy as np
import pandas as pd
import torch
import pickle

from utils.args import pitch_sequence_parser
from utils.utils import all_files_exist, divide_sequence_pitch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter

class StatcastDataset(Dataset):
    def __init__(self, 
                 data_path,
                 data_name,
                 flag,
                 target_list = ['pitch_name', 'zone'],
                 use_col_list = ['pitch_name', 'release_speed', 'release_pos_x', 'release_pos_z', 'pitcher',
                               'batter', 'zone', 'stand_R', 'p_throws_R', 'balls', 'strikes', 'pfx_x', 'pfx_z', 'innings',
                               'plate_x', 'plate_z', 'outs_when_up', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
                               'effective_speed', 'release_spin_rate', 'release_extension', 'release_pos_y',
                               'pitch_number', 'spin_axis', 'target_loc'],
                 pitch_name_list = ['4-Seam Fastball', 'Sinker', 'Slider', 'Changeup', 'Cutter', 'Sweeper', 'Curveball',
                       'Split-Finger', 'Knuckle Curve'],
                 scaler_mode = 'standard',
                 train_size = 0.8,
                 val_size = 0.2
                 ):
        self.data_path = data_path
        self.data_name = data_name
        self.target_list = target_list
        self.use_columns = use_col_list
        self.pitch_types = pitch_name_list
        self.scaler_mode = scaler_mode
        self.flag = flag
        self.train_size = train_size
        self.val_size = val_size
        
        self.sequence_data_path = os.path.join('./data', self.data_name)
        
        if not os.path.exists(self.sequence_data_path) or not all_files_exist(self.sequence_data_path):
            self.__read_data__()
            self.__preprocess__()
            self.__devide_sequence__()
            self.__pad_and_split_sequence__()
            
        else:
            with open(os.path.join(self.sequence_data_path, "seq.pkl"), "rb") as f:
                self.sequences = pickle.load(f)
                
            with open(os.path.join(self.sequence_data_path, "targets.pkl"), "rb") as g:
                self.targets = pickle.load(g)
                
            self.__pad_and_split_sequence__()
            
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self.data_name + '.csv'))

        # pitcher 칼럼을 정수형 인코딩으로 변환
        convert_lst = ['pitcher', 'batter', 'game_pk']
        for col in convert_lst:
            df_raw[col] = df_raw[col].astype('category').cat.codes
        # df_raw['pitch_name'] = df_raw['pitch_name'].astype('category').cat.codes

        # 필요 칼럼만 필터링
        df_raw = df_raw[self.use_columns]
        self.numeric_columns = [col for col in df_raw.columns if col not in ['pitcher', 'batter']]
        self.int_chr_columns = ['inning', 'zone', 'balls', 'strikes', 'outs_when_up', 'pitch_number', 'target_loc']
        self.encoding_columns = [col for col in df_raw.columns if col.startswith('pitch_type_') or col.startswith('p_throws_') or col.startswith('stand_') or col.startswith('n_') or col.startswith('on_')]
        self.cont_columns = [col for col in self.numeric_columns if col not in self.int_chr_columns and col not in self.encoding_columns and col != 'pitch_name']

        self.data = df_raw.copy()  # 처리된 데이터 그대로 저장
        

    def __preprocess__(self):
        # 수치형 데이터 정규화
        if self.scaler_mode == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_mode == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise Exception("Scaler must be standardscaler or minmaxscaler")

        self.data[self.cont_columns] = self.scaler.fit_transform(self.data[self.cont_columns])
        
    def __devide_sequence__(self):
        # Pitch type에 따른 dictionary 생성 및 mapping
        self.pitch_mapping = {pitch: idx for idx, pitch in enumerate(self.pitch_types, start=0)}
        self.pitch_mapping['Others'] = len(self.pitch_mapping)
        self.data['pitch_name'] = self.data['pitch_name'].map(self.pitch_mapping).fillna(self.pitch_mapping['Others'])
        
        # Pitch location에 따른 dictionary 생성 및 mapping
        location = ['in_high', 'out_high', 'in_low', 'out_low']
        self.loc_mapping = {location: idx for idx, location in enumerate(location, start = 0)}
        self.data['target_loc'] = self.data['target_loc'].map(self.loc_mapping)
        
        ## target 뒤로 빼고 정렬
        use_col_list = [col for col in self.use_columns if col not in ['pitch_name', 'target_loc']] + ['pitch_name', 'target_loc']
        self.data = self.data[use_col_list]
        
        print(self.pitch_mapping)
        
        print(self.loc_mapping)
        
        self.sequences, self.targets = divide_sequence_pitch(self.data,
                                                             numeric_columns=self.numeric_columns,
                                                             encoding_columns=self.encoding_columns,
                                                             pitch_num_col='pitch_number', 
                                                             target_columns=self.target_list)
        
        # numpy 저장
        os.makedirs(self.sequence_data_path, exist_ok = True)
        
        with open(os.path.join(self.sequence_data_path, "seq.pkl"), "wb") as f:
            pickle.dump(self.sequences, f)
        
        with open(os.path.join(self.sequence_data_path, "targets.pkl"), "wb") as g:
            pickle.dump(self.targets, g)
            
        with open(os.path.join(self.sequence_data_path, "pitch_mapping.pkl"), 'wb') as h:
            pickle.dump(self.pitch_mapping, h)
            
        with open(os.path.join(self.sequence_data_path, "loc_mapping.pkl"), 'wb') as i:
            pickle.dump(self.loc_mapping, i)
        
        
    def __pad_and_split_sequence__(self):
        # 전체 sequence 패딩 처리
        seq_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in self.sequences]
        self.padded_sequences = pad_sequence(seq_tensors, batch_first=True, padding_value=0.0).permute(0, 2, 1)
        self.max_seq_len = self.padded_sequences.shape[-1]  # 모든 세트에서 동일한 max_seq_len을 사용

        # 시퀀스 길이와 패딩 마스크 생성
        self.real_sequence_length = torch.tensor([len(seq) for seq in self.sequences])
        self.padding_mask = torch.zeros(self.padded_sequences.shape[0], self.max_seq_len, dtype=torch.bool)
        for i, length in enumerate(self.real_sequence_length):
            self.padding_mask[i, length:] = True

        # 타겟을 텐서로 변환
        self.target_tensors = torch.stack([torch.tensor(tar_seq, dtype=torch.int) for tar_seq in self.targets], axis=0)

        # train, valid, test 나누기
        total_length = len(self.sequences)
        train_valid_length = int(total_length * self.train_size)
        val_length = int(train_valid_length * self.val_size)
        train_length = train_valid_length - val_length

        if self.flag == 'train':
            self.padded_sequences = self.padded_sequences[:train_length]
            self.real_sequence_length = self.real_sequence_length[:train_length]
            self.padding_mask = self.padding_mask[:train_length]
            self.target_tensors = self.target_tensors[:train_length]

        elif self.flag == 'valid':
            self.padded_sequences = self.padded_sequences[train_length:train_valid_length]
            self.real_sequence_length = self.real_sequence_length[train_length:train_valid_length]
            self.padding_mask = self.padding_mask[train_length:train_valid_length]
            self.target_tensors = self.target_tensors[train_length:train_valid_length]

        elif self.flag == 'test':
            self.padded_sequences = self.padded_sequences[train_valid_length:]
            self.real_sequence_length = self.real_sequence_length[train_valid_length:]
            self.padding_mask = self.padding_mask[train_valid_length:]
            self.target_tensors = self.target_tensors[train_valid_length:]
            
        ## 범주형 빈도 리스트
        pitch_type_count = [0] * 10
        pitch_location_count = [0] * 4
        
        for target in self.target_tensors[: train_length]:
            pitch_type_count[int(target[0])] += 1
            pitch_location_count[int(target[1])] += 1
            
        ## 빈도를 1에서 빼서 비율 계산
        self.pitch_type_freq = [1 - (i / sum(pitch_type_count)) for i in pitch_type_count]
        self.pitch_location_freq = [1 - (i / sum(pitch_location_count)) for i in pitch_location_count]
    
    def __len__(self):
        return len(self.padded_sequences)

    def __getitem__(self, index):
        data_dict = {
            "padded_data": self.padded_sequences[index],
            "real_sequence_length": self.real_sequence_length[index],
            "padding_mask": self.padding_mask[index],
            "targets": self.target_tensors[index]
        }
        return data_dict
    
    def _return_weight(self):
        return self.pitch_type_freq, self.pitch_location_freq

    def inverse_transform(self, data):
        """
        정규화 역변환
        """
        return self.scaler.inverse_transform(data)
    
    
def get_loader(args, flag):
    statcast_dataset = StatcastDataset(
        data_name = args.data_name,
        data_path = args.data_path,
        target_list = args.target_list,
        use_col_list = args.use_col_list,
        pitch_name_list = args.pitch_name_list,
        scaler_mode = args.scaler_mode,
        flag = flag,
        train_size = 0.8,
        val_size = 0.2
    )
    
    # print(statcast_dataset.pitch_type_dictionary())
    
    shuffle = False

    if flag == 'train':
        shuffle = True
    
    statcast_loader = DataLoader(
        statcast_dataset,
        batch_size = args.batch_size,
        shuffle = shuffle,
        drop_last = True
    )
    
    if flag == 'train':
        type_weight, location_weight = statcast_dataset._return_weight()
        return statcast_dataset, statcast_loader, type_weight, location_weight
        
    return statcast_dataset, statcast_loader


if __name__ == '__main__':
    parser = pitch_sequence_parser()
    args = parser.parse_args()
    dataset, loader = get_loader(args, flag = 'train')
    print(next(iter(loader))['padded_data'].shape)