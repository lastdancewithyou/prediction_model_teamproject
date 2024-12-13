import os
from tqdm import tqdm

def all_files_exist(data_path):
    files = [
        os.path.join(data_path, 'seq.pkl'),
        os.path.join(data_path, 'targets.pkl')
    ]
    return all(os.path.exists(f) for f in files)

# def divide_sequence_pitch(df, numeric_columns, encoding_columns, pitch_num_col='pitch_number', target_columns=['pitch_name', 'zone'], max_seq_len=20):
#     """
#     Pitch Sequence 구분
#     """
#     sequences = []
#     targets = []
#     total_pitches = 0
#     ignored_pitches = 0

#     # 타석 단위로 그룹핑: game_pk, at_bat_number, inning 기준으로 그룹핑하여 타석 단위로 나눔
#     player_group = df.groupby(['game_pk', 'at_bat_number', 'inning', 'pitcher', 'batter'], observed=True)

#     for _, group in tqdm(player_group, desc="Processing at-bats", total=len(player_group)):  
#         group = group.sort_values(by=pitch_num_col)
        
#         total_pitches += len(group)

#         temp_sequence = []
#         temp_target = []
        
#         for i in range(len(group)):
#             temp_sequence.append(group.iloc[i][numeric_columns + encoding_columns].values)
#             temp_target.append(group.iloc[i][target_columns].values)

#             if len(temp_sequence) >= max_seq_len:
#                 sequences.append(temp_sequence[:max_seq_len])
#                 targets.append(temp_target[:max_seq_len])
#                 ignored_pitches += len(temp_sequence) - max_seq_len
#                 temp_sequence = []
#                 temp_target = []

#         if len(temp_sequence) > 0:
#             if len(temp_sequence) > max_seq_len:
#                 ignored_pitches += len(temp_sequence) - max_seq_len
#                 temp_sequence = temp_sequence[:max_seq_len]
#                 temp_target = temp_target[:max_seq_len]

#             sequences.append(temp_sequence)
#             targets.append(temp_target)

#     ignored_ratio = ignored_pitches / total_pitches * 100 if total_pitches > 0 else 0
#     print(f"Total pitches: {total_pitches}, Ignored pitches: {ignored_pitches}, Ignored ratio: {ignored_ratio:.2f}%")

#     return sequences, targets, ignored_ratio

def divide_sequence_pitch(df, numeric_columns, encoding_columns, pitch_num_col='pitch_number', target_columns=['pitch_name', 'zone']):
    """
    Pitch Sequence 구분
    """
    sequences = []
    targets = []
    player_group = df.groupby(['pitcher', 'batter', 'inning', 'outs_when_up','game_pk'], observed=True)
    
    print(len(list(set(numeric_columns + encoding_columns))))

    for player, group in tqdm(player_group):
        group = group.sort_values(by=pitch_num_col)
        
        if len(group) == 1:
            pass
        else:
            for i in range(1, len(group)):
                sequences.append(group.iloc[:i][list(set(numeric_columns + encoding_columns))].values)
                targets.append(group.iloc[i][target_columns].values)

    return sequences, targets

def calculate_pitch_frequencies(df, pitch_types):
    """
    Calculates the pitch type frequency for each pitcher and adds new columns for each pitch type frequency.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing pitcher and pitch type dummy variables.

    Returns:
    pd.DataFrame: A DataFrame with added columns representing pitch type frequencies for each pitcher.
    """
    
    # 각 투수의 구종 빈도 계산
    pitcher_pitch_counts = df.groupby('pitcher')[pitch_types].sum()

    # 투수별 총 투구 수 계산
    pitcher_total_pitches = pitcher_pitch_counts.sum(axis=1)

    # 각 투수별 구종 비율 계산
    pitcher_pitch_frequencies = pitcher_pitch_counts.div(pitcher_total_pitches, axis=0)

    # 구종 비율 컬럼 이름을 변경하여 pitch_freq_로 시작하도록 함
    pitcher_pitch_frequencies.columns = [f'pitch_freq_{ptype.split("_")[2]}' for ptype in pitch_types]

    # 원래 데이터프레임에 구종 비율을 병합
    df = df.merge(pitcher_pitch_frequencies, on='pitcher', how='left')
    
    return df