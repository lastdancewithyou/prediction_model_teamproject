import argparse

def pitch_sequence_parser():
    parser = argparse.ArgumentParser(description= "Experiment for Pitch Sequence Classification")
    
    ## seed
    parser.add_argument('--seed', type = int, default = 42)
    
    ## data
    parser.add_argument('--data_name', type = str, default = 'pybaseball_data_cleaned_v3')
    parser.add_argument('--target_list', type = list, default = ['pitch_side_type', 'target_loc'])
    parser.add_argument('--use_col_list', type = list, default = ['pitch_name', 'release_speed', 'release_pos_x', 'release_pos_z',
       'pitcher', 'batter', 'zone', 'balls', 'strikes', 'pfx_x', 'pfx_z',
       'plate_x', 'plate_z', 'outs_when_up', 'vx0', 'vy0', 'vz0', 'ax', 'ay',
       'az', 'effective_speed', 'release_spin_rate', 'release_extension',
       'release_pos_y', 'pitch_number', 'spin_axis', 'game_pk',
       'at_bat_number', 'score_diff', 'stand_R', 'p_throws_R', 'on_3b_1',
       'on_2b_1', 'on_1b_1', 'score_diff', 'inning', 'cumulative_obp', 'cumulative_slg', 'cumulative_ops',
       'game_obp', 'game_slg', 'game_ops', 'winning', 'losing', 'tied', 'tight' , 'low_zone_ratio',
       'n_ff', 'n_si', 'n_fc', 'n_sl', 'n_ch',	'n_cu','n_fs','n_kn','n_st','n_sv', 'target_loc', 'is_fastball', 'pitch_side_type',
       'pp_ff', 'pp_si', 'pp_sl', 'pp_ch', 'pp_ct', 'pp_sw', 'pp_cu', 'pp_fs', 'pp_kn', 'pp_other'])
    parser.add_argument('--pitch_name_list', type = list, default = ['4-Seam Fastball', 'Sinker', 'Slider', 'Changeup', 'Cutter', 'Sweeper', 'Curveball',
                       'Split-Finger', 'Knuckle Curve'])
    parser.add_argument('--pitch_side_type_list', type = list, default = ['Fastball', 'GV_side', 'Arm_side', 'Other'])
    parser.add_argument('--c_in', type = int, default = 61)
    parser.add_argument('--seq_len', type = int, default = 15)
    parser.add_argument('--data_path', type = str, default = './download_data')
    parser.add_argument('--scaler_mode', type = str, default = 'minmax')
    
    parser.add_argument('--model', type = str, default = 'Transformer')
    
    ## LSTM
    
    parser.add_argument('--input_size', type = int, default = 61)
    parser.add_argument('--hidden_size', type = int, default = 1024)
    parser.add_argument('--fc_hidden_size', type = int, default =128)
    parser.add_argument('--dropout', type = float, default = 0.1)
    
    ## Transformer
    parser.add_argument('--d_model', type = int, default = 128)
    parser.add_argument('--d_ff', type = int, default = 512)
    parser.add_argument('--n_layers', type = int, default = 6)
    parser.add_argument('--nhead', type = int, default = 8)
    parser.add_argument('--use_norm', type = str, default = False)
    
    ## training
    parser.add_argument('--learning_rate', type = float, default = 1e-4)
    parser.add_argument('--ld', type = float, default = 0.5)
    parser.add_argument('--batch_size', type = int, default = 1024)
    parser.add_argument('--epoch', type = int, default =20)
    
    ## gpu
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    
    ## save
    parser.add_argument('--model_save_pth', type =str, default = './save')
    parser.add_argument('--model_path', type = str, default = None)
    
    return parser

