import time
import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
import sys

from pystoi import stoi
from torchaudio import load
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd

from flowmse.data_module import SpecsDataModule
from flowmse.model import VFModel
import pdb
import os
from flowmse.util.other import pad_spec
from flowmse.sampling import get_white_box_solver
from utils import energy_ratios, ensure_dir, print_mean_std

import pdb

if __name__ == '__main__':
    parser = ArgumentParser()
   

    parser.add_argument("--odesolver", type=str,
                        default="euler", help="euler")
    parser.add_argument("--reverse_starting_point", type=float, default=1.0, help="Starting point for the reverse SDE.")
    parser.add_argument("--reverse_end_point", type=float, default=0.03)
    
    parser.add_argument("--test_dir")
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--N", type=int)
    parser.add_argument("--time_step_type", type=str, choices=('gerkmann', 'uniform'), default="gerkmann")


    args = parser.parse_args()
    N = args.N
    time_step_type = args.time_step_type

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")
    dataset_name= os.path.basename(os.path.normpath(args.test_dir))
    
    
    checkpoint_file = args.ckpt
    # int_list = "_".join(map(str, args.int_list))
    # raise("target_dir 부터 확인해")
    

    # Settings
    sr = 16000
    # print(args.int_list)
    odesolver = args.odesolver
    # int_list = args.int_list
    
    
    # Load score model

    model = VFModel.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=8, num_workers=4, kwargs=dict(gpu=False)
    )

    import re

    def extract_epoch(checkpoint_file):
        match = re.search(r'epoch=([0-9]{1,4})', checkpoint_file)
        if match:
            return int(match.group(1))  # 정수로 변환하여 반환
        return None  # 매칭되지 않으면 None 반환

    # 예제 경로
    match = re.search(r"mode_([a-zA-Z0-9]+)", checkpoint_file)

    # 결과 출력
    if match:
        mode_value = match.group(1)
        # print(mode_value)
    else:
        print("No match found")
    epoch_number = extract_epoch(checkpoint_file)
    target_dir = f"/workspace/results/condition_explore/{dataset_name}_mode_{model.mode_}_epoch_{epoch_number}_timestep_{time_step_type}_evaluationnumber_{N}/"
    results_candidate_path = os.path.join(target_dir, "_avg_results.txt")
    if os.path.exists(results_candidate_path):  # 파일 존재 여부 확인
        print(f"파일이 존재하므로 프로그램을 종료합니다: {results_candidate_path}")
        sys.exit()  # 프로그램 종료
    # print("evaluation sijak")
    ensure_dir(target_dir + "files/")
    reverse_starting_point = args.reverse_starting_point
    reverse_end_point = args.reverse_end_point
    
    model.ode.T_rev = reverse_starting_point
    
    mode_ = model.mode_    
    
    model.eval(no_ema=False)
    model.cuda()

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    



    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": []}
    for cnt, noisy_file in tqdm(enumerate(noisy_files)):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        x, _ = load(join(clean_dir, filename))
        y, _ = load(noisy_file)

        #pdb.set_trace()        

         
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        Y = Y.cuda()
        with torch.no_grad():
            
            xt, z = model.ode.prior_sampling(Y.shape, Y)
            sigma = model.ode._std(1)
            Y_plus_sigma_z = Y+sigma *z
            xt = xt.to(Y.device)
            if time_step_type=="gerkmann":
                timesteps = torch.linspace(reverse_starting_point, reverse_end_point, N, device=Y.device)
            elif time_step_type=="uniform":
                timesteps = torch.linspace(1, 1/N, N, device=Y.device)
                
            print("N, ", N)
            for i in range(len(timesteps)):
                t = timesteps[i]
                if i == len(timesteps)-1:
                    dt = 0-t
                else:
                    dt = timesteps[i+1]-t
                vect = torch.ones(Y.shape[0], device=Y.device)*t
                if mode_ == "noisemean_conditionfalse_timefalse":
                    xt = xt + dt * model(vect, xt)     
                elif mode_ == "noisemean_noxt_conditiony_timefalse":
                    xt = xt+dt*model(vect,Y)
                elif mode_ == "noisemean_y_plus_sigmaz":
                    xt = xt+dt*model(vect,Y_plus_sigma_z)
                elif mode_ == "noisemean_xt_t": #noisemean_xt_t: v_theta(xt,t)
                    xt = xt + dt * model(vect, xt)
                elif mode_ == "noisemean_xt_y": #noisemean_xt_y: v_theta(xt,y)
                    xt = xt + dt * model(vect, xt,Y)
                elif mode_ == "noisemean_xt_y_plus_sigmaz": #v_theta(xt,y+sigma z)
                    xt = xt + dt * model(vect, xt, Y_plus_sigma_z)
                elif mode_ == "noisemean_xtplusy_divide_2": #v_theta((xt+y)/2)
                    xt = xt + dt *model(vect, (xt+Y)/2)
                elif mode_ == "noisemean_xt_y_t": # noisemean_xt_y_t v_theta (xt,y,t)
                    xt = xt + dt * model(vect, xt, Y)
                elif mode_ == "noisemean_xt_t: v_theta(xt,t)": # noisemean_xt_t: v_theta(xt,t) 
                    xt = xt + dt * model(vect, xt)
                elif mode_ == "noisemean_t_times_y_plus_sigmaz_1minust_times_s": # "noisemean_t_times_y_plus_sigmaz_1minust_times_s": #": v_theta(t(y+sigma z), (1-t)s)"
                    first_variable = vect*(Y_plus_sigma_z) # t(y+sigma z)
                    second_variable = xt - first_variable # (1-t)s
                    xt = xt + dt * model(vect, first_variable, second_variable)
                elif mode_ == "noisemean_xt_y_sigmaz": #v_theta(xt,y,sigmaz)
                    xt = xt + dt * model(vect, xt, Y, sigma * z)
                elif mode_ == "noisemean_t_y": #v_theta(t,y)
                    xt = xt + dt * model(vect, Y)
        
        sample = xt.clone()
        
        
        sample = sample.squeeze()
        
        x_hat = model.to_audio(sample, T_orig)
        # print("완료")
        y = y * norm_factor
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        
      
        # Convert to numpy
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        n = y - x

        # Write enhanced wav file
        write(target_dir + "files/" + filename, x_hat, 16000)

        # Append metrics to data frame
        data["filename"].append(filename)
        try:
            p = pesq(sr, x, x_hat, 'wb')
        except: 
            p = float("nan")
        data["pesq"].append(p)
        data["estoi"].append(stoi(x, x_hat, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])

    # Save results as DataFrame
    df = pd.DataFrame(data)
    df.to_csv(join(target_dir, "_results.csv"), index=False)

    # Save average results
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w') as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))

    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        
        file.write("odesolver: {}\n".format(odesolver))
       
        file.write("N: {}\n".format(N))
        
        file.write("Reverse starting point: {}\n".format(reverse_starting_point))
        file.write("Reverse end point: {}\n".format(reverse_end_point))
        
        file.write("data: {}\n".format(args.test_dir))
        file.write("epoch: {}\n".format(epoch_number))
        # file.write("evaluationnumbers: {}\n".format(int_list_str))
        file.write("mode: {}\n".format(model.mode_))
        file.write('time_step: {}\n'.format(time_step_type))
        