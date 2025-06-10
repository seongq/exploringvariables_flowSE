import subprocess
import os
import re
import random
# 체크포인트 파일들이 있는 폴더 경로
time_steps = ['uniform', 'gerkmann']
gpu = input("gpu 0 or 1")
startpoint_types =['mean','noise']
while True:
    # "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_flowse_KDfromclean_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_wu3jmxyj"
    ckpt_folders= ["/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_CTFSE_KDfromclean_dataset_WSJ0-CHiME3_low_snr_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_9w30dujy", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_direct_estimation_xt_y_t_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_rxhmzv6r", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_CTFSE_KDfromclean_dataset_WSJ0-CHiME3_low_snr_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_qy0it00y", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_CTFSE_KDfromclean_dataset_WSJ0-CHiME3_low_snr_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_9w30dujy", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_ugfh3wbm", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_08ekodcs", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_k5diz2ec", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_qzyqjy43", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_qzyqjy43", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_ugfh3wbm", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_ugfh3wbm", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_qzyqjy43", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_k5diz2ec", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_sigmaz_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_mahk2d9v", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_ugfh3wbm",  "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_plus_sigmaz_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_hjimhuur", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_t_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_9uajbfsv", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_sigmaz_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_mahk2d9v", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_t_times_y_plus_sigmaz_1minust_times_s_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_7cv4vhio", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_t_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_fcj0edg6", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_t_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_fcj0edg6", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_t_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_ae3y764q", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_t_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_4zmkpazt", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xtplusy_divide_2_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_qubvep19", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_plus_sigmaz_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_88u92bli", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_qzyqjy43", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_k5diz2ec", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_xt_y_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_08ekodcs", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_y_plus_sigmaz_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_nc7ndoya", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_y_plus_sigmaz_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_u4rxkp4i", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_y_plus_sigmaz_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_3tk6bbj0", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_noxt_conditiony_timefalse_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_uz6yw8ox", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_noxt_conditiony_timefalse_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_8r536q1p", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_noxt_conditiony_timefalse_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_8r536q1p","/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_conditionfalse_timefalse_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_x7lhaqt6", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_conditionfalse_timefalse_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_plmzd2xl", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_conditionfalse_timefalse_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_ogp7nxna", "/workspace/flowse_condition_explor/flowse_KD_big/logs/mode_noisemean_conditionfalse_timefalse_dataset_VCTK_corpus_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_26tu468l"]
    # random.shuffle(ckpt_folders)
    for ckpt_folder in ckpt_folders:
        # 정규표현식으로 dataset 이름 추출
        match = re.search(r"dataset_(.*?)_sigma", ckpt_folder)
        if match:
            dataset_name = match.group(1)
            print(f"Extracted dataset name: {dataset_name}")
        else:
            print("Dataset name not found in the path.")

        test_dir = f"/workspace/datasets/{dataset_name}"
        int_lists = ["1","2", "3", "4", "5"] # int_list 값들
        random.shuffle(int_lists)
        # ckpt 폴더에서 모든 .ckpt 파일 찾기
        ckpt_files = sorted([f for f in os.listdir(ckpt_folder) if f.endswith(".ckpt")])
        random.shuffle(ckpt_files)
        # 실행할 명령어 생성 및 실행
        for N in int_lists:
            for ckpt_file in ckpt_files:
                ckpt_path = os.path.join(ckpt_folder, ckpt_file)

                random.shuffle(startpoint_types)
                for time_step_type in time_steps:
                    for startpoint_type in startpoint_types:
                        cmd = f"CUDA_VISIBLE_DEVICES={gpu} python evaluate_cascading.py --ckpt {ckpt_path} --test_dir {test_dir} --N {N} --time_step_type {time_step_type} --startpoint_type {startpoint_type}"
                        print(f"Executing: {cmd}")
                        
                        process = subprocess.run(cmd, shell=True)

                        if process.returncode != 0:
                            print(f"Command failed: {cmd}")
                            break  # 실패 시에만 반복 종료
                        
                    # random.shuffle(int_lists)
