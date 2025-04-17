import subprocess
import os
import re
# 체크포인트 파일들이 있는 폴더 경로

while True:
    ckpt_folders= ["/workspace/general_cascading/logs/dataset_WSJ0-CHiME3_mode_cn_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_a0tkvwsh"]

    for ckpt_folder in ckpt_folders:
        # 정규표현식으로 dataset 이름 추출
        match = re.search(r"dataset_(.*?)_mode", ckpt_folder)
        if match:
            dataset_name = match.group(1)
            print(f"Extracted dataset name: {dataset_name}")
        else:
            print("Dataset name not found in the path.")

        test_dir = f"/workspace/database/{dataset_name}"
        int_lists = ["1", "2", "3", "4", "5", "6", "7","8","9","10"] # int_list 값들

        # ckpt 폴더에서 모든 .ckpt 파일 찾기
        ckpt_files = sorted([f for f in os.listdir(ckpt_folder) if f.endswith(".ckpt")])

        # 실행할 명령어 생성 및 실행
        for ckpt_file in ckpt_files:
            ckpt_path = os.path.join(ckpt_folder, ckpt_file)

            for int_list in int_lists:
                cmd = f"CUDA_VISIBLE_DEVICES=5 python evaluate_cascading.py --ckpt {ckpt_path} --test_dir {test_dir} --int_list {int_list}"
                print(f"Executing: {cmd}")
                
                process = subprocess.run(cmd, shell=True)

                if process.returncode != 0:
                    print(f"Command failed: {cmd}")
                    break  # 실패 시에만 반복 종료
