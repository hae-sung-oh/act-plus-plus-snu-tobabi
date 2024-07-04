#### Modified by Haesung Oh
### Original repository:  
Act-plus-plus(모델 및 학습 관련 코드): https://github.com/MarkFzp/act-plus-plus

Mobile-Aloha (실제 로봇 컨트롤 코드): https://github.com/MarkFzp/mobile-aloha

### 설치
아래 [Installation](#installation) 항목 참고

설치 후 [변경점](#변경점) 항목의 wandb initialize를 위해 환경변수를 설정해주어야 한다.
```bash
$ export WANDB_USER_NAME='YOUR_USER_NAME'
# or
$ echo "export WANDB_USER_NAME='YOUR_USER_NAME'" >> ~/.bashrc
# then
$ source ~/.bashrc
```

### 데이터셋 구조
Mobile-aloha에서는 `hdf5`라는 데이터셋 포맷을 사용한다.

그 구조는 다음과 같다.

    episode_{i}.hdf5  (n: episode length, m: number of cameras)
    │
    ├── action (n x 14) 
    │
    ├── base_action (n x 2)
    |   
    ├── compress_len (m x n)
    |   
    ├── observations
    |   │
    |   ├── effort (n x 14)
    |   │
    |   └── images (m x n x enconded length)
    |       │
    |       ├── cam_name_1 (n x enconded length)
    |       │
    |       ├── cam_name_2 (n x enconded length)
    |       │   ...
    |       |
    |       └── cam_name_m (n x enconded length)
    |
    ├── qpos (n x 14)
    |
    └── qvel (n x 14)

* `action`:
    * 7 DOF arm joint space x2
    * Mobile-aloha에 주어진 사람의 supervising action y
    * Tocabi에서는 pretrained 된 액션이 될 듯

* `base_action`:
    * base velocity, base angular velocity
    * Tocabi에서는 없어도 될 듯

* `compress_len`
    * 각 이미지를 compress한 길이
    * OpenCV image encoding, decoding에 필요함
    * 데이터셋을 만들 때 compress 할지 말지 결정할 수 있음

* `observations/effort`:
    * 각 조인트마다 가해진 힘/토크

* `observations/image`:
    * m개의 카메라 존재
    * 데이터셋을 만들 때 m개의 카메라 이름 정해줘야 함
    * 학습 단계에서도 이 카메라 이름 리스트를 넣어줘야 함 (constant.py에서 넣어주기)

* `qpos`:
    * Mobile-aloha의 실제 joint position
    * Tocabi에서는 `action`과 동일하게 넣어주어도 될 듯

* `qvel`: 
    * Mobile-aloha의 실제 joint velocity

### 데이터셋 만들기
#### [create_episode.py](./create_episode.py) 참고
```python
# 파이썬 함수
create_episode(
    images: Optional[Union[List[ndarray], ndarray[int], Tensor[int], str]],
    qpos: Union[List[float], ndarray[float], Tensor[float], str],
    qvel: Union[List[float], ndarray[float], Tensor[float], str],
    effort: Union[List[float], ndarray[float], Tensor[float], str],
    action: Union[List[float], ndarray[float], Tensor[float], str],
    base_action: Union[List[float], ndarray[float], Tensor[float], str],
    image_root_dir: Optional[str],
    camera_names: Optional[List[str]],
    dataset_dir: str,
    dataset_name: str,
    overwrite: bool = False,
    no_compress: bool = False,
)
```
```bash
# 터미널 커맨드
$ python create_episode_module.py --qpos path/to/qpos.npy --qvel path/to/qvel.npy --effort path/to/effort.npy --action path/to/action.npy --base_action path/to/base_action.npy --dataset_dir path/to/dataset_dir --dataset_name my_dataset --overwrite --no_compress

```

- `images` (List[ndarray], ndarray[int], Tensor[int], str, or None):
    - 이미지 데이터

- `qpos` (List[float], ndarray[float], Tensor[float], or str):
    - 조인트 포지션 데이터

- `qvel` (List[float], ndarray[float], Tensor[float], or str):
    - 조인트 속도 데이터

- `effort` (List[float], ndarray[float], Tensor[float], or str):
    - 힘/토크 데이터

- `action` (List[float], ndarray[float], Tensor[float], or str):
    - 조인트 액션 데이터

- `base_action` (List[float], ndarray[float], Tensor[float], or str):
    - 베이스 액션 데이터

- `image_root_dir` (str or None):
    - 이미지들의 root directory: `images` argument를 무시하고 덮어씀 
    - 다음과 같이 폴더를 구성하고, root directory를 넘겨주면 데이터셋을 생성해줌
    ```
    image_root_dir
    │
    ├── cam_name_1
    │   ├── 0.jpg
    │   ├── ...
    │   └── n.jpg
    │
    ├── cam_name_2 
    │   ├── 0.jpg
    │   ├── ...
    │   └── n.jpg
    │   ...
    |
    └── cam_name_m 
        ├── 0.jpg
        ├── ...
        └── n.jpg
    ```
- `camera_names` (List[str] or None):
    - 카메라 이름 리스트
    - m개의 개수가 `images.shape[0]`, `image_root_dir`의 폴더 개수와 맞아야함

- `dataset_dir` (str):
    - 데이터셋이 저장될 폴더명

- `dataset_name` (str):
    - 데이터셋 이름 (`dataset_name.hdf5`로 저장됨)

- `overwrite` (bool):
    - 존재하는 데이터셋을 덮어쓰는 옵션 (기본값: False)

- `no_compress` (bool):
    - 이미지 파일을 압축하지 않는 옵션 (기본값: False, 압축함)

### 학습
```bash
# 사용예시
$ python3 imitate_episodes.py \
--task_name aloha_mobile_cabinet \
--ckpt_dir ckpt/ \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 100 \
--hidden_dim 512 \
--batch_size 8 \
--dim_feedforward 3200  \
--lr 1e-5 \
--seed 0 \
--num_steps 2000 \
--onscreen_render
```

### 변경점
- [imitate_episode.py](./imitate_episodes.py)
    - line 148
        
        ```python
        # wandb 사용자 변경: 환경변수 설정 필요
        # From
        wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name)
        # To
        wandb.init(project="mobile-aloha2", reinit=True, entity=os.getenv("WANDB_USER_NAME"), name=expr_name)
        ```
    - line 599, 600
        
        ```python
        # success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)
        # wandb.log({'success': success}, step=step)
        ```
        
        real_robot, simulation 둘 다 환경 설정 어려워 batch evaluation 생략
        
- [aloha_scripts](./aloha_scripts) 폴더 추가

- [detr/models/detr_vae.py](./detr/models/detr_vae.py)
    - line 285
        
        ```python
        # From
        encoder = build_transformer(args)
        # To
        encoder = build_encoder(args)
        ```
        
- [aloha_scripts/constants.py](./aloha_scripts/constants.py)
    
    ```python
    # config 예시
    DATA_DIR = os.path.expanduser('~/dataset/mobile_aloha') # 실제 데이터셋 위치로 변경
    TASK_CONFIGS = {
    		# tocabi task를 위한 config 추가
        'tocabi_peg_in_hole':{
            'dataset_dir': DATA_DIR + '/tocabi_peg_in_hole',
            'episode_len': 1000,
            'camera_names': ['cam_head'],
            'stats_dir': None,
            'sample_weights': None,
            'train_ration': 0.99,
        },
        
     # ...
    ```
    
- Evaluation, visualization
    - [eval.py](./eval.py) 작성
        - 생성된 action을 visualize하는 코드
        
    - [sim_env.py](./sim_env.py)
        - line 117-119
        
        ```python
        # obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # obs["images"]["left_wrist"] = physics.render(height=480, width=640, camera_id="left_wrist")
        # obs["images"]["right_wrist"] = physics.render(height=480, width=640, camera_id="right_wrist")
        obs["images"]["cam_high"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["cam_left_wrist"] = physics.render(height=480, width=640, camera_id="left_wrist")
        obs["images"]["cam_right_wrist"] = physics.render(height=480, width=640, camera_id="right_wrist")
        ```

- [create_episode.py](./create_episode.py) 작성
    * [데이터셋 만들기](#데이터셋-만들기) 참고


---
아래 부터는 original repository의 문서


</br>

# Imitation Learning algorithms and Co-training for Mobile ALOHA


#### Project Website: https://mobile-aloha.github.io/

This repo contains the implementation of ACT, Diffusion Policy and VINN, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate them in sim or real.
For real, you would also need to install [Mobile ALOHA](https://github.com/MarkFzp/mobile-aloha). This repo is forked from the [ACT repo](https://github.com/tonyzhaozh/act).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

- also need to install https://github.com/ARISE-Initiative/robomimic/tree/r2d2 (note the r2d2 branch) for Diffusion Policy by `pip install -e .`

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments (LEGACY table-top ALOHA environments)

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir <data save dir> --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the simulated episodes after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

Note: to visualize data from the mobile-aloha hardware, use the visualize_episodes.py from https://github.com/MarkFzp/mobile-aloha

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir <ckpt dir> --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.

### [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.
