## Dataset and Checkpoint Setup

[OPV2V Dataset Preparation](https://github.com/DerrickXuNu/OpenCOOD/blob/main/docs/md_files/data_intro.md#opv2v)

[V2V4Real Dataset Prepation](https://mobility-lab.seas.ucla.edu/v2v4real/)



1. Firstly, make sure the dataset path in `opencood/hypes_yaml/*` are correctly configured. Those config files are used for perception model training and testing.

   ```yaml
   root_dir: '/data1/Datasets/OPV2V/train/' # !!Change this to your own path!!
   validate_dir: '/data1/Datasets/OPV2V/test/' # !!Change this to your own path!!
   ```

   ```yaml
   root_dir: '/data1/Datasets/V2V4Real/train/' # !!Change this to your own path!!
   validate_dir: '/data1/Datasets/V2V4Real/test' # !!Change this to your own path!!
   ```

   

2. Secondly, make sure the dataset path in MTR/tools/cfg/* are correctly configured. Those config files are used for prediction model training and testing.

   ```yaml
   train_dir: '/data1/Datasets/OPV2V/train/'  # !!Change this to your own path!!
   validate_dir: '/data1/Datasets/OPV2V/test/'  # !!Change this to your own path!!
   additional_dir: '/data1/Datasets/OPV2V/additional/'  # !!Change this to your own path!!
   ```

   ```yaml
   train_dir: '/data1/Datasets/V2V4Real/train/'  # !!Change this to your own path!!
   validate_dir: '/data1/Datasets/V2V4Real/test/'  # !!Change this to your own path!!
   ```

   

3. Next, we gonna config the checkpoint path for perception module. The perception model can be either CoBEVT or V2VNet for two different datasets, the checkpoint can be downloaded from [HERE](https://drive.google.com/drive/folders/1EizY6ZFMi__HnqeFPQ2Wf9yRJeD_-S82?usp=drive_link). For each pretrained model folder, there should be one net_epochXX.pth file with a config.yaml. After you get the checkpoint. Move them into pretrained folder like below structure. Also, make sure change `root_dir` and `validate_dir` to your dataset path in config.yaml. Moreover, a pretrained Swim Transformer for prediction module on OPV2V dataset is also saved here.

   ```bash
   ├── pretrained
   |   |── opv2v
           |── corpbevtlidar_delay_1_frame_aug
              |── config.yaml
              |── net_epoch25.pth
           |── corpbevtlidar_delay_1_frame_aug_c256
              |── config.yaml
              |── net_epoch25.pth
           |── point_pillar_v2vnet_multiego
              |── config.yaml
              |── net_epoch83.pth
           |── point_pillar_sinbevt
              |── config.yaml
              |── net_epoch30.pth
           |── swin-base-patch4-window7-224
       |── v2v4real
           |── point_pillar_cobevt_multiego_1x
              |── config.yaml
              |── net_epoch75.pth
           |── point_pillar_cobevt_multiego_256x
              |── config.yaml
              |── net_epoch75.pth
           |── point_pillar_v2vnet_multiego
              |── config.yaml
              |── net_epoch75.pth
           |── point_pillar_sinbevt
              |── config.yaml
              |── net_epoch60.pth
   ```



4. Finally, we gonna config the checkpoint for prediction module, in which we need a checkpoint of no cooperation model, a cooperative perception only model (without prediction aggregation), our cooperative perception and prediction model, and v2vnet baseline. Download from [HERE](https://drive.google.com/drive/folders/1ZUJ5a5VuNfxV34I9FmIefHDGixaJ7gM2?usp=drive_link) and move them into `MTR/output` folder.

   ```
   ├── MTR
   |   |── output
           |── opv2v_multiego_cobevt_c256 (our cmp)
           |── opv2v_multiego_cobevt_c256_no_agg (cooperative perception only)
           |── opv2v_multiego_no_coop (no cooperation)
           |── opv2v_multiego_v2vnet (v2vnet baseline)
           |── v2v4real_multiego_cobevt_c256
           |── v2v4real_multiego_cobevt_c256_no_agg
           |── v2v4real_multiego_no_coop
           |── v2v4real_multiego_v2vnet
   ```



5. We provide the tracking label and motion prediction GT in the repo.
