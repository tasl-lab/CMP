## Visualizations

The prediction results will be saved in `MTR/output/{prediction_model_name}/default/eval/inference_results`.

The visulization results will be saved in `Plotter/visualizations`.

Plot and save figure of predicted trajecotories and GT trajecotories for each timestamp.

CMP (ours)

```
python Plotter/main.py --prediction_model_name cmp --dataset opv2v --dataset_path /data1/Datasets/OPV2V/test
```

No Cooperation

```
python Plotter/main.py --prediction_model_name no_coop --dataset opv2v --dataset_path /data1/Datasets/OPV2V/test
```

Cooperative Perception Only

```
python Plotter/main.py --prediction_model_name cooperative_perception_only --dataset opv2v --dataset_path /data1/Datasets/OPV2V/test
```

V2VNet

```
python Plotter/main.py --prediction_model_name v2vnet --dataset opv2v --dataset_path /data1/Datasets/OPV2V/test
```

