# MS1 Status

## Dataset Existence
| Dataset | Path | Exists |
|---|---|---|
| FAUST | `data/datasets/MPI-FAUST` | OK |
| ShapeNet | `data/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal` | OK |
| ModelNet40 | `data/datasets/ModelNet40` | OK |
| ModelNet10 | `data/datasets/ModelNet10` | OK |
| COSEG | `data/datasets/CoSeg` | OK |
| SMAL | `data/datasets/smal_online_V1.0` | OK |
| Human3.6M | `data/datasets/Human3.6M` | MISSING |

> TODO: Human3.6M dataset still absent; GEM remains disabled until the licensed data is available.

## Subjects Configuration Snapshot
| Subject | Enabled | Env | Dataset Keys | Entry Templates |
|---|---|---|---|---|
| MeshCNN | yes | meshcnn | ShapeNet,FAUST | classification: python train.py --dataroot ${DATASET:ShapeNet} --name ms1_meshcnn_cls --niter 1 --niter_decay 0 --save_epoch_freq 1 --batch_size 2<br>segmentation: python train.py --dataroot ${DATASET:FAUST} --name ms1_meshcnn_seg --niter 1 --niter_decay 0 --save_epoch_freq 1 --batch_size 2 |
| HodgeNet | yes | HodgeNet | ModelNet40 | eval: python train_classification.py --out runs/ms1_eval --n_epochs 100 --bs 16 --data ${DATASET:ModelNet40} |
| MeshSDF | yes | meshsdf | ShapeNet | svr: python train_svr.py -e experiments/cars_svr --max_epoch 1 --batch_size 2 --data_root ${DATASET:ShapeNet} |
| Point2Mesh | yes | point2mesh | COSEG | infer: python point2mesh.py --input ${DATASET:COSEG}/chair/chair_001.obj --output_dir workdir/COSEG/Point2Mesh/results --iters 10 |
| MeshWalker | yes | meshwalker | COSEG | eval: python train_val.py coseg chairs --epochs 1 --data_root ${DATASET:COSEG} |
| DeepGCNs | yes | deepgcn | ModelNet40 | eval: python examples/modelnet_cls/main.py --cfg examples/modelnet_cls/config.yaml --epochs 1 --batch_size 8 --data_dir ${DATASET:ModelNet40} |
| GEM | no | gem | Human3.6M | eval: CUDA_VISIBLE_DEVICES=0 python evaluate_scannet.py --model bagem_scannet --batch_size 2 --model_path checkpoints/bagem_scannet.ckpt --with_rgb --data_root ${DATASET:Human3.6M} |
| SpiderCNN | no | spidercnn-tf1 | ModelNet10 | eval: python train.py --eval --data_root ${DATASET:ModelNet10} |

Notes:
- GEM disabled pending Human3.6M download/licensing.
- SpiderCNN disabled; requires TF1.3 + CUDA 8 + Python 2.7 and `modelnet40_ply_hdf5_2048` under `ModelNet40`.

## Smoke Test Results
| Subject | Command (condensed) | Result | Log | Next Action |
|---|---|---|---|---|
| MeshCNN | `python train.py --dataroot datasets/shrec_16 --name ms1_smoke --niter 1 --niter_decay 0 --batch_size 2 --gpu_ids -1 --num_threads 0` | ✅ Runs; loss printed, model saved | `logs/meshcnn_smoke.log` | Restore GPU config when running full MS1; current smoke uses CPU + tiny dataset slice. |
| HodgeNet | `python train_classification.py --out runs/ms1_eval --n_epochs 1 --bs 2 --data data/shrec` | ✅ Runs on SHREC sample | `logs/hodgenet_smoke.log` | For ModelNet40, preprocess to produce `labels.txt` and splits (repo script). Update entry once processed. |
| MeshSDF | `python train_deep_sdf.py -e experiments/bob_and_spot` (timeout after 10 min) → `python demo_optimizer.py -e ... --fast` | ⚠️ Long-running (still printing epochs; no early stop) | `logs/meshsdf_smoke.log` | Use CPU-friendly spec or reduce `NumEpochs` in `experiments/bob_and_spot/specs.json`; verify CUDA before full run. |
| Point2Mesh | `bash scripts/examples/giraffe.sh` | ❌ Fails: `ImportError: libtorch_cpu.so` (PyTorch3D missing) | `logs/point2mesh_smoke.log` | Install PyTorch3D wheels matching torch/cu11.x (`pip install pytorch3d==0.7.2 -f https://dl.fbaipublicfiles...)` and ensure Manifold build present. |
| MeshWalker | `python train_val.py coseg chairs --epochs 1 --data_root ../../data/datasets/CoSeg` | ❌ Fails: `DATASET error` (no processed `.npz`) | `logs/meshwalker_smoke.log` | `dataset_prepare.py` created empty processed folders; populate COSEG meshes per README or point to prepared `.npz`. |
| DeepGCNs | `python examples/modelnet_cls/main.py --epochs 1 --batch_size 8 --data_dir ../../data/datasets/ModelNet40` | ❌ Fails: download of `modelnet40_ply_hdf5_2048` blocked → empty data | `logs/deepgcn_smoke.log` | Manually place `modelnet40_ply_hdf5_2048` under `ModelNet40` or set `--data_dir` to existing preprocessed folder; re-run. |

## TODO / Follow-ups
- Download Human3.6M and update `configs/datasets.yml`; flip `GEM.enabled` once data is licensed.
- Populate ModelNet40 with `modelnet40_ply_hdf5_2048` or adjust SpiderCNN entry to pass explicit path.
- HodgeNet: preprocess ModelNet40 to produce `labels.txt` (per repo instructions) or update command to point at processed data.
- MeshSDF: revisit experiment specs and argument expectations; run provided preprocessing to create splits and JSON configs.
- Point2Mesh: install PyTorch3D (matching CUDA/PyTorch build) or use Docker image; verify Manifold binaries path in `options.py`.
- MeshWalker: pin `protobuf<=3.20.x` (or set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) in `meshwalker` environment.
- DeepGCNs: install `torch_cluster` (matching PyTorch/CUDA) and confirm ModelNet40 pre-processing.
- SpiderCNN: provision TF1.3/CUDA8 environment or container when ready.
- Automate dataset path placeholder replacement in `SubjectRunner` to cover multi-dataset references beyond the active dataset.

### Point2Mesh (CPU) smoke
- env: point2mesh (torch 2.3.0+cpu / torchvision 0.18.0+cpu / pytorch3d 0.7.6)
- cmd: `python main.py --name ms1_smoke_p2m_cpu --input ./data/examples/giraffe.ply --iters 10 --save_freq 10 --gpu_ids -1`
- device: CPU, workdir `ms1/workdir/COSEG/Point2Mesh`
- duration ≈1.84s, exit code 2 (script rejects `--name/--iters` flags)
- TODO: swap in COSEG-derived input + proper CLI flags once migrating off the toy example.
