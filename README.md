
This repo shows an unofficial and partial implementation of SIGGRAPH 2025 Best Paper Nominate [CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image](https://arxiv.org/abs/2502.12894). 

To simplify the pipeline, almost all modules of the system are based on existing API (e.g. Replicate/Qwen/Kontext/Tripo3D), making it easy to be deployed on Windows/Linux/MacOS.

### Workflow


### Differences with the Paper 
| Modules | Original Paper | This Repo |
| --- | --- | --- |
| Detection and Caption | Florence2    |  RAM |
| Segmentation          | Grounded-SAM |  RAM | 
| Detection Filtering   |  GPT-4       |  Qwen-VL | 
| Depth Estimation & PointCloud | MoGev1 |  MoGev2 | 
| Mesh Generation       | occlusion-aware self-trained 3D Generative model| Kontext + Tripo3D/TRELLIS | 
| Pose Registration     | occlusion-aware self-trained 3D Generative model | ICP / DR | 
| Physical Post-Processing | SceneGraph guided SDF | TBD |
----
1. 


### Quick Start 
#### 1. Setup 
``` shell 
conda create -n cast python=3.11 -y
conda activte cast 
python -m pip insatll -r requirements.txt 
```

### TODO 
- [ ] what 


### Comparison with Existing Works
1. [MIDI-3D](https://github.com/VAST-AI-Research/MIDI-3D)
2. [PartCrafter](https://github.com/wgsxm/PartCrafter)
3. [ArtiScene](https://github.com/NVlabs/ArtiScene)
4. [ReconViaGen](https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen)