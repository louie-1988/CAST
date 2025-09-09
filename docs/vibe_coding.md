You are an expert in Python, Deep Learning and Generative Models. Now I am re-implementing a generative algorithm which aims at component-aligned 3D scene reconstruction from a single RGB Image, according to the Paper https://arxiv.org/pdf/2502.12894. 
Now I need you implement the algoirthm in the current codebase. Now I will descibe each stage/part of the pipeline as follows, please notice that some of them are based on off-the-shelf API providers (e.g. Tripo/Replicate/Qwen)

1. Open-Vocabulary Object Detection, Caption, and Segmentation
In this step, use the florence-2 to identify objects, generate their descriptions, and localize each object with bounding boxes from the input scene image. After that, use GroundedSAM to generate the mask for each object. Both of the models are based on replicate.
Florence-2: https://replicate.com/lucataco/florence-2-large
Grounded-SAM: https://replicate.com/schananas/grounded_sam

Inputs: A single RGB scene image 
Outputs: Multiple objects, each associated with a bounding box, a description and a segmentation mask
Where to run: use existing API from replicate

2. Inpainting 
For each object, crop the image and mask. Then do image-inpainting to recover the occluded region using stable diffusion inpainting algorithm. 
Replicate API: https://replicate.com/stability-ai/stable-diffusion-inpainting

Inputs: A cropped RGB image and binary mask of each object 
Outputs: Inpainted RGB image of each object
Where to run: use existing API from replicate

3. Metric-Level Depth Estimation 
For the scene image, generate a metric-space depth map and point cloud. Use the algoirthm MoGe: https://github.com/microsoft/MoGe, notice that this algorithm should be run locally

Inputs: The single RGB scene image 
Outputs: A metric-space depth map and a point cloud map 
Where to run: local

4. 3D Object Generation 
In this step, for each cropped and inpainted image, you should use Tripo3D API to generate a textured mesh model. Here Tripo3D API takes a single RGB image as the input and generates a high-fidelity mesh with texture. Refer to the doc of Tripo3D here https://platform.tripo3d.ai/docs/generation.

Inputs: the (inpainted) RGB image(s) of each object 
Outputs: the textured 3D mesh model of each object 
Where to run: use existing API from tripo3d

5. Point Cloud Registration and Object Pose Estimation 
In this step, use the robust ICP algoirthm to register each generated 3D object mesh to the scene point cloud. Specifically, first sample the point cloud from the generated mesh, then get the partial point cloud of that object from the point cloud of the scene, finally use Open3D ICP with robust kernels and global registration to get the 6D pose of each object.

Inputs: the estimated point cloud and generated 3D mesh of each object (taken out from the scene point cloud), 
Outputs: the 6D pose of the registered mesh 
Where to run: run locally with open3d

6. Scene Graph Extraction and Physical Correctness
In this step, first use multi-modal visual-language-model to extract the scene graph from the large image of the scene. The scene graph should descibe the contact/support relationships between each pair of object in the scene. Then optimize (finetune) the 6D poses of each object(initialized with previous registration) so that the relationships between all the scene objects are physically plausiable (e.g. no penetration and the contact is correct). The idea and algorithm is to use SDF(Open3D and PyTorch) to penalize penetration and ensure contact is correct. Refer to Section 5 of the paper for specific details. And refer to @ for the prompt to query VLM to get the scene graph.
Qwen-VL DOC: https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.7eaa7b08LNcQj4&tab=doc#/doc/?type=model&url=2845871

Inputs: The RGB scene image and 3D objects with initial 6D poses
Outputs: The scene graph, and 3D objects with physically plausiable 6D poses
Where to run: use existing API from Qwen-VL and run SDF optimization locally

Requirements: 
1. Notice that Replicate/Qwen/Tripo API providers all require API keys, setup them in the .env variables. Use playwright mcp to read the docs and the paper if needed.
2. I have provided a number of examples at @api_examples for each module
3. Well orgnaize the codebase ane ensure that it's scalable and extensiable, for example write some wrappers for replicate calling service

Make good plans before you begin, it's a very complicated task and implement the algorithm step-by-step. You are an expert machine learning engineer, don't hold back and give it your all.
