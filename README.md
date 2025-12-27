# Federated Active Learning with Reference Point Personalized Calibration for Medical Image Segmentation


## Key Features

1. **FedAvg Algorithm**: Standard federated averaging algorithm
2. **Site Encoding**: Generate unique encoding vectors for each site
3. **DINO Object Detection**: Use pre-trained DINO network for image preprocessing and object detection
4. **Active Learning**: Support 5 rounds of manual annotation with automatic representative sample selection
5. **Clustering Algorithm**: Cluster decoder outputs and select representative samples
6. **Cluster Center Difference Calculation**: Calculate the difference of cluster centers before and after parameter upload

## File Structure

```
code/
├── train.py      # Main training file
├── test_ours.py        # Test file
├── runner_fedavg_dino_al.py      # Run script
└── utils/
    ├── site_encoder.py            # Site encoding module
    ├── dino_detector.py          # DINO object detection module
    ├── active_learning.py         # Active learning module
    └── clustering.py             # Clustering algorithm module
```

## Usage

### 1. Training

Start training using the run script:

```bash
python runner_fedavg_dino_al.py \
    --port 8080 \
    --exp fedavg_dino_al_experiment \
    --gpus 0 1 2 3 4 5 \
    --img_class faz \
    --model unet \
    --max_iterations 30000 \
    --iters 20 \
    --eval_iters 200 \
    --batch_size 5 \
    --al_rounds 5 \
    --al_samples_per_round 10 \
    --n_clusters 10
```

### 2. Testing

```bash
python trainl.py \
    --root_path ../data/FAZ_h5 \
    --exp fedavg_dino_al_experiment \
    --model unet \
    --client client1 \
    --img_class faz \
    --num_classes 2 \
    --in_chns 1
```

## Parameter Description

### Training Parameters
- `--port`: Communication port
- `--exp`: Experiment name
- `--gpus`: GPU index list (at least 6: 1 server + 5 clients)
- `--img_class`: Image type ('Prostrate', 'Fundus', 'polyp')
- `--model`: Model name ('unet', 'unet_lc', etc.)
- `--max_iterations`: Maximum number of iterations
- `--iters`: Number of local iterations per round
- `--eval_iters`: Evaluation interval
- `--batch_size`: Batch size
- `--al_rounds`: Number of active learning rounds (default: 5)
- `--al_samples_per_round`: Number of samples selected per round
- `--n_clusters`: Number of clusters

### Active Learning Strategies
- `uncertainty`: Select samples based on uncertainty
- `diversity`: Select samples based on diversity
- `cluster_center`: Select representative samples based on cluster centers (default)

## Workflow

1. **Initialization**:
   - Create site encoder to generate unique codes for each site
   - Initialize DINO detector
   - Initialize active learning manager
   - Initialize clusterer

2. **Training Loop**:
   - Server distributes global model parameters
   - Clients receive parameters and save cluster centers before upload
   - Clients preprocess images using DINO
   - Clients perform local training
   - Clients extract decoder features and perform clustering
   - Clients select representative samples for active learning
   - Clients compute cluster centers after upload
   - Clients calculate cluster center differences
   - Clients upload updated parameters

3. **Active Learning**:
   - After each training round, select samples according to strategy
   - Support 5 rounds of manual annotation
   - Annotated samples are added to the training set

4. **Clustering Analysis**:
   - Perform K-means clustering on decoder outputs
   - Select representative samples from each cluster
   - Calculate cluster center differences before and after parameter updates

## Notes

1. **DINO Model**: The DINO model will be automatically downloaded on first run, requiring network connection
2. **Active Learning Annotation**: Selected samples require manual annotation, and annotation data needs to be manually added to the dataset
3. **Cluster Center Difference**: Difference information is recorded in training metrics and can be used to analyze model convergence
4. **GPU Memory**: Ensure sufficient GPU memory, recommend at least 4GB per process

## Dependencies

- PyTorch >= 1.8.0
- Flower >= 1.0.0
- scikit-learn
- numpy
- torchvision
- PIL
- opencv-python

## Extended Features

- Support for custom active learning strategies
- Support for adaptive number of clusters
- Support for multiple clustering algorithms
- Support for site feature fusion
