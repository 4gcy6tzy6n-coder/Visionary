# Visionary - 3D Point Cloud Localization System

## Overview

Visionary is an advanced 3D point cloud localization system that enables precise position localization within 3D point cloud maps using natural language descriptions. This is the official implementation of our enhanced Text2Loc system with advanced natural language understanding capabilities.

## Key Features

- **Natural Language Localization**: Describe a location in natural language, and the system will find the corresponding position in the 3D point cloud map
- **Enhanced NLU Integration**: Advanced natural language understanding using state-of-the-art embedding models
- **Hybrid Retrieval**: Combines template matching and vector similarity for robust localization
- **REST API**: Complete API interface for easy integration into other systems
- **Modular Architecture**: Clean separation between core algorithms, API services, and frontend

## Project Structure

```
Visionary/
├── text2loc_visionary/     # Core localization algorithms
│   ├── dataloading/        # Data loading modules
│   ├── datapreparation/    # Data preparation utilities
│   ├── models/             # Neural network models
│   └── training/           # Training scripts
├── api/                    # REST API service
│   ├── text2loc_api.py     # Main API implementation
│   └── text2loc_adapter.py # Adapter for core algorithms
├── enhancements/          # Advanced features
│   ├── nlu/               # Natural language understanding
│   └── integration/       # System integration
├── frontend/              # Web interface
└── deployment/           # Docker deployment configurations
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/4gcy6tzy6n-coder/Visionary.git
cd Visionary

# Install dependencies
pip install -r requirements.txt
```

### Running the API Service

```bash
# Start the API server
python -m api.server --port 8080
```

### Using the API

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I am standing about 5 meters north of the red building",
    "top_k": 5
  }'
```

## Architecture

### Core Components

1. **Text2Loc Visionary Core**: The main localization engine that processes natural language queries and matches them against 3D point cloud maps
2. **Enhanced NLU Module**: Advanced natural language understanding using embedding models for semantic similarity
3. **Vector Retrieval System**: Fast and accurate location retrieval using vector databases
4. **API Layer**: RESTful interface for easy integration

### Technology Stack

- **Backend**: Python, Flask
- **Frontend**: React, Three.js
- **NLP**: Sentence Transformers, Ollama
- **Vector Database**: FAISS or compatible
- **Deployment**: Docker, Docker Compose

## Datasets

This project uses the KITTI360Pose dataset for training and evaluation. Please refer to the original Text2Loc repository for dataset preparation instructions.

## Related Projects

- [Text2Loc Original](https://github.com/Yan-Xia/Text2Loc) - CVPR 2024 paper implementation
- [Visionary Data](https://github.com/4gcy6tzy6n-coder/Visionary-data) - Experimental data and scripts

## License

This project is for research purposes only.

## Citation

If you use this code, please cite the original Text2Loc paper:

```bibtex
@article{xia2023text2loc,
  title={Text2Loc: 3D Point Cloud Localization from Natural Language},
  author={Xia, Yan and others},
  journal={CVPR},
  year={2024}
}
```
