# TransE Knowledge Graph Embeddings with LC-QuAD

A collaborative project implementing TransE (Translating Embeddings) for knowledge graph embeddings using PyTorch, with evaluation on the LC-QuAD dataset for question answering tasks.

## Overview

This project combines:
- **TransE**: A method for learning embeddings of entities and relations in knowledge graphs
- **LC-QuAD**: A gold standard dataset for question answering over knowledge graphs

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # TransE model implementation
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training scripts and utilities
│   └── evaluation/        # Evaluation metrics and scripts
├── data/                  # Dataset files (LC-QuAD, knowledge graphs)
├── experiments/           # Experiment configurations and results
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── requirements.txt       # Python dependencies
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA (optional, for GPU training)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd TransE-Model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Setup

1. Download LC-QuAD dataset:
```bash
# Instructions for downloading and setting up LC-QuAD data
```

2. Prepare knowledge graph data:
```bash
# Instructions for preparing KG data
```

## Usage

### Training TransE Model
```bash
python src/training/train_transe.py --config experiments/config.yaml
```

### Evaluation
```bash
python src/evaluation/evaluate.py --model_path models/transe_model.pt
```

## Contributing

We welcome contributions from all team members! Please follow these guidelines:

1. **Branching Strategy**: 
   - `main`: Stable, production-ready code
   - `develop`: Integration branch for features
   - `feature/your-feature-name`: Individual feature branches

2. **Before Starting Work**:
   - Pull latest changes: `git pull origin main`
   - Create a new branch: `git checkout -b feature/your-feature-name`

3. **Code Standards**:
   - Follow PEP 8 for Python code
   - Add docstrings to all functions and classes
   - Write unit tests for new features
   - Run tests before committing: `pytest tests/`

4. **Committing Changes**:
   - Use descriptive commit messages
   - Reference issues when applicable: `fixes #123`

5. **Pull Requests**:
   - Create PR from your feature branch to `develop`
   - Add description of changes
   - Request review from team members

## Team Members

- [Add team member names and roles here]

## References

- [TransE-PyTorch Implementation](https://github.com/mklimasz/TransE-PyTorch)
- [LC-QuAD Dataset](https://github.com/AskNowQA/LC-QuAD/tree/data)
- TransE Paper: "Translating Embeddings for Modeling Multi-relational Data" (Bordes et al., 2013)

## License

[Choose appropriate license]

## Issues and Support

Please use GitHub Issues for bug reports, feature requests, and questions. 