# TransE-KGQA: Knowledge Graph Question Answering with TransE

A collaborative project extending the [TransE-PyTorch implementation](https://github.com/mklimasz/TransE-PyTorch) for knowledge graph question answering using the LC-QuAD dataset.

## Overview

This project combines:
- **TransE-PyTorch**: Original implementation of TransE (Translating Embeddings) for knowledge graph embeddings
- **LC-QuAD**: A gold standard dataset for question answering over knowledge graphs
- **KGQA Extension**: Our collaborative extension for knowledge graph question answering tasks

## Original TransE Implementation

This project is based on the TransE-PyTorch implementation by [mklimasz](https://github.com/mklimasz/TransE-PyTorch), which provides:
- PyTorch implementation of TransE model from Bordes et al. (2013)
- Training and evaluation on knowledge graph datasets
- Synthetic data for testing and development

## Project Structure

```
├── main.py                    # Main training script (from original repo)
├── model.py                   # TransE model implementation
├── data.py                    # Data loading utilities
├── metric.py                  # Evaluation metrics
├── storage.py                 # Model storage utilities
├── synth_data/                # Synthetic data for testing
├── data/                      # LC-QuAD and other datasets (to be added)
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
└── .gitignore                 # Git ignore patterns
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA (optional, for GPU training)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Karthik-Pydishetty/TransE-KGQA.git
cd TransE-KGQA
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

## Usage

### Quick Test with Synthetic Data
```bash
python main.py --nouse_gpu
```

### Training on Custom Dataset
```bash
python main.py --nouse_gpu --dataset_path=<path_to_your_dataset>
```

### Command Line Options
```bash
python main.py --help
```

### Unit Tests
```bash
python -m unittest discover -p "*_test.py"
```

## LC-QuAD Integration (Planned)

We will extend this implementation to work with the [LC-QuAD dataset](https://github.com/AskNowQA/LC-QuAD/tree/data) for:
- Question answering over knowledge graphs
- Evaluation on KGQA tasks
- Integration with SPARQL query generation

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
   - Run tests before committing: `python -m unittest discover -p "*_test.py"`

4. **Committing Changes**:
   - Use descriptive commit messages
   - Reference issues when applicable: `fixes #123`

5. **Pull Requests**:
   - Create PR from your feature branch to `develop`
   - Add description of changes
   - Request review from team members

## Team Members

- [Add team member names and roles here]

## Original Results (TransE-PyTorch)

### FB15k Dataset

| Source/Metric  | Hits@1 (raw) | Hits@3 (raw) | Hits@10 (raw) | MRR (raw) |
| ---------------| ------------ | ------------ | ------------- | --------- |
| Paper [[1]](#references) | X | X | 34.9 | X |
| TransE-PyTorch | 11.1 | 25.33 | **46.53** | 22.29 |

## References

- **Original Implementation**: [TransE-PyTorch](https://github.com/mklimasz/TransE-PyTorch)
- **LC-QuAD Dataset**: [LC-QuAD](https://github.com/AskNowQA/LC-QuAD/tree/data)
- **TransE Paper**: [Bordes et al., "Translating embeddings for modeling multi-relational data," NIPS 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)

## License

Based on the original TransE-PyTorch implementation. [Choose appropriate license]

## Issues and Support

Please use GitHub Issues for bug reports, feature requests, and questions. 