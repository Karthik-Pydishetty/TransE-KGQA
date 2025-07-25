# TransE Knowledge Graph Question Answering (KGQA)

A comprehensive implementation of the TransE model for Knowledge Graph Question Answering, integrated with the LC-QuAD dataset for compliance research applications.

## 🚀 Quick Start for Collaborators

### 1. Clone the Repository (Including Dataset)
```bash
# Option 1: Clone with dataset in one command (Recommended)
git clone --recurse-submodules https://github.com/Karthik-Pydishetty/TransE-KGQA.git
cd TransE-KGQA

# Option 2: Clone first, then get dataset
git clone https://github.com/Karthik-Pydishetty/TransE-KGQA.git
cd TransE-KGQA
git submodule update --init --recursive
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Dataset Access
```bash
# Check dataset is available
ls data/LC-QuAD/
# Should show: train-data.json, test-data.json, resources/, etc.
```

### 4. Run the Model
```bash
python main.py
```

## 📊 Dataset Information

This project includes the **LC-QuAD dataset** as a Git submodule:
- **Location**: `data/LC-QuAD/`
- **Size**: 5,000 question-SPARQL query pairs
- **Format**: JSON files with natural language questions and corresponding SPARQL queries
- **Source**: [AskNowQA/LC-QuAD](https://github.com/AskNowQA/LC-QuAD)

## 🔧 Development Workflow

### Working with the Dataset
The LC-QuAD dataset is included as a Git submodule. This means:
- ✅ Everyone gets the same dataset version
- ✅ Dataset updates are tracked and versioned
- ✅ No need to manually download large files

### Updating the Dataset (Advanced)
If you need to update to a newer version of LC-QuAD:
```bash
cd data/LC-QuAD
git pull origin main
cd ../..
git add data/LC-QuAD
git commit -m "Update LC-QuAD dataset"
git push origin main
```

## 📁 Project Structure

```
TransE-KGQA/
├── data/
│   └── LC-QuAD/              # LC-QuAD dataset (submodule)
│       ├── train-data.json   # Training data (5000 examples)
│       ├── test-data.json    # Test data
│       └── resources/        # Entities, predicates, templates
├── synth_data/               # Synthetic training data
├── runs/                     # Training logs and outputs
├── main.py                   # Main training script
├── model.py                  # TransE model implementation
├── data.py                   # Data loading and preprocessing
├── metric.py                 # Evaluation metrics
├── storage.py                # Model persistence utilities
├── checkpoint.tar            # Pre-trained model weights
└── requirements.txt          # Python dependencies
```

## 🎯 Research Focus

This implementation focuses on:
- **Knowledge Graph Embeddings**: TransE model for entity and relation representation
- **Question Answering**: Natural language to SPARQL query translation
- **Compliance Research**: Applications in regulatory and compliance domains
- **Collaborative Development**: Easy setup for team-based research

## 📚 Key Features

- ✅ Complete TransE implementation with training and evaluation
- ✅ LC-QuAD dataset integration for realistic QA scenarios  
- ✅ Pre-trained model checkpoints for quick experimentation
- ✅ Comprehensive evaluation metrics
- ✅ Easy collaborative setup with Git submodules

## 🤝 Contributing

1. Clone the repository with submodules (see Quick Start)
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m "Description"`
5. Push to your branch: `git push origin feature-name`
6. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

The LC-QuAD dataset included as a submodule has its own licensing terms - see `data/LC-QuAD/LICENSE.txt`.

---

**Note**: This repository uses Git submodules for dataset management. Always use `--recurse-submodules` when cloning to ensure you get the complete project including the dataset.

## 🔧 Troubleshooting

### Problem: Empty `data/LC-QuAD/` folder after pulling
If you pulled changes but the `data/LC-QuAD/` folder is empty, run:

```bash
# Initialize and update all submodules
git submodule update --init --recursive

# Or alternatively, update to latest remote version
git submodule update --remote
```

### 🚨 COMPREHENSIVE DEBUGGING FOR ARJUN (Step-by-Step)

If the above doesn't work, follow these detailed steps:

#### Step 1: Check Current Status
```bash
# Check what branch you're on
git branch

# Check if you're in the right directory
pwd
# Should show: .../TransE-KGQA (or your repo name)

# Check submodule status
git submodule status
```

#### Step 2: Verify Submodule Configuration
```bash
# Check if .gitmodules exists
cat .gitmodules
# Should show:
# [submodule "data/LC-QuAD"]
#     path = data/LC-QuAD
#     url = https://github.com/AskNowQA/LC-QuAD.git
```

#### Step 3: Force Submodule Initialization (Try These in Order)

**Option A: Clean slate approach**
```bash
# Remove existing (empty) submodule directory
rm -rf data/LC-QuAD

# Re-initialize submodule
git submodule update --init --recursive
```

**Option B: If Option A doesn't work**
```bash
# Deinitialize submodule completely
git submodule deinit data/LC-QuAD

# Re-add and initialize
git submodule update --init data/LC-QuAD
```

**Option C: Manual clone (last resort)**
```bash
# Remove empty directory
rm -rf data/LC-QuAD

# Manually clone the dataset
git clone https://github.com/AskNowQA/LC-QuAD.git data/LC-QuAD

# Re-initialize submodule tracking
git submodule update --init
```

#### Step 4: Verify Success
```bash
# Check if files are now present
ls data/LC-QuAD/
# Should show: train-data.json, test-data.json, resources/, README.md, LICENSE.txt

# Check file sizes to confirm they're not empty
du -sh data/LC-QuAD/
# Should show: ~97M data/LC-QuAD/
```

#### Step 5: If STILL not working - Network/Permission Issues
```bash
# Test if you can access the repository
curl -I https://github.com/AskNowQA/LC-QuAD.git

# Try with SSH instead of HTTPS (if you have SSH key set up)
git config --global url."git@github.com:".insteadOf "https://github.com/"
git submodule update --init --recursive
```

### Problem: Submodule not updating to latest version
To pull the latest changes for all submodules:

```bash
# Update all submodules to latest commit on their remote branches
git submodule update --remote

# Then commit the updated submodule references
git add .
git commit -m "Update submodules to latest version"
git push origin your-branch
```

### Daily Workflow for Team Members
When pulling changes from main:

```bash
# 1. Pull latest changes
git pull origin main

# 2. Update submodules (IMPORTANT!)
git submodule update --init --recursive

# 3. Verify dataset is available
ls data/LC-QuAD/  # Should show train-data.json, test-data.json, etc.
``` 