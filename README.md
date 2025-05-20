# 🦀CRAB: Multi-Label Bayesian Active Learning with Inter-Label Relationships
This is the official implementation of CRAB (CoRrelation-Aware active learning with Beta scoring rules), as proposed in: Multi-Label Bayesian Active Learning with Inter-Label Relationships ([paper link](https://arxiv.org/abs/2406.09008)).


## ⚙️ Installation
```
conda env create -f environment.yml
conda activate your-env-name
```

## 💡 Usage
To run the code with the demo data on **DistillBet**, use the following command:
``` 
$bash scripts/trainBertALE.sh
```
To run the code with the demo data on **TextCNN** or **TextRNN**, use:
```
$bash scripts/trainALE.sh
``` 
You can specify the backbone model by modifying the `--model` parameter in the script. The output will be save under the `output` directory, at the path specified by the `--output_dir` argument.

## 🔍 Project Structure
```
CRAB/
├── conf/             # Configuration files (editable)
├── dataset/          # Dataset-related code for TextCNN and TextRNN
├── demoData/         # Demo data files (editable)
├── evaluate/         # Evaluation-related code
├── output/           # Output directory for results and checkpoints
├── scripts/          # Shell scripts to run experiments
├── config.py         # Configuration file handler
├── correlation.py    # Code for label-wise correlation calculation
├── environment.yml   # Conda environment specification
├── labelAnalysis.py  # Code for label-wise correlation analysis
├── Qdatasets.py      # Dataset-related code for BERT-based models
├── qureyStrategy.py  # Code for active learning query strategy
├── README.md         # Project documentation
├── trainALE.py       # Main code for MLAL with TextCNN/TextRNN backbones
├── trainBertALE.py   # Main code for MLAL with DistilBERT backbone
└── util.py           # Logging and utility functions
```


## 🔗 Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{qi2025multi,
  title={Multi-Label Bayesian Active Learning with Inter-Label Relationships},
  author={Qi, Yuanyuan and Lu, Jueqing and Yang, Xiaohao and Enticott, Joanne and Du, Lan},
  booktitle={41th Conference on Uncertainty in Artificial Intelligence (UAI)},
  year={2025}
}
```
