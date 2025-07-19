## Introduction

AutoSurvey is a speedy and well-organized framework for automating the creation of comprehensive literature surveys. However, the original system relied on a static database of references, which limited its ability to meet the demand for up-to-date research in survey writing. To address this, I have enhanced the framework by integrating a function to automatically fetch the latest articles from arXiv, thus satisfying the need for up-to-date, current research. Our new framework is called AutomaticSurvey.

## Requirements

- Python 3.10.x
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/GAry45v/AutomaticSurvey.git
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Download the database: 
   https://1drv.ms/u/c/8761b6d10f143944/EaqWZ4_YMLJIjGsEB_qtoHsBoExJ8bdppyBc1uxgijfZBw?e=2EIzti
   ```sh
   unzip database.zip -d ./database/
   ```

## Usage

### Generation
Here is an example command to generate survey on the topic "Image Fusion":

```sh
python main.py --topic "Image Fusion" 
               --saving_path ./output/
               --requirement "包含五年内的文献"
               --model o3-mini
               --section_num 8
               --subsection_len 700
               --rag_num 60
               --outline_reference_num 1200
               --db_path ./database
               --embedding_model nomic-ai/nomic-embed-text-v1
               --api_url https://api.openai.com/v1/chat/completions
               --api_key sk-xxxxxx 
```

The generated content will be saved in the `./output/` directory.

- `--requirement`: Specific requirement for generation.
- `--saving_path`: Directory of the generated survey.
- `--model`: Model to use. (recommend: o3-mini)
- `--topic`: Topic to generate content for.
- `--section_num`: Number of sections in the outline.
- `--subsection_len`: Length of each subsection.
- `--rag_num`: Number of references to use for RAG.
- `--outline_reference_num`: Number of references for outline generation.
- `--db_path`: Directory of the database.
- `--embedding_model`: Embedding model for retrieval.
- `--api_key`: API key for the model.
- `--api_url`: url for API request.

### Evaluation

Here is an example command to evaluate the generated survey on the topic "LLMs for education":

```sh
python evaluation.py --topic "LLMs for education" 
               --saving_path ./output/
               --model o3-mini
               --db_path ./database
               --embedding_model nomic-ai/nomic-embed-text-v1
               --api_url https://api.openai.com/v1/chat/completions
               --api_key sk-xxxxxx 
```

Make sure the generated survey is in the `./output/` directory

The evaluation result will be saved in the `./output/` directory.

- `--saving_path`: Directory to save the evaluation results (default: './output/').
- `--model`: Model for evaluation. (recommend use different models for evaluation)
- `--topic`: Topic of generated survey. (use the same topic as generation)
- `--db_path`: Directory of the database.
- `--embedding_model`: Embedding model for retrieval.
- `--api_key`: API key for the model.
- `--api_url`: url for API request.

## Acknowledgement
If you find our framework helpful, please give me a star!

And please cite the original project:

```
@inproceedings{
2024autosurvey,
title={AutoSurvey: Large Language Models Can Automatically Write Surveys},
author = {Wang, Yidong and Guo, Qi and Yao, Wenjin and Zhang, Hongbo and Zhang, Xin and Wu, Zhen and Zhang, Meishan and Dai, Xinyu and Zhang, Min and Wen, Qingsong and Ye, Wei and Zhang, Shikun and Zhang, Yue},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```
