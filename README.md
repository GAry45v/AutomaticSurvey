## Introduction

AutoSurvey is a speedy and well-organized framework for automating the creation of comprehensive literature surveys. However, the original system relied on a static database of references, which limited its ability to meet the demand for up-to-date research in survey writing. To address this, I have enhanced the framework by integrating a function to automatically fetch the latest articles from arXiv, thus satisfying the need for up-to-date, current research.

## Requirements

- Python 3.10.x
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Download the database: (Here we provide a database containing 530,000 arXiv paper abstracts and all papers are under the CS category. You can contact [us](mailto:qguo@smail.nju.edu.cn) to obtain the database containing the full content of the papers. )
   https://1drv.ms/u/c/8761b6d10f143944/EaqWZ4_YMLJIjGsEB_qtoHsBoExJ8bdppyBc1uxgijfZBw?e=2EIzti
   ```sh
   unzip database.zip -d ./database/
   ```

## Usage

### Generation
Here is an example command to generate survey on the topic "LLMs for education":

```sh
python main.py --topic "LLMs for education" 
               --gpu 0
               --saving_path ./output/
               --model gpt-4o-2024-05-13
               --section_num 7
               --subsection_len 700
               --rag_num 60
               --outline_reference_num 1500
               --db_path ./database
               --embedding_model nomic-ai/nomic-embed-text-v1
               --api_url https://api.openai.com/v1/chat/completions
               --api_key sk-xxxxxx 
```

The generated content will be saved in the `./output/` directory.

- `--gpu`: Specify the GPU to use.
- `--saving_path`: Directory to save the output survey.
- `--model`: Model to use.
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
               --gpu 0
               --saving_path ./output/
               --model gpt-4o-2024-05-13
               --db_path ./database
               --embedding_model nomic-ai/nomic-embed-text-v1
               --api_url https://api.openai.com/v1/chat/completions
               --api_key sk-xxxxxx 
```

Make sure the generated survey is in the `./output/` directory

The evaluation result will be saved in the `./output/` directory.

- `--gpu`: Specify the GPU to use (default: '0').
- `--saving_path`: Directory to save the evaluation results (default: './output/').
- `--model`: Model for evaluation.
- `--topic`: Topic of generated survey.
- `--db_path`: Directory of the database.
- `--embedding_model`: Embedding model for retrieval.
- `--api_key`: API key for the model.
- `--api_url`: url for API request.

## Citing Autosurvey

Please cite us if you find this project helpful for your project/paper:

```
@article{wang2024autosurvey,
  title={AutoSurvey: Large Language Models Can Automatically Write Surveys},
  author={Wang, Yidong and Guo, Qi and Yao, Wenjin and Zhang, Hongbo and Zhang, Xin and Wu, Zhen and Zhang, Meishan and Dai, Xinyu and Zhang, Min and Wen, Qingsong and others},
  journal={arXiv preprint arXiv:2406.10252},
  year={2024}
}

@inproceedings{
2024autosurvey,
title={AutoSurvey: Large Language Models Can Automatically Write Surveys},
author = {Wang, Yidong and Guo, Qi and Yao, Wenjin and Zhang, Hongbo and Zhang, Xin and Wu, Zhen and Zhang, Meishan and Dai, Xinyu and Zhang, Min and Wen, Qingsong and Ye, Wei and Zhang, Shikun and Zhang, Yue},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```

## Contributing

Contributions are welcome! Please open an issue to discuss what you would like to change.

## License

This project is licensed under the MIT License.
