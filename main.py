import os
import json
import argparse
import hashlib
from src.agents.outline_writer import outlineWriter
from src.agents.writer import subsectionWriter
from src.agents.judge import Judge
from src.database import database
from tqdm import tqdm
import time
import re

def remove_descriptions(text):
    lines = text.split('\n')
    
    filtered_lines = [line for line in lines if not line.strip().startswith("Description")]
    
    result = '\n'.join(filtered_lines)
    
    return result

def write(topic, model, section_num, subsection_len, rag_num, refinement):
    outline, outline_wo_description = write_outline(topic, model, section_num)

    if refinement:
        raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = write_subsection(topic, model, outline, subsection_len = subsection_len, rag_num = rag_num, refinement = True)
        return refined_survey_with_references
    else:
        raw_survey, raw_survey_with_references, raw_references = write_subsection(topic, model, outline, subsection_len = subsection_len, rag_num = rag_num, refinement = False)
        return raw_survey_with_references

def write_outline(topic, requirement, model, section_num, outline_reference_num, db, api_key, api_url):
    outline_writer = outlineWriter(model=model, api_key=api_key, api_url = api_url, database=db)
    print(outline_writer.api_model.chat('hello'))
    outline = outline_writer.draft_outline(topic, requirement, outline_reference_num, 30000, section_num)
    return outline, remove_descriptions(outline)

def write_subsection(topic, requirement, model, outline, subsection_len, rag_num, db, api_key, api_url, refinement = True):

    subsection_writer = subsectionWriter(model=model, api_key=api_key, api_url = api_url, database=db)
    if refinement:
        raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = subsection_writer.write(topic, requirement, outline, subsection_len = subsection_len, rag_num = rag_num, refining = True)
        return raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references
    else:
        raw_survey, raw_survey_with_references, raw_references = subsection_writer.write(topic, requirement, outline, subsection_len = subsection_len, rag_num = rag_num, refining = False)
        return raw_survey, raw_survey_with_references, raw_references

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--saving_path',default='./output/', type=str, help='Directory to save the output survey')
    parser.add_argument('--model',default='o3-mini', type=str, help='Model to use')
    parser.add_argument('--topic',default='Graph Anomaly Detection', type=str, help='Topic to generate survey for')
    parser.add_argument('--requirement',default='一、项目背景:图异常检测（Graph Anomaly Detection）是当前机器学习与数据挖掘中的活跃研究方向，在金融风控、网络安全、工业故障预警等领域具有重要应用价值。为了帮助研究生快速了解前沿技术、锻炼科研与工程实践能力，本次夏令营设置以下两条并行研究方向，学生可根据自身兴趣与专长任选其一。二、项目目标:文献调研与深度总结。1.系统梳理 2020–2025 年图异常检测最新研究成果（不少于 12 篇高水平论文）。2.从问题定义、理论创新、模型设计、实验设置、应用场景等维度进行横向与纵向比较，总结研究热点、技术趋势与未来挑战。3.输出高质量综述报告，培养学生的批判性思维与学术写作能力。三、预期成果方向一：≥ 7 页中文或英文综述报告（含图表、引用不少于 13 篇）。', type=str, help='Specific requirements')
    parser.add_argument('--search_num',default=200, type=int, help='Number of the latest papers to search for')
    parser.add_argument('--section_num',default=8, type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len',default=700, type=int, help='Length of each subsection')
    parser.add_argument('--outline_reference_num',default=1200, type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num',default=60, type=int, help='Number of references to use for RAG')
    parser.add_argument('--api_url',default='https://jeniya.cn/v1/chat/completions', type=str, help='url for API request')
    parser.add_argument('--api_key',default='sk-mxUE8rrgcb4YfySnuwLQkwBvcKr41iAkGHrigZV0VMZp4oLR', type=str, help='API key for the model')
    parser.add_argument('--db_path',default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--embedding_model',default='nomic-ai/nomic-embed-text-v1', type=str, help='Embedding model for retrieval.')
    args = parser.parse_args()
    return args

def main(args):

    db = database(db_path = args.db_path, embedding_model = args.embedding_model)
    db.update_and_rebuild_database(topic=args.topic, max_results=200)
    print("-" * 30)
    print("\n--- 操作完成，重新加载数据库以供使用 ---")
    db = database(db_path='./database', embedding_model = args.embedding_model)
    #如果要更新数据库
    api_key = args.api_key

    if not os.path.exists(args.saving_path):
        os.mkdir(args.saving_path)
    
    outline_with_description, outline_wo_description = write_outline(args.topic, args.requirement, args.model, args.section_num, args.outline_reference_num, db, args.api_key, args.api_url)

    raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = write_subsection(args.topic, args.requirement, args.model, outline_with_description, args.subsection_len, args.rag_num, db, args.api_key, args.api_url)

    # Generate a short, unique filename from the topic
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 2. (可选但推荐) 清理topic字符串，移除不适合做文件名的特殊字符
    sanitized_topic = re.sub(r'[^\w\-_]', '_', args.topic) # 将非字母、数字、下划线、连字符的都替换成下划线
    
    # 3. 组合成新的、更具描述性的文件名
    filename_base = f"{timestamp}_{sanitized_topic}"
    # ======================================================

    # 使用新的文件名基础来保存文件
    with open(f'{args.saving_path}/{filename_base}.md', 'w', encoding='utf-8') as f: # 使用 'w' 模式来创建新文件
        f.write(refined_survey_with_references)
        
    with open(f'{args.saving_path}/{filename_base}.json', 'w', encoding='utf-8') as f: # 使用 'w' 模式
        save_dic = {}
        save_dic['survey'] = refined_survey_with_references
        save_dic['reference'] = refined_references
        f.write(json.dumps(save_dic, indent=4))
        
    print(f"\nSurvey has been successfully saved to '{args.saving_path}' with base name '{filename_base}'")

if __name__ == '__main__':

    args = paras_args()

    main(args)