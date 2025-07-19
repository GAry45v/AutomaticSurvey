import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import h5py
import json
from tqdm import tqdm
import faiss
from tinydb import TinyDB, Query
import arxiv

class database():

    def __init__(self, db_path, embedding_model) -> None:
        """
        初始化数据库。如果索引文件已存在，则加载它们；否则，提示用户先构建数据库。
        """
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.embedding_model.to(torch.device('cpu'))
        self.db_path = db_path
        self.raw_db_file = f'{self.db_path}/arxiv_paper_db.json'
        self.title_index_file = f'{self.db_path}/faiss_paper_title_embeddings.bin'
        self.abs_index_file = f'{self.db_path}/faiss_paper_abs_embeddings.bin'
        self.map_file = f'{self.db_path}/arxivid_to_index_abs.json'

        self.db_exists = all(os.path.exists(f) for f in [self.title_index_file, self.abs_index_file, self.map_file])

        if not self.db_exists:
            print("数据库索引文件不完整或不存在。请先调用 'build_and_save_database' 或 'update_and_rebuild_database' 来构建数据库。")
            self.table = None
            self.title_loaded_index = None
            self.abs_loaded_index = None
            self.id_to_index = None
            self.index_to_id = None
            return

        self.db = TinyDB(self.raw_db_file)
        self.table = self.db.table('cs_paper_info')
        self.User = Query()
        self.title_loaded_index = faiss.read_index(self.title_index_file)
        self.abs_loaded_index = faiss.read_index(self.abs_index_file)
        self.id_to_index, self.index_to_id = self.load_index_arxivid(self.db_path)

    def update_with_new_papers(self, topic: str, max_results: int = 20) -> list:
        """
        根据主题从ArXiv搜索新文章，并将其合并到现有的JSON数据库文件中。
        这个方法会更新JSON文件，并返回新增文章的列表。
        """
        def _format_paper_entry(result: arxiv.Result):
            return {"id": result.get_short_id(), "title": result.title, "url": result.pdf_url, "date": result.published.strftime('%Y-%m-%d'),"abs": result.summary.replace('\n', ' '),"cat": result.primary_category, "authors": [author.name for author in result.authors]}

        print(f"开始从 ArXiv 搜索关于 '{topic}' 的最新文章...")
        client = arxiv.Client()
        search = arxiv.Search(query=f'ti:"{topic}" OR abs:"{topic}"', max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        new_papers_results = list(tqdm(client.results(search), desc="下载新文章元数据"))

        if not new_papers_results:
            print("没有找到新的文章。")
            return []

        print(f"成功获取到 {len(new_papers_results)} 篇新文章。")

        try:
            with open(self.raw_db_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            print(f"警告：找不到原始数据库文件 {self.raw_db_file}。将创建一个新的数据库。")
            existing_data = {"cs_paper_info": {}}

        paper_table = existing_data.get("cs_paper_info", {})
        start_index = max([int(k) for k in paper_table.keys()]) + 1 if paper_table else 1
        
        newly_added_papers = []
        current_index = start_index
        for paper_result in new_papers_results:
            if any(p['id'].split('v')[0] == paper_result.get_short_id().split('v')[0] for p in paper_table.values()):
                continue
            
            formatted_paper = _format_paper_entry(paper_result)
            paper_table[str(current_index)] = formatted_paper
            newly_added_papers.append(formatted_paper)
            current_index += 1
        
        if not newly_added_papers:
            print("没有新增文章（可能所有获取的文章都已存在）。")
            return []
            
        print(f"新增了 {len(newly_added_papers)} 篇文章。当前总文章数: {len(paper_table)}")
        print(f"正在将更新后的数据保存回: {self.raw_db_file}")
        with open(self.raw_db_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
            
        return newly_added_papers

    def incrementally_update_index(self, newly_added_papers: list, batch_size: int = 32):
        """
        [高效] 仅为新增文章生成向量，并将其添加到现有Faiss索引中。
        """
        if not self.db_exists:
            print("错误：找不到基础索引文件。请先运行一次完整的 build_and_save_database。")
            return
            
        if not newly_added_papers:
            print("没有需要添加到索引的新文章。")
            return

        print("开始增量更新Faiss索引...")
        existing_vector_count = self.abs_loaded_index.ntotal
        print(f"现有索引中包含 {existing_vector_count} 个向量。")

        # 1. 只为新文章提取标题和摘要
        title_l = [p['title'] for p in newly_added_papers]
        abs_l = [p['abs'] for p in newly_added_papers]

        # 2. 只为新文章生成向量
        def get_embeddings(text_l):
            res = []
            for i in tqdm(range(0, len(text_l), batch_size), desc="为新文章生成向量"):
                batch_text = ['search_document: ' + _ for _ in text_l[i:i+batch_size]]
                res.append(self.embedding_model.encode(batch_text, convert_to_numpy=True))
            return np.concatenate(res, axis=0)

        new_title_embeddings = get_embeddings(title_l)
        new_abs_embeddings = get_embeddings(abs_l)

        # 3. 将新向量添加到已加载的索引中
        print("正在将新向量添加到现有索引...")
        self.title_loaded_index.add(new_title_embeddings.astype('float32'))
        self.abs_loaded_index.add(new_abs_embeddings.astype('float32'))

        # 4. 更新ID映射
        print("正在更新ID映射...")
        for i, paper in enumerate(newly_added_papers):
            new_faiss_index = existing_vector_count + i
            arxiv_id = paper['id']
            # 注意：这里的 self.id_to_index 和 self.index_to_id 是已经从文件加载到内存中的
            self.id_to_index[arxiv_id] = new_faiss_index
            self.index_to_id[new_faiss_index] = arxiv_id
            
        # 5. 将更新后的索引和映射写回文件
        print("正在保存更新后的索引和映射文件...")
        faiss.write_index(self.title_loaded_index, self.title_index_file)
        faiss.write_index(self.abs_loaded_index, self.abs_index_file)
        with open(self.map_file, 'w') as f:
            # 注意：这里需要将整个更新后的映射写回
            json.dump(self.id_to_index, f, indent=4)
            
        print(f"增量更新完成！索引中现在总共有 {self.abs_loaded_index.ntotal} 个向量。")
    
    def update_and_rebuild_database(self, topic: str, max_results: int = 20):
        """
        一键式调度函数：自动从ArXiv获取新文章，然后智能地更新或重建数据库。
        - 如果数据库已存在，执行高效的“增量更新”。
        - 如果数据库不存在，执行“完全重建”。
        """
        # --- 步骤 1: 更新源JSON文件，并获取新增条目 ---
        newly_added = self.update_with_new_papers(topic=topic, max_results=max_results)
        
        # --- 步骤 2: 根据情况选择重建或增量更新 ---
        if not self.db_exists:
            # 如果是第一次，必须完全重建
            print("\n检测到是首次构建，将执行完整的数据库重建流程...")
            self.build_and_save_database(raw_json_path=self.raw_db_file)
        elif newly_added:
            # 如果有新增条目，并且数据库已存在，则执行增量更新
            print("\n检测到有新增文章，将执行高效的增量索引更新...")
            self.incrementally_update_index(newly_added_papers=newly_added)
        else:
            print("\n没有新增文章可用于更新索引。")

        print("\n--- 流程结束 ---")
        print("重要提示：请重新初始化您的database对象以确保加载了最新的索引。")

    def build_and_save_database(self, raw_json_path, batch_size=32):
        print("开始构建数据库...")
        with open(raw_json_path, 'r') as f:
            papers = json.load(f.read())
        papers_l = list(papers.get('cs_paper_info', papers.get('_default', {})).items())
        if not papers_l:
            raise ValueError("在JSON文件中找不到论文数据。")
        title_l = [paper[1]['title'] for paper in papers_l]
        abs_l = [paper[1]['abs'] for paper in papers_l]
        def get_embeddings(text_l):
            res = []
            for i in tqdm(range(0, len(text_l), batch_size), desc="生成向量中"):
                batch_text = ['search_document: ' + _ for _ in text_l[i:i+batch_size]]
                res.append(self.embedding_model.encode(batch_text, convert_to_numpy=True))
            return np.concatenate(res, axis=0)
        title_embeddings = get_embeddings(title_l)
        abs_embeddings = get_embeddings(abs_l)
        title_index = faiss.IndexFlatL2(title_embeddings.shape[1])
        title_index.add(title_embeddings.astype('float32'))
        abs_index = faiss.IndexFlatL2(abs_embeddings.shape[1])
        abs_index.add(abs_embeddings.astype('float32'))
        faiss.write_index(title_index, self.title_index_file)
        faiss.write_index(abs_index, self.abs_index_file)
        paperid_2_index = {paper[1]['id']: int(paper[0]) for paper in papers_l}
        with open(self.map_file, 'w') as f:
            json.dump(paperid_2_index, f, indent=4)
        print("数据库构建完成！现在可以重新初始化 database 类来加载新构建的索引。")

    def load_index_arxivid(self, db_path):
        with open(f'{db_path}/arxivid_to_index_abs.json','r') as f:
            id_to_index = json.loads(f.read())
        id_to_index = {str(id): int(index) for id, index in id_to_index.items()}
        index_to_id = {int(index): str(id) for id, index in id_to_index.items()}
        return id_to_index, index_to_id
        
    def get_embeddings(self, batch_text):
        batch_text = ['search_query: ' + _ for _ in batch_text]
        embeddings = self.embedding_model.encode(batch_text)
        return embeddings

    def get_embeddings_documents(self, batch_text):
        batch_text = ['search_document: ' + _ for _ in batch_text]
        embeddings = self.embedding_model.encode(batch_text)
        return embeddings

    def batch_search(self, query_vectors, top_k=1, title=False):
        if not self.abs_loaded_index:
            raise RuntimeError("数据库未正确加载，请先构建或检查文件路径。")
        query_vectors = np.array(query_vectors).astype('float32')
        if title:
            distances, indices = self.title_loaded_index.search(query_vectors, top_k)
        else:
            distances, indices = self.abs_loaded_index.search(query_vectors, top_k)
        results = []
        for i, query in enumerate(query_vectors):
            result = [(self.index_to_id[idx], distances[i][j]) for j, idx in enumerate(indices[i]) if idx != -1 and idx in self.index_to_id]
            results.append([_[0] for _ in result])
        return results

    def search(self, query_vector, top_k=1, title=False):
        if not self.abs_loaded_index:
            raise RuntimeError("数据库未正确加载，请先构建或检查文件路径。")
        query_vector = np.array([query_vector]).astype('float32')
        if title:
            distances, indices = self.title_loaded_index.search(query_vector, top_k)
        else:
            distances, indices = self.abs_loaded_index.search(query_vector, top_k)
        results = [(self.index_to_id[idx], distances[0][i]) for i, idx in enumerate(indices[0]) if idx != -1 and idx in self.index_to_id]
        return [_[0] for _ in results]

    def get_ids_from_query(self, query, num,  shuffle = False):
        q = self.get_embeddings([query])[0]
        return self.search(q, top_k=num)

    def get_titles_from_citations(self, citations):
        q = self.get_embeddings_documents(citations)
        ids = self.batch_search(q,1, True)
        return [_[0] for _ in ids]

    def get_ids_from_queries(self, queries, num,  shuffle = False):
        q = self.get_embeddings(queries)
        ids = self.batch_search(q,num)
        return ids

    def get_date_from_ids(self, ids):
        if not self.table:
            raise RuntimeError("数据库未正确加载。")
        result = self.table.search(self.User.id.one_of(ids))
        dates = [r['date'] for r in result]
        return dates

    def get_title_from_ids(self, ids):
        if not self.table:
            raise RuntimeError("数据库未正确加载。")
        result = self.table.search(self.User.id.one_of(ids))
        titles = [r['title'] for r in result]
        return titles

    def get_abs_from_ids(self, ids):
        if not self.table:
            raise RuntimeError("数据库未正确加载。")
        result = self.table.search(self.User.id.one_of(ids))
        abs_l = [r['abs'] for r in result]
        return abs_l

    def get_paper_info_from_ids(self, ids):
        if not self.table:
            raise RuntimeError("数据库未正确加载。")
        result = self.table.search(self.User.id.one_of(ids))
        return result

    def get_paper_from_ids(self, ids, max_len = 1500):
        if not self.table:
            raise RuntimeError("数据库未正确加载。")
        loaded_data = {}
        # 注意：这里假设 h5 文件在当前工作目录，如果不是，需要修改路径
        with h5py.File('./paper_content.h5', 'r') as f:
            for key in f.keys():
                if key in ids:
                    loaded_data[key] = str(f[key][()])
                if len(ids) == len(loaded_data):
                    break
        if not loaded_data:
            return []
        print(loaded_data[list(loaded_data.keys())[0]])
        # return [self.token_counter.text_truncation(loaded_data[_], max_len) for _ in ids] # 假设 token_counter 存在
        return loaded_data # 简单返回字典