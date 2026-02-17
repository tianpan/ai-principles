"""
表格模板存储模块

用于存储和加载成功识别的Excel表格结构模板，优化后续相似表格的识别效率。
"""
import os
import json
import hashlib
import pandas as pd
from datetime import datetime

class TemplateStore:
    """表格模板存储类，用于管理Excel表格的结构模板"""
    
    def __init__(self, template_dir=None):
        """
        初始化模板存储
        
        Args:
            template_dir: 模板存储目录，默认为当前目录下的templates文件夹
        """
        if template_dir is None:
            self.template_dir = os.path.join(os.getcwd(), "templates")
        else:
            self.template_dir = template_dir
            
        # 确保模板目录存在
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir)
            
        # 资产负债表和损益表的模板存储路径
        self.balance_sheet_templates_path = os.path.join(self.template_dir, "balance_sheet_templates.json")
        self.income_statement_templates_path = os.path.join(self.template_dir, "income_statement_templates.json")
        
        # 初始化模板库
        self.balance_sheet_templates = self._load_templates(self.balance_sheet_templates_path)
        self.income_statement_templates = self._load_templates(self.income_statement_templates_path)
        
    def _load_templates(self, template_path):
        """
        加载模板数据
        
        Args:
            template_path: 模板文件路径
            
        Returns:
            dict: 加载的模板数据
        """
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载模板文件出错: {e}")
                return {}
        else:
            return {}
            
    def _save_templates(self, templates, template_path):
        """
        保存模板数据
        
        Args:
            templates: 模板数据
            template_path: 保存路径
        """
        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(templates, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存模板文件出错: {e}")
            
    def _generate_template_hash(self, df):
        """
        为DataFrame生成哈希值作为模板标识
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            str: 哈希值
        """
        # 提取列名和前5行数据的字符串表示
        header_str = str(df.columns.tolist())
        data_sample = str(df.head().values.flatten().tolist())
        
        # 计算哈希值
        hash_obj = hashlib.md5((header_str + data_sample).encode())
        return hash_obj.hexdigest()
        
    def _extract_template_features(self, df, sheet_type, successful_patterns):
        """
        从成功识别的DataFrame中提取模板特征
        
        Args:
            df: 成功处理的DataFrame
            sheet_type: 表格类型 ('balance_sheet' 或 'income_statement')
            successful_patterns: 成功的匹配模式
            
        Returns:
            dict: 模板特征
        """
        # 提取列名
        columns = df.columns.tolist()
        
        # 提取关键行索引
        key_rows = []
        for idx, row in df.iterrows():
            row_str = ' '.join([str(x) for x in row if pd.notna(x)])
            if any(keyword in row_str for keyword in successful_patterns):
                key_rows.append((idx, row_str[:100]))  # 只保存前100个字符作为标识
                
        # 提取表格结构特征
        template = {
            "sheet_type": sheet_type,
            "columns": columns,
            "key_rows": key_rows,
            "column_count": len(columns),
            "successful_patterns": successful_patterns,
            "created_at": datetime.now().isoformat(),
            "usage_count": 1
        }
        
        return template
        
    def save_balance_sheet_template(self, df, successful_patterns):
        """
        保存成功处理的资产负债表模板
        
        Args:
            df: 成功处理的DataFrame
            successful_patterns: 成功的匹配模式
            
        Returns:
            str: 模板ID
        """
        template_id = self._generate_template_hash(df)
        template = self._extract_template_features(df, 'balance_sheet', successful_patterns)
        
        # 更新或添加模板
        if template_id in self.balance_sheet_templates:
            self.balance_sheet_templates[template_id]["usage_count"] += 1
            self.balance_sheet_templates[template_id]["last_used"] = datetime.now().isoformat()
        else:
            self.balance_sheet_templates[template_id] = template
            
        # 保存模板库
        self._save_templates(self.balance_sheet_templates, self.balance_sheet_templates_path)
        
        return template_id
        
    def save_income_statement_template(self, df, successful_patterns):
        """
        保存成功处理的损益表模板
        
        Args:
            df: 成功处理的DataFrame
            successful_patterns: 成功的匹配模式
            
        Returns:
            str: 模板ID
        """
        template_id = self._generate_template_hash(df)
        template = self._extract_template_features(df, 'income_statement', successful_patterns)
        
        # 更新或添加模板
        if template_id in self.income_statement_templates:
            self.income_statement_templates[template_id]["usage_count"] += 1
            self.income_statement_templates[template_id]["last_used"] = datetime.now().isoformat()
        else:
            self.income_statement_templates[template_id] = template
            
        # 保存模板库
        self._save_templates(self.income_statement_templates, self.income_statement_templates_path)
        
        return template_id
        
    def find_best_balance_sheet_template(self, df):
        """
        查找与输入DataFrame最匹配的资产负债表模板
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            tuple: (模板ID, 模板数据, 相似度得分) 如果没有匹配则为 (None, None, 0)
        """
        return self._find_best_template(df, self.balance_sheet_templates)
        
    def find_best_income_statement_template(self, df):
        """
        查找与输入DataFrame最匹配的损益表模板
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            tuple: (模板ID, 模板数据, 相似度得分) 如果没有匹配则为 (None, None, 0)
        """
        return self._find_best_template(df, self.income_statement_templates)
        
    def _find_best_template(self, df, templates):
        """
        在模板库中查找最匹配的模板
        
        Args:
            df: 输入的DataFrame
            templates: 模板库
            
        Returns:
            tuple: (模板ID, 模板数据, 相似度得分) 如果没有匹配则为 (None, None, 0)
        """
        if not templates:
            return None, None, 0
            
        best_match = None
        best_score = 0
        best_id = None
        
        # 提取当前DataFrame的特征
        columns = df.columns.tolist()
        column_count = len(columns)
        
        for template_id, template in templates.items():
            # 列数匹配度
            col_count_similarity = 1 - abs(column_count - template["column_count"]) / max(column_count, template["column_count"])
            
            # 关键行匹配
            key_row_matches = 0
            for idx, row in df.iterrows():
                row_str = ' '.join([str(x) for x in row if pd.notna(x)])
                for pattern in template["successful_patterns"]:
                    if pattern in row_str:
                        key_row_matches += 1
                        break
            
            key_row_similarity = key_row_matches / max(1, len(template["successful_patterns"]))
            
            # 计算总得分 (加权平均)
            score = (col_count_similarity * 0.3) + (key_row_similarity * 0.7)
            
            if score > best_score:
                best_score = score
                best_match = template
                best_id = template_id
                
        if best_score >= 0.6:  # 只有当得分足够高时才返回匹配的模板
            return best_id, best_match, best_score
        else:
            return None, None, 0

    def get_template_suggestions(self, df, sheet_type):
        """
        获取针对特定DataFrame的模板处理建议
        
        Args:
            df: 输入的DataFrame
            sheet_type: 表格类型 ('balance_sheet' 或 'income_statement')
            
        Returns:
            dict: 处理建议
        """
        if sheet_type == 'balance_sheet':
            template_id, template, score = self.find_best_balance_sheet_template(df)
        else:
            template_id, template, score = self.find_best_income_statement_template(df)
            
        if template is None:
            return {
                "matched": False,
                "message": "未找到匹配的模板，将尝试通用识别方法"
            }
            
        suggestions = {
            "matched": True,
            "template_id": template_id,
            "match_score": score,
            "patterns": template["successful_patterns"],
            "message": f"找到匹配模板，匹配度: {score:.2f}"
        }
        
        return suggestions 