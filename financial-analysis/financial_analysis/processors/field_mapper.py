"""
字段映射模块

用于定义和处理财务报表中字段的映射关系，支持精确匹配、模糊匹配和上下文推断
"""
import re
import difflib
from financial_analysis.logger import log_info, log_debug, log_warning, log_error

class FieldMapper:
    """财务字段映射器，用于识别和映射不同表述的同一财务指标"""
    
    def __init__(self):
        """初始化字段映射器"""
        # 资产负债表字段映射
        self.balance_sheet_mappings = {
            # 资产部分
            "流动资产": [
                "流动资产", "流动资产合计", "流动资产总计", "流动资产小计", 
                "current assets", "total current assets"
            ],
            "货币资金": [
                "货币资金", "现金", "现金及现金等价物", "库存现金", "银行存款",
                "cash", "cash and cash equivalents", "monetary fund"
            ],
            "应收账款": [
                "应收账款", "应收账款净额", "应收账款净值", "应收款项", 
                "accounts receivable", "account receivable", "ar"
            ],
            "其他应收款": [
                "其他应收款", "其他应收账款", "其他应收项目", 
                "other receivables", "other accounts receivable"
            ],
            "存货": [
                "存货", "库存", "库存商品", "原材料", "在产品", 
                "inventory", "inventories", "stock"
            ],
            "预付款项": [
                "预付款项", "预付账款", "预付费用", "预付款", 
                "prepayments", "advances to suppliers", "prepaid expenses"
            ],
            "交易性金融资产": [
                "交易性金融资产", "以公允价值计量且其变动计入当期损益的金融资产", 
                "trading financial assets", "financial assets at fair value through profit or loss"
            ],
            "一年内到期的非流动资产": [
                "一年内到期的非流动资产", "一年内到期非流动资产", 
                "current portion of non-current assets", "non-current assets due within one year"
            ],
            
            # 非流动资产
            "非流动资产": [
                "非流动资产", "非流动资产合计", "非流动资产总计", "非流动资产小计", 
                "non-current assets", "total non-current assets"
            ],
            "固定资产": [
                "固定资产", "固定资产净值", "固定资产净额", "固定资产账面价值", 
                "fixed assets", "property, plant and equipment", "ppe"
            ],
            "在建工程": [
                "在建工程", "在建项目", "在建资产", 
                "construction in progress", "projects under construction"
            ],
            "无形资产": [
                "无形资产", "无形资产净值", "无形资产账面价值", 
                "intangible assets", "intangibles"
            ],
            "商誉": [
                "商誉", "商誉净值", "商誉账面价值", 
                "goodwill"
            ],
            "长期股权投资": [
                "长期股权投资", "长期投资", "股权投资", 
                "long-term equity investments", "equity investments"
            ],
            "长期应收款": [
                "长期应收款", "长期应收项目", 
                "long-term receivables"
            ],
            "递延所得税资产": [
                "递延所得税资产", "递延税款", "递延税资产", 
                "deferred tax assets", "deferred income tax assets"
            ],
            
            # 资产总计
            "资产总计": [
                "资产总计", "资产合计", "总资产", "资产总额", 
                "total assets", "assets total"
            ],
            
            # 负债部分
            "流动负债": [
                "流动负债", "流动负债合计", "流动负债总计", "流动负债小计", 
                "current liabilities", "total current liabilities"
            ],
            "短期借款": [
                "短期借款", "短期贷款", "短期债务", 
                "short-term borrowings", "short-term loans"
            ],
            "应付账款": [
                "应付账款", "应付款项", "应付票据及应付账款", 
                "accounts payable", "account payable", "ap"
            ],
            "应付职工薪酬": [
                "应付职工薪酬", "应付工资", "应付薪金", "工资和福利费用", 
                "employee benefits payable", "salaries and wages payable"
            ],
            "应交税费": [
                "应交税费", "应交税款", "应缴税金", "应付税款", 
                "taxes payable", "tax payable"
            ],
            "其他应付款": [
                "其他应付款", "其他应付项目", 
                "other payables", "other accounts payable"
            ],
            "一年内到期的非流动负债": [
                "一年内到期的非流动负债", "一年内到期非流动负债", 
                "current portion of non-current liabilities", "non-current liabilities due within one year"
            ],
            
            # 非流动负债
            "非流动负债": [
                "非流动负债", "非流动负债合计", "非流动负债总计", "非流动负债小计", 
                "non-current liabilities", "total non-current liabilities", "long-term liabilities"
            ],
            "长期借款": [
                "长期借款", "长期贷款", "长期债务", 
                "long-term borrowings", "long-term loans"
            ],
            "应付债券": [
                "应付债券", "债券", "公司债", 
                "bonds payable", "debentures"
            ],
            "长期应付款": [
                "长期应付款", "长期应付项目", 
                "long-term payables"
            ],
            "递延所得税负债": [
                "递延所得税负债", "递延税负债", 
                "deferred tax liabilities", "deferred income tax liabilities"
            ],
            
            # 负债总计
            "负债合计": [
                "负债合计", "负债总计", "总负债", "负债总额", 
                "total liabilities", "liabilities total"
            ],
            
            # 所有者权益
            "所有者权益": [
                "所有者权益", "所有者权益合计", "所有者权益总计", "权益合计", 
                "股东权益", "股东权益合计", "股东权益总计", 
                "total equity", "equity", "shareholders' equity", "total shareholders' equity"
            ],
            "实收资本": [
                "实收资本", "股本", "注册资本", "资本金", 
                "paid-in capital", "share capital", "capital stock"
            ],
            "资本公积": [
                "资本公积", "资本公积金", "股本溢价", 
                "capital reserve", "capital surplus", "share premium"
            ],
            "盈余公积": [
                "盈余公积", "盈余公积金", "法定盈余公积", 
                "surplus reserve", "statutory reserve"
            ],
            "未分配利润": [
                "未分配利润", "未分配收益", "留存收益", "累积利润", 
                "retained earnings", "retained profits", "undistributed profits"
            ],
            "少数股东权益": [
                "少数股东权益", "少数股东", "非控股权益", 
                "minority interests", "non-controlling interests"
            ],
            
            # 负债和所有者权益总计
            "负债和所有者权益总计": [
                "负债和所有者权益总计", "负债及所有者权益总计", "负债和股东权益总计", 
                "负债及股东权益总计", "负债与所有者权益总计", "负债和权益总计", 
                "total liabilities and equity", "total liabilities and shareholders' equity"
            ]
        }
        
        # 损益表字段映射
        self.income_statement_mappings = {
            # 收入部分
            "营业收入": [
                "营业收入", "营业总收入", "主营业务收入", "收入总额", "收入", 
                "revenue", "operating revenue", "total revenue", "sales revenue"
            ],
            "其他业务收入": [
                "其他业务收入", "其他收入", "辅助业务收入", 
                "other revenue", "other operating revenue"
            ],
            
            # 成本部分
            "营业成本": [
                "营业成本", "营业总成本", "主营业务成本", "成本总额", "成本", 
                "cost of revenue", "operating cost", "total cost", "cost of sales"
            ],
            "其他业务成本": [
                "其他业务成本", "其他成本", "辅助业务成本", 
                "other costs", "other operating costs"
            ],
            "销售费用": [
                "销售费用", "营销费用", "销售及营销费用", 
                "selling expenses", "marketing expenses", "selling and marketing expenses"
            ],
            "管理费用": [
                "管理费用", "行政费用", "行政管理费用", 
                "administrative expenses", "general and administrative expenses", "g&a expenses"
            ],
            "研发费用": [
                "研发费用", "研究与开发费用", "研发支出", 
                "research and development expenses", "r&d expenses", "research expenses"
            ],
            "财务费用": [
                "财务费用", "利息费用", "融资费用", 
                "financial expenses", "interest expenses", "finance costs"
            ],
            
            # 利润部分
            "营业利润": [
                "营业利润", "经营利润", "营业盈利", 
                "operating profit", "operating income", "business profit"
            ],
            "利润总额": [
                "利润总额", "税前利润", "税前盈利", 
                "profit before tax", "total profit", "profit before income tax"
            ],
            "净利润": [
                "净利润", "税后利润", "净盈利", "净收益", 
                "net profit", "net income", "profit after tax", "net earnings"
            ],
            "毛利润": [
                "毛利润", "毛利", "毛利总额", 
                "gross profit", "gross margin"
            ],
            "归属于母公司股东的净利润": [
                "归属于母公司股东的净利润", "归属于母公司所有者的净利润", "归属于公司普通股股东的净利润", 
                "profit attributable to owners of the parent", "net profit attributable to shareholders"
            ],
            "少数股东损益": [
                "少数股东损益", "少数股东利润", "非控股权益损益", 
                "minority interests", "non-controlling interests"
            ],
            
            # 其他项目
            "营业外收入": [
                "营业外收入", "非经营性收入", "非经常性收入", 
                "non-operating income", "extraordinary income"
            ],
            "营业外支出": [
                "营业外支出", "非经营性支出", "非经常性支出", 
                "non-operating expenses", "extraordinary expenses"
            ],
            "资产减值损失": [
                "资产减值损失", "减值损失", "资产减值准备", 
                "asset impairment losses", "impairment losses"
            ],
            "所得税费用": [
                "所得税费用", "所得税", "所得税支出", 
                "income tax expense", "income tax", "tax expense"
            ],
            "基本每股收益": [
                "基本每股收益", "每股收益", "每股基本收益", 
                "basic earnings per share", "earnings per share", "basic eps"
            ],
            "稀释每股收益": [
                "稀释每股收益", "每股稀释收益", 
                "diluted earnings per share", "diluted eps"
            ]
        }
        
        # 新增映射的历史记录
        self.new_mappings = []
        
    def match_balance_sheet_field(self, field_name, threshold=0.75):
        """
        匹配资产负债表字段
        
        Args:
            field_name: 待匹配的字段名
            threshold: 模糊匹配的阈值
            
        Returns:
            tuple: (标准字段名, 匹配度)
        """
        return self._match_field(field_name, self.balance_sheet_mappings, threshold)
        
    def match_income_statement_field(self, field_name, threshold=0.75):
        """
        匹配损益表字段
        
        Args:
            field_name: 待匹配的字段名
            threshold: 模糊匹配的阈值
            
        Returns:
            tuple: (标准字段名, 匹配度)
        """
        return self._match_field(field_name, self.income_statement_mappings, threshold)
        
    def _match_field(self, field_name, mappings, threshold):
        """
        匹配字段的通用方法
        
        Args:
            field_name: 待匹配的字段名
            mappings: 映射字典
            threshold: 模糊匹配的阈值
            
        Returns:
            tuple: (标准字段名, 匹配度)
        """
        if not field_name or not isinstance(field_name, str):
            return None, 0
            
        # 清理字段名
        field_name = field_name.strip().lower()
        
        # 1. 精确匹配
        for standard_name, variants in mappings.items():
            if field_name == standard_name.lower():
                return standard_name, 1.0
                
            for variant in variants:
                if field_name == variant.lower():
                    return standard_name, 1.0
                    
        # 2. 包含匹配
        for standard_name, variants in mappings.items():
            if standard_name.lower() in field_name:
                return standard_name, 0.9
                
            for variant in variants:
                if variant.lower() in field_name:
                    return standard_name, 0.9
                    
        # 3. 模糊匹配
        best_match = None
        best_score = 0
        
        for standard_name, variants in mappings.items():
            # 与标准名称比较
            score = difflib.SequenceMatcher(None, field_name, standard_name.lower()).ratio()
            if score > best_score:
                best_match = standard_name
                best_score = score
                
            # 与变体比较
            for variant in variants:
                score = difflib.SequenceMatcher(None, field_name, variant.lower()).ratio()
                if score > best_score:
                    best_match = standard_name
                    best_score = score
                    
        if best_score >= threshold:
            return best_match, best_score
            
        # 4. 关键字匹配
        # 将字段名拆分为关键字
        keywords = re.findall(r'\w+', field_name)
        for standard_name, variants in mappings.items():
            # 计算与标准名称的关键字重叠度
            standard_keywords = set(re.findall(r'\w+', standard_name.lower()))
            overlap = len(set(keywords) & standard_keywords) / max(len(keywords), len(standard_keywords))
            
            if overlap > best_score:
                best_match = standard_name
                best_score = overlap
                
            # 计算与变体的关键字重叠度
            for variant in variants:
                variant_keywords = set(re.findall(r'\w+', variant.lower()))
                overlap = len(set(keywords) & variant_keywords) / max(len(keywords), len(variant_keywords))
                
                if overlap > best_score:
                    best_match = standard_name
                    best_score = overlap
                    
        if best_score >= threshold:
            return best_match, best_score
            
        return None, 0
        
    def learn_new_mapping(self, field_name, standard_name, sheet_type="balance_sheet"):
        """
        学习新的字段映射
        
        Args:
            field_name: 新的字段名
            standard_name: 标准字段名
            sheet_type: 表格类型 ('balance_sheet' 或 'income_statement')
            
        Returns:
            bool: 是否成功学习
        """
        if not field_name or not standard_name:
            return False
            
        # 清理字段名
        field_name = field_name.strip()
        
        # 根据表格类型选择映射字典
        if sheet_type == "balance_sheet":
            mappings = self.balance_sheet_mappings
        elif sheet_type == "income_statement":
            mappings = self.income_statement_mappings
        else:
            return False
            
        # 检查标准名称是否存在
        if standard_name not in mappings:
            return False
            
        # 检查是否已经存在该映射
        if field_name in mappings[standard_name]:
            return True
            
        # 添加新的映射
        mappings[standard_name].append(field_name)
        
        # 记录新增的映射
        self.new_mappings.append({
            "field_name": field_name,
            "standard_name": standard_name,
            "sheet_type": sheet_type
        })
        
        log_info(f"学习到新的映射: {field_name} -> {standard_name}在{sheet_type}", "FieldMapper")
        
        return True
        
    def suggest_mappings(self, field_name, sheet_type="balance_sheet", top_n=3):
        """
        建议可能的字段映射
        
        Args:
            field_name: 字段名
            sheet_type: 表格类型 ('balance_sheet' 或 'income_statement')
            top_n: 返回的建议数量
            
        Returns:
            list: 建议的映射列表 [(标准字段名, 匹配度), ...]
        """
        if not field_name or not isinstance(field_name, str):
            return []
            
        # 根据表格类型选择映射字典
        if sheet_type == "balance_sheet":
            mappings = self.balance_sheet_mappings
        elif sheet_type == "income_statement":
            mappings = self.income_statement_mappings
        else:
            return []
            
        # 清理字段名
        field_name = field_name.strip().lower()
        
        # 存储匹配结果
        matches = []
        
        # 对每个标准名称计算匹配度
        for standard_name, variants in mappings.items():
            # 与标准名称比较
            score = difflib.SequenceMatcher(None, field_name, standard_name.lower()).ratio()
            matches.append((standard_name, score))
            
            # 与变体比较
            for variant in variants:
                score = difflib.SequenceMatcher(None, field_name, variant.lower()).ratio()
                if score > next((s for n, s in matches if n == standard_name), 0):
                    # 更新该标准名称的最高分
                    matches = [(n, s) for n, s in matches if n != standard_name]
                    matches.append((standard_name, score))
                    
        # 对结果按匹配度排序并返回前N个
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_n]
        
    def get_learned_mappings(self):
        """
        获取学习到的新映射
        
        Returns:
            list: 学习到的新映射列表
        """
        return self.new_mappings
        
    def classify_field(self, field_name):
        """
        根据字段名推断其所属类别
        
        Args:
            field_name: 字段名
            
        Returns:
            str: 推断的类别 ('asset', 'liability', 'equity', 'revenue', 'cost', 'profit', 'unknown')
        """
        if not field_name or not isinstance(field_name, str):
            return "unknown"
            
        field_name = field_name.strip().lower()
        
        # 资产类关键词
        asset_keywords = [
            "资产", "资金", "现金", "存款", "应收", "固定资产", "无形资产", "在建", "长期", 
            "asset", "cash", "receivable", "inventory", "fixed", "intangible", "construction"
        ]
        
        # 负债类关键词
        liability_keywords = [
            "负债", "借款", "贷款", "应付", "应交", "预收", "债券", "税费", 
            "liability", "loan", "borrowing", "payable", "tax", "bond", "debt"
        ]
        
        # 权益类关键词
        equity_keywords = [
            "权益", "股东", "资本", "实收", "公积", "盈余", "未分配", "留存", 
            "equity", "shareholder", "capital", "reserve", "surplus", "retained"
        ]
        
        # 收入类关键词
        revenue_keywords = [
            "收入", "营业", "主营业务", "销售", 
            "revenue", "income", "sales", "turnover"
        ]
        
        # 成本类关键词
        cost_keywords = [
            "成本", "费用", "支出", "销售费", "管理费", "研发费", "财务费", 
            "cost", "expense", "expenditure", "selling", "administrative", "r&d", "financial"
        ]
        
        # 利润类关键词
        profit_keywords = [
            "利润", "盈利", "收益", "亏损", "毛利", "税前", "税后", "净利", 
            "profit", "earnings", "gain", "loss", "margin", "income"
        ]
        
        # 判断字段所属类别
        for keyword in asset_keywords:
            if keyword in field_name:
                return "asset"
                
        for keyword in liability_keywords:
            if keyword in field_name:
                return "liability"
                
        for keyword in equity_keywords:
            if keyword in field_name:
                return "equity"
                
        for keyword in revenue_keywords:
            if keyword in field_name:
                return "revenue"
                
        for keyword in cost_keywords:
            if keyword in field_name:
                return "cost"
                
        for keyword in profit_keywords:
            if keyword in field_name:
                return "profit"
                
        return "unknown" 