"""
财务数据验证模块

用于验证处理后的财务数据，包括资产负债表平衡性检查、数据合理性验证等。
"""
import numpy as np
import pandas as pd

class DataValidator:
    """财务数据验证类"""
    
    def __init__(self, log_errors=True):
        """
        初始化数据验证器
        
        Args:
            log_errors: 是否记录错误信息
        """
        self.log_errors = log_errors
        self.validation_errors = []
        self.validation_warnings = []
        
    def clear_logs(self):
        """清除错误和警告日志"""
        self.validation_errors = []
        self.validation_warnings = []
        
    def _log_error(self, message):
        """
        记录错误信息
        
        Args:
            message: 错误信息
        """
        if self.log_errors:
            self.validation_errors.append(message)
            print(f"验证错误: {message}")
            
    def _log_warning(self, message):
        """
        记录警告信息
        
        Args:
            message: 警告信息
        """
        if self.log_errors:
            self.validation_warnings.append(message)
            print(f"验证警告: {message}")
    
    def validate_balance_sheet(self, balance_sheet_data):
        """
        验证资产负债表数据
        
        Args:
            balance_sheet_data: 处理后的资产负债表数据
            
        Returns:
            dict: 验证结果
        """
        self.clear_logs()
        
        is_valid = True
        validation_result = {
            "is_valid": True,
            "balance_check": {},
            "missing_data": [],
            "warnings": []
        }
        
        if not balance_sheet_data['dates']:
            self._log_error("未找到有效的日期数据")
            validation_result["is_valid"] = False
            return validation_result
            
        # 检查资产负债表平衡性
        for date in balance_sheet_data['dates']:
            total_assets = 0
            total_liabilities = 0
            total_equity = 0
            
            # 计算总资产
            for item, values in balance_sheet_data['assets'].items():
                if '资产总计' in item and date in values:
                    total_assets = values[date] if not np.isnan(values[date]) else 0
                    break
            
            # 如果没有找到资产总计，则尝试累加各项资产
            if total_assets == 0:
                for item, values in balance_sheet_data['assets'].items():
                    if date in values and not any(keyword in item.lower() for keyword in ['资产总计', '合计']):
                        total_assets += values[date] if not np.isnan(values[date]) else 0
            
            # 计算总负债
            for item, values in balance_sheet_data['liabilities'].items():
                if '负债合计' in item and date in values:
                    total_liabilities = values[date] if not np.isnan(values[date]) else 0
                    break
            
            # 如果没有找到负债合计，则尝试累加各项负债
            if total_liabilities == 0:
                for item, values in balance_sheet_data['liabilities'].items():
                    if date in values and not any(keyword in item.lower() for keyword in ['负债合计', '合计']):
                        total_liabilities += values[date] if not np.isnan(values[date]) else 0
            
            # 计算所有者权益
            for item, values in balance_sheet_data['equity'].items():
                if any(keyword in item for keyword in ['所有者权益合计', '股东权益合计']) and date in values:
                    total_equity = values[date] if not np.isnan(values[date]) else 0
                    break
            
            # 如果没有找到所有者权益合计，则尝试累加各项所有者权益
            if total_equity == 0:
                for item, values in balance_sheet_data['equity'].items():
                    if date in values and not any(keyword in item.lower() for keyword in ['所有者权益合计', '合计']):
                        total_equity += values[date] if not np.isnan(values[date]) else 0
            
            # 检查资产是否等于负债加所有者权益（允许小误差）
            liability_plus_equity = total_liabilities + total_equity
            difference = abs(total_assets - liability_plus_equity)
            tolerance = max(total_assets, liability_plus_equity) * 0.01  # 允许1%的误差
            
            validation_result["balance_check"][date] = {
                "assets": total_assets,
                "liabilities": total_liabilities,
                "equity": total_equity,
                "liability_plus_equity": liability_plus_equity,
                "difference": difference,
                "is_balanced": difference <= tolerance
            }
            
            if difference > tolerance:
                message = f"日期 {date} 的资产负债表不平衡: 资产 = {total_assets}, 负债+所有者权益 = {liability_plus_equity}, 差额 = {difference}"
                self._log_warning(message)
                validation_result["warnings"].append(message)
                is_valid = False
        
        # 检查关键数据项是否缺失
        key_asset_items = ['流动资产', '非流动资产', '资产总计']
        key_liability_items = ['流动负债', '非流动负债', '负债合计']
        key_equity_items = ['实收资本', '资本公积', '未分配利润', '所有者权益']
        
        for date in balance_sheet_data['dates']:
            # 检查资产
            for key_item in key_asset_items:
                found = False
                for item in balance_sheet_data['assets'].keys():
                    if key_item in item:
                        found = True
                        break
                        
                if not found:
                    message = f"日期 {date} 缺少关键资产项: {key_item}"
                    self._log_warning(message)
                    validation_result["missing_data"].append(message)
            
            # 检查负债
            for key_item in key_liability_items:
                found = False
                for item in balance_sheet_data['liabilities'].keys():
                    if key_item in item:
                        found = True
                        break
                        
                if not found:
                    message = f"日期 {date} 缺少关键负债项: {key_item}"
                    self._log_warning(message)
                    validation_result["missing_data"].append(message)
            
            # 检查所有者权益
            for key_item in key_equity_items:
                found = False
                for item in balance_sheet_data['equity'].keys():
                    if key_item in item:
                        found = True
                        break
                        
                if not found:
                    message = f"日期 {date} 缺少关键所有者权益项: {key_item}"
                    self._log_warning(message)
                    validation_result["missing_data"].append(message)
        
        validation_result["is_valid"] = is_valid
        return validation_result
    
    def validate_income_statement(self, income_statement_data):
        """
        验证损益表数据
        
        Args:
            income_statement_data: 处理后的损益表数据
            
        Returns:
            dict: 验证结果
        """
        self.clear_logs()
        
        is_valid = True
        validation_result = {
            "is_valid": True,
            "calculation_check": {},
            "missing_data": [],
            "warnings": []
        }
        
        if not income_statement_data['dates']:
            self._log_error("未找到有效的日期数据")
            validation_result["is_valid"] = False
            return validation_result
            
        # 检查损益表的计算逻辑
        for date in income_statement_data['dates']:
            total_revenue = 0
            total_costs = 0
            gross_profit = 0
            net_profit = 0
            
            # 获取收入
            for item, values in income_statement_data['revenue'].items():
                if '营业收入' in item and date in values:
                    total_revenue = values[date] if not np.isnan(values[date]) else 0
                    break
            
            # 获取成本
            for item, values in income_statement_data['costs'].items():
                if '营业成本' in item and date in values:
                    total_costs = values[date] if not np.isnan(values[date]) else 0
                    break
            
            # 获取毛利润
            calculated_gross_profit = total_revenue - total_costs
            
            for item, values in income_statement_data['profit'].items():
                if '毛利' in item and date in values:
                    gross_profit = values[date] if not np.isnan(values[date]) else 0
                    break
            
            # 获取净利润
            for item, values in income_statement_data['profit'].items():
                if '净利润' in item and date in values:
                    net_profit = values[date] if not np.isnan(values[date]) else 0
                    break
            
            # 检查毛利润计算是否合理（允许小误差）
            if gross_profit != 0:
                gp_difference = abs(calculated_gross_profit - gross_profit)
                gp_tolerance = max(abs(calculated_gross_profit), abs(gross_profit)) * 0.05  # 允许5%的误差
                
                gp_check = {
                    "revenue": total_revenue,
                    "costs": total_costs,
                    "calculated_gross_profit": calculated_gross_profit,
                    "reported_gross_profit": gross_profit,
                    "difference": gp_difference,
                    "is_reasonable": gp_difference <= gp_tolerance
                }
                
                if gp_difference > gp_tolerance:
                    message = f"日期 {date} 的毛利润计算可能有误: 收入 - 成本 = {calculated_gross_profit}, 报告毛利润 = {gross_profit}, 差额 = {gp_difference}"
                    self._log_warning(message)
                    validation_result["warnings"].append(message)
                    is_valid = False
            else:
                gp_check = {
                    "revenue": total_revenue,
                    "costs": total_costs,
                    "calculated_gross_profit": calculated_gross_profit,
                    "reported_gross_profit": "未找到",
                    "is_reasonable": True  # 如果没有报告毛利润，则不检查
                }
            
            validation_result["calculation_check"][date] = {
                "gross_profit": gp_check,
                "net_profit": net_profit
            }
        
        # 检查关键数据项是否缺失
        key_revenue_items = ['营业收入', '主营业务收入']
        key_cost_items = ['营业成本', '主营业务成本']
        key_profit_items = ['营业利润', '利润总额', '净利润']
        
        for date in income_statement_data['dates']:
            # 检查收入
            found_revenue = False
            for key_item in key_revenue_items:
                for item in income_statement_data['revenue'].keys():
                    if key_item in item:
                        found_revenue = True
                        break
                if found_revenue:
                    break
                    
            if not found_revenue:
                message = f"日期 {date} 缺少关键收入项"
                self._log_warning(message)
                validation_result["missing_data"].append(message)
            
            # 检查成本
            found_cost = False
            for key_item in key_cost_items:
                for item in income_statement_data['costs'].keys():
                    if key_item in item:
                        found_cost = True
                        break
                if found_cost:
                    break
                    
            if not found_cost:
                message = f"日期 {date} 缺少关键成本项"
                self._log_warning(message)
                validation_result["missing_data"].append(message)
            
            # 检查利润
            found_profit = False
            for key_item in key_profit_items:
                for item in income_statement_data['profit'].keys():
                    if key_item in item:
                        found_profit = True
                        break
                if found_profit:
                    break
                    
            if not found_profit:
                message = f"日期 {date} 缺少关键利润项"
                self._log_warning(message)
                validation_result["missing_data"].append(message)
        
        validation_result["is_valid"] = is_valid
        return validation_result
    
    def get_validation_report(self):
        """
        获取验证报告
        
        Returns:
            dict: 验证报告
        """
        return {
            "errors": self.validation_errors,
            "warnings": self.validation_warnings,
            "error_count": len(self.validation_errors),
            "warning_count": len(self.validation_warnings)
        } 