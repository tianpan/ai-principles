"""
Excel数据捕获日志模块

专门负责记录Excel处理过程中的详细信息，包括文件信息、字段识别、数值捕获、特殊处理、
数据验证结果、数据捕获统计及提供数据追踪支持。
"""
import os
import json
import pandas as pd
from datetime import datetime
from financial_analysis.logger import log_info, log_debug, log_warning, log_error

class ExcelCaptureLogger:
    """Excel数据捕获日志记录器，详细记录Excel数据处理过程"""
    
    def __init__(self, log_dir=None):
        """
        初始化Excel数据捕获日志记录器
        
        Args:
            log_dir: 日志保存目录，默认为用户目录下的.financial_analysis/excel_logs
        """
        # 设置日志目录
        if log_dir is None:
            self.log_dir = os.path.join(os.path.expanduser("~"), ".financial_analysis", "excel_logs")
        else:
            self.log_dir = log_dir
            
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志记录
        self.current_log = {
            "timestamp": datetime.now().isoformat(),
            "file_info": {},
            "field_recognition": {
                "fields": [],
                "statistics": {
                    "total_fields": 0,
                    "recognized_fields": 0,
                    "recognition_rate": 0
                }
            },
            "value_capture": {
                "values": [],
                "statistics": {
                    "total_values": 0,
                    "captured_values": 0,
                    "capture_rate": 0
                }
            },
            "special_processing": {
                "merged_cells": [],
                "hidden_rows_cols": [],
                "formulas": [],
                "references": []
            },
            "data_validation": {
                "completeness_checks": [],
                "logical_checks": [],
                "anomalies": [],
                "statistics": {
                    "total_checks": 0,
                    "passed_checks": 0,
                    "validation_rate": 0
                }
            },
            "data_tracking": {
                "sources": [],
                "error_locations": []
            }
        }
        
        log_info("初始化Excel数据捕获日志记录器", "ExcelCaptureLogger")
        
    def log_file_info(self, file_path, sheet_name=None, sheet_count=None, file_size=None):
        """
        记录Excel文件基本信息
        
        Args:
            file_path: Excel文件路径
            sheet_name: 当前处理的工作表名称
            sheet_count: 工作表总数
            file_size: 文件大小（字节）
        """
        # 获取文件信息
        file_name = os.path.basename(file_path)
        
        # 获取文件修改时间
        try:
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        except:
            mod_time = "未知"
            
        # 获取文件大小
        if file_size is None:
            try:
                file_size = os.path.getsize(file_path)
            except:
                file_size = 0
                
        # 更新日志
        self.current_log["file_info"] = {
            "file_name": file_name,
            "file_path": file_path,
            "sheet_name": sheet_name,
            "sheet_count": sheet_count,
            "file_size": file_size,
            "modified_time": mod_time
        }
        
        log_info(f"记录Excel文件信息: {file_name}, 工作表: {sheet_name}", "ExcelCaptureLogger")
        
    def log_data_range(self, data_shape, valid_range=None, header_range=None):
        """
        记录Excel数据范围信息
        
        Args:
            data_shape: 数据形状，格式为(行数, 列数)
            valid_range: 有效数据范围，格式为{start_row, end_row, start_col, end_col}
            header_range: 表头范围，格式为{start_row, end_row, start_col, end_col}
        """
        # 更新日志
        self.current_log["file_info"]["data_shape"] = data_shape
        
        if valid_range:
            self.current_log["file_info"]["valid_range"] = valid_range
            
        if header_range:
            self.current_log["file_info"]["header_range"] = header_range
            
        log_debug(f"记录数据范围: 形状{data_shape}, 有效范围{valid_range}", "ExcelCaptureLogger")
        
    def log_field_recognition(self, field_name, position, data_type, unit=None, confidence=0):
        """
        记录字段识别结果
        
        Args:
            field_name: 识别的字段名称
            position: 字段位置，格式为(行, 列)或单元格引用(如A1)
            data_type: 字段数据类型
            unit: 字段单位信息
            confidence: 识别置信度(0-100)
        """
        # 创建字段记录
        field_record = {
            "field_name": field_name,
            "position": position,
            "data_type": data_type,
            "unit": unit,
            "confidence": confidence
        }
        
        # 添加到日志
        self.current_log["field_recognition"]["fields"].append(field_record)
        
        # 更新统计信息
        stats = self.current_log["field_recognition"]["statistics"]
        stats["total_fields"] += 1
        if confidence > 50:  # 如果置信度大于50，认为识别成功
            stats["recognized_fields"] += 1
            
        # 计算识别率
        if stats["total_fields"] > 0:
            stats["recognition_rate"] = stats["recognized_fields"] / stats["total_fields"] * 100
            
        log_debug(f"记录字段识别: {field_name}, 位置{position}, 置信度{confidence}", "ExcelCaptureLogger")
        
    def log_value_capture(self, field_name, position, original_value, standardized_value, 
                          original_format=None, unit_conversion=None):
        """
        记录数值捕获详情
        
        Args:
            field_name: 对应的字段名称
            position: 值的位置，格式为(行, 列)或单元格引用(如A1)
            original_value: 原始值
            standardized_value: 标准化后的值
            original_format: 原始格式
            unit_conversion: 单位转换过程
        """
        # 创建值记录
        value_record = {
            "field_name": field_name,
            "position": position,
            "original_value": original_value,
            "standardized_value": standardized_value,
            "original_format": original_format,
            "unit_conversion": unit_conversion
        }
        
        # 添加到日志
        self.current_log["value_capture"]["values"].append(value_record)
        
        # 更新统计信息
        stats = self.current_log["value_capture"]["statistics"]
        stats["total_values"] += 1
        if standardized_value is not None:  # 如果标准化后的值不为空，认为捕获成功
            stats["captured_values"] += 1
            
        # 计算捕获率
        if stats["total_values"] > 0:
            stats["capture_rate"] = stats["captured_values"] / stats["total_values"] * 100
            
        log_debug(f"记录数值捕获: {field_name}, 原值{original_value}, 标准值{standardized_value}", 
                 "ExcelCaptureLogger")
        
    def log_merged_cell(self, range_str, value, processing_result):
        """
        记录合并单元格处理情况
        
        Args:
            range_str: 合并单元格范围，如"A1:B2"
            value: 单元格值
            processing_result: 处理结果
        """
        # 创建合并单元格记录
        merged_cell_record = {
            "range": range_str,
            "value": value,
            "processing_result": processing_result
        }
        
        # 添加到日志
        self.current_log["special_processing"]["merged_cells"].append(merged_cell_record)
        
        log_debug(f"记录合并单元格处理: {range_str}, 值{value}", "ExcelCaptureLogger")
        
    def log_hidden_row_col(self, type_str, index, is_hidden, processing_method):
        """
        记录隐藏行列处理情况
        
        Args:
            type_str: 类型，"row"或"column"
            index: 行号或列号
            is_hidden: 是否隐藏
            processing_method: 处理方式
        """
        # 创建隐藏行列记录
        hidden_record = {
            "type": type_str,
            "index": index,
            "is_hidden": is_hidden,
            "processing_method": processing_method
        }
        
        # 添加到日志
        self.current_log["special_processing"]["hidden_rows_cols"].append(hidden_record)
        
        log_debug(f"记录隐藏{type_str}处理: 索引{index}, 处理方式{processing_method}", 
                 "ExcelCaptureLogger")
        
    def log_formula(self, position, formula_text, calculated_value):
        """
        记录公式处理情况
        
        Args:
            position: 公式位置
            formula_text: 公式文本
            calculated_value: 计算结果
        """
        # 创建公式记录
        formula_record = {
            "position": position,
            "formula_text": formula_text,
            "calculated_value": calculated_value
        }
        
        # 添加到日志
        self.current_log["special_processing"]["formulas"].append(formula_record)
        
        log_debug(f"记录公式处理: 位置{position}, 公式{formula_text}", "ExcelCaptureLogger")
        
    def log_reference(self, source_position, target_position, reference_type):
        """
        记录引用关系
        
        Args:
            source_position: 源位置
            target_position: 目标位置
            reference_type: 引用类型
        """
        # 创建引用记录
        reference_record = {
            "source": source_position,
            "target": target_position,
            "type": reference_type
        }
        
        # 添加到日志
        self.current_log["special_processing"]["references"].append(reference_record)
        
        log_debug(f"记录引用关系: {source_position} -> {target_position}, 类型{reference_type}", 
                 "ExcelCaptureLogger")
        
    def log_data_validation(self, check_type, field_name, result, details=None):
        """
        记录数据验证结果
        
        Args:
            check_type: 检查类型，"completeness"或"logical"
            field_name: 字段名称
            result: 检查结果，True表示通过，False表示失败
            details: 详细信息
        """
        # 创建验证记录
        validation_record = {
            "check_type": check_type,
            "field_name": field_name,
            "result": result,
            "details": details or {}
        }
        
        # 添加到日志
        if check_type == "completeness":
            self.current_log["data_validation"]["completeness_checks"].append(validation_record)
        elif check_type == "logical":
            self.current_log["data_validation"]["logical_checks"].append(validation_record)
            
        # 更新统计信息
        stats = self.current_log["data_validation"]["statistics"]
        stats["total_checks"] += 1
        if result:  # 如果检查通过
            stats["passed_checks"] += 1
            
        # 计算验证通过率
        if stats["total_checks"] > 0:
            stats["validation_rate"] = stats["passed_checks"] / stats["total_checks"] * 100
            
        log_debug(f"记录数据验证: {check_type} 检查, 字段{field_name}, 结果{result}", 
                 "ExcelCaptureLogger")
        
    def log_anomaly(self, field_name, position, value, reason, severity="warning"):
        """
        记录异常值
        
        Args:
            field_name: 字段名称
            position: 异常值位置
            value: 异常值
            reason: 异常原因
            severity: 严重程度，"warning"或"error"
        """
        # 创建异常记录
        anomaly_record = {
            "field_name": field_name,
            "position": position,
            "value": value,
            "reason": reason,
            "severity": severity
        }
        
        # 添加到日志
        self.current_log["data_validation"]["anomalies"].append(anomaly_record)
        
        # 记录日志
        if severity == "warning":
            log_warning(f"异常值: {field_name} 在 {position}, 值{value}, 原因: {reason}", 
                       "ExcelCaptureLogger")
        else:
            log_error(f"严重异常值: {field_name} 在 {position}, 值{value}, 原因: {reason}", 
                     None, "ExcelCaptureLogger")
        
    def log_data_source(self, target_field, source_cell, processing_chain=None):
        """
        记录数据来源
        
        Args:
            target_field: 目标字段
            source_cell: 源单元格
            processing_chain: 处理链路
        """
        # 创建数据源记录
        source_record = {
            "target_field": target_field,
            "source_cell": source_cell,
            "processing_chain": processing_chain or []
        }
        
        # 添加到日志
        self.current_log["data_tracking"]["sources"].append(source_record)
        
        log_debug(f"记录数据来源: {target_field} <- {source_cell}", "ExcelCaptureLogger")
        
    def log_error_location(self, position, error_message, correction_suggestion=None):
        """
        记录错误位置
        
        Args:
            position: 错误位置
            error_message: 错误信息
            correction_suggestion: 修正建议
        """
        # 创建错误位置记录
        error_record = {
            "position": position,
            "error_message": error_message,
            "correction_suggestion": correction_suggestion
        }
        
        # 添加到日志
        self.current_log["data_tracking"]["error_locations"].append(error_record)
        
        log_error(f"错误位置: {position}, 错误: {error_message}", None, "ExcelCaptureLogger")
        
    def update_capture_statistics(self, field_count=None, recognized_count=None, 
                                 value_count=None, captured_count=None):
        """
        更新数据捕获统计信息
        
        Args:
            field_count: 字段总数
            recognized_count: 识别成功的字段数
            value_count: 值总数
            captured_count: 捕获成功的值数
        """
        # 更新字段识别统计
        if field_count is not None:
            self.current_log["field_recognition"]["statistics"]["total_fields"] = field_count
            
        if recognized_count is not None:
            self.current_log["field_recognition"]["statistics"]["recognized_fields"] = recognized_count
            
        # 计算识别率
        stats = self.current_log["field_recognition"]["statistics"]
        if stats["total_fields"] > 0:
            stats["recognition_rate"] = stats["recognized_fields"] / stats["total_fields"] * 100
            
        # 更新值捕获统计
        if value_count is not None:
            self.current_log["value_capture"]["statistics"]["total_values"] = value_count
            
        if captured_count is not None:
            self.current_log["value_capture"]["statistics"]["captured_values"] = captured_count
            
        # 计算捕获率
        stats = self.current_log["value_capture"]["statistics"]
        if stats["total_values"] > 0:
            stats["capture_rate"] = stats["captured_values"] / stats["total_values"] * 100
            
        log_info(f"更新捕获统计: 字段识别率 {self.current_log['field_recognition']['statistics']['recognition_rate']:.2f}%, " + 
                f"值捕获率 {self.current_log['value_capture']['statistics']['capture_rate']:.2f}%", 
                "ExcelCaptureLogger")
        
    def save_log(self, file_name=None):
        """
        保存当前日志
        
        Args:
            file_name: 日志文件名，默认使用时间戳和Excel文件名
            
        Returns:
            str: 保存的日志文件路径
        """
        # 生成文件名
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_name = os.path.basename(self.current_log["file_info"].get("file_path", "unknown"))
            excel_name = os.path.splitext(excel_name)[0]
            file_name = f"excel_log_{timestamp}_{excel_name}.json"
            
        # 构造文件路径
        log_path = os.path.join(self.log_dir, file_name)
        
        # 保存日志
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.current_log, f, ensure_ascii=False, indent=2)
            
        log_info(f"保存Excel捕获日志: {log_path}", "ExcelCaptureLogger")
        
        return log_path
        
    def get_capture_summary(self):
        """
        获取数据捕获摘要
        
        Returns:
            dict: 捕获摘要信息
        """
        # 创建摘要
        summary = {
            "file_name": self.current_log["file_info"].get("file_name", "未知"),
            "timestamp": self.current_log["timestamp"],
            "field_recognition_rate": self.current_log["field_recognition"]["statistics"]["recognition_rate"],
            "value_capture_rate": self.current_log["value_capture"]["statistics"]["capture_rate"],
            "validation_rate": self.current_log["data_validation"]["statistics"]["validation_rate"],
            "anomaly_count": len(self.current_log["data_validation"]["anomalies"]),
            "error_count": len(self.current_log["data_tracking"]["error_locations"])
        }
        
        return summary
        
    def reset_log(self):
        """重置当前日志，开始新的记录"""
        # 保存当前日志摘要统计
        summary = self.get_capture_summary()
        
        # 初始化新日志
        self.current_log = {
            "timestamp": datetime.now().isoformat(),
            "file_info": {},
            "field_recognition": {
                "fields": [],
                "statistics": {
                    "total_fields": 0,
                    "recognized_fields": 0,
                    "recognition_rate": 0
                }
            },
            "value_capture": {
                "values": [],
                "statistics": {
                    "total_values": 0,
                    "captured_values": 0,
                    "capture_rate": 0
                }
            },
            "special_processing": {
                "merged_cells": [],
                "hidden_rows_cols": [],
                "formulas": [],
                "references": []
            },
            "data_validation": {
                "completeness_checks": [],
                "logical_checks": [],
                "anomalies": [],
                "statistics": {
                    "total_checks": 0,
                    "passed_checks": 0,
                    "validation_rate": 0
                }
            },
            "data_tracking": {
                "sources": [],
                "error_locations": []
            },
            "previous_summary": summary
        }
        
        log_info("重置Excel捕获日志", "ExcelCaptureLogger") 