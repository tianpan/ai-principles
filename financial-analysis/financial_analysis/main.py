import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QFileDialog, QWidget,
                           QProgressBar, QMessageBox, QComboBox, QTextEdit, 
                           QTabWidget, QGroupBox, QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
import time

from financial_analysis.processors.excel_processor import ExcelProcessor
from financial_analysis.processors.report_generator import ReportGenerator
from financial_analysis.processors.excel_capture_logger import ExcelCaptureLogger
# Import the logging components
from financial_analysis.logger import initialize_logging, log_info, log_debug, log_warning, log_error, LogViewerDialog

# Initialize logging at module level
initialize_logging()

class FinancialAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.balance_sheet_path = ""
        self.income_statement_path = ""
        self.output_folder = ""
        log_info("Application initialization started", "FinancialAnalysisApp")
        self.init_ui()
        log_info("Application initialization completed", "FinancialAnalysisApp")
        
    def init_ui(self):
        log_debug("Setting up user interface", "FinancialAnalysisApp")
        self.setWindowTitle("港华集团财务分析工具")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("港华集团财务分析工具")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 文件选择区域
        file_selection_layout = QVBoxLayout()
        
        # 资产负债表选择
        balance_sheet_layout = QHBoxLayout()
        balance_sheet_label = QLabel("选择资产负债表:")
        self.balance_sheet_path_label = QLabel("未选择文件")
        balance_sheet_button = QPushButton("浏览...")
        balance_sheet_button.clicked.connect(self.select_balance_sheet)
        balance_sheet_layout.addWidget(balance_sheet_label)
        balance_sheet_layout.addWidget(self.balance_sheet_path_label)
        balance_sheet_layout.addWidget(balance_sheet_button)
        file_selection_layout.addLayout(balance_sheet_layout)
        
        # 损益表选择
        income_statement_layout = QHBoxLayout()
        income_statement_label = QLabel("选择损益表:")
        self.income_statement_path_label = QLabel("未选择文件")
        income_statement_button = QPushButton("浏览...")
        income_statement_button.clicked.connect(self.select_income_statement)
        income_statement_layout.addWidget(income_statement_label)
        income_statement_layout.addWidget(self.income_statement_path_label)
        income_statement_layout.addWidget(income_statement_button)
        file_selection_layout.addLayout(income_statement_layout)
        
        # 输出文件夹选择
        output_folder_layout = QHBoxLayout()
        output_folder_label = QLabel("选择输出文件夹:")
        self.output_folder_label = QLabel("未选择文件夹")
        output_folder_button = QPushButton("浏览...")
        output_folder_button.clicked.connect(self.select_output_folder)
        output_folder_layout.addWidget(output_folder_label)
        output_folder_layout.addWidget(self.output_folder_label)
        output_folder_layout.addWidget(output_folder_button)
        file_selection_layout.addLayout(output_folder_layout)
        
        main_layout.addLayout(file_selection_layout)
        
        # 分析选项
        analysis_options_layout = QVBoxLayout()
        analysis_options_label = QLabel("分析选项:")
        analysis_options_layout.addWidget(analysis_options_label)
        
        # 时间维度选择
        time_dimension_layout = QHBoxLayout()
        time_dimension_label = QLabel("时间维度:")
        self.time_dimension_combo = QComboBox()
        self.time_dimension_combo.addItems(["月度", "季度", "年度"])
        time_dimension_layout.addWidget(time_dimension_label)
        time_dimension_layout.addWidget(self.time_dimension_combo)
        analysis_options_layout.addLayout(time_dimension_layout)
        
        main_layout.addLayout(analysis_options_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        self.generate_button = QPushButton("生成分析报告")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate_report)
        button_layout.addWidget(self.generate_button)
        
        # 添加日志查看按钮
        self.view_logs_button = QPushButton("查看日志")
        self.view_logs_button.clicked.connect(self.show_logs)
        button_layout.addWidget(self.view_logs_button)
        
        main_layout.addLayout(button_layout)
        
        # 设置主窗口
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        log_debug("User interface setup completed", "FinancialAnalysisApp")
        
    def select_balance_sheet(self):
        log_debug("Opening balance sheet file selection dialog", "FinancialAnalysisApp")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择资产负债表", "", "Excel文件 (*.xlsx *.xls)"
        )
        if file_path:
            # 检查文件是否可读
            try:
                with open(file_path, 'rb') as f:
                    pass
                self.balance_sheet_path = file_path
                self.balance_sheet_path_label.setText(os.path.basename(file_path))
                log_info(f"Balance sheet selected: {file_path}", "FinancialAnalysisApp")
                self.check_generate_button()
            except (PermissionError, OSError) as e:
                log_error(f"无法访问所选文件: {e}", e, "FinancialAnalysisApp")
                QMessageBox.critical(self, "错误", f"所选文件无法访问: {str(e)}")
        else:
            log_debug("Balance sheet selection canceled", "FinancialAnalysisApp")
            
    def select_income_statement(self):
        log_debug("Opening income statement file selection dialog", "FinancialAnalysisApp")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择损益表", "", "Excel文件 (*.xlsx *.xls)"
        )
        if file_path:
            # 检查文件是否可读
            try:
                with open(file_path, 'rb') as f:
                    pass
                self.income_statement_path = file_path
                self.income_statement_path_label.setText(os.path.basename(file_path))
                log_info(f"Income statement selected: {file_path}", "FinancialAnalysisApp")
                self.check_generate_button()
            except (PermissionError, OSError) as e:
                log_error(f"无法访问所选文件: {e}", e, "FinancialAnalysisApp")
                QMessageBox.critical(self, "错误", f"所选文件无法访问: {str(e)}")
        else:
            log_debug("Income statement selection canceled", "FinancialAnalysisApp")
            
    def select_output_folder(self):
        log_debug("Opening output folder selection dialog", "FinancialAnalysisApp")
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder_path:
            # 检查目录是否可写
            try:
                test_file = os.path.join(folder_path, f"test_write_{int(time.time())}.tmp")
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                
                self.output_folder = folder_path
                self.output_folder_label.setText(folder_path)
                log_info(f"Output folder selected: {folder_path}", "FinancialAnalysisApp")
                self.check_generate_button()
            except (PermissionError, OSError) as e:
                log_error(f"所选目录没有写入权限: {e}", e, "FinancialAnalysisApp")
                QMessageBox.warning(
                    self, 
                    "警告", 
                    f"所选目录可能没有写入权限: {str(e)}\n报告将生成在系统临时目录中。"
                )
                # 使用临时目录作为备选
                import tempfile
                self.output_folder = tempfile.gettempdir()
                self.output_folder_label.setText(f"{folder_path} (无权限，将使用临时目录)")
                self.check_generate_button()
        else:
            log_debug("Output folder selection canceled", "FinancialAnalysisApp")
            
    def check_generate_button(self):
        if (self.balance_sheet_path and 
            self.income_statement_path and 
            self.output_folder):
            self.generate_button.setEnabled(True)
            log_debug("Generate report button enabled", "FinancialAnalysisApp")
        else:
            self.generate_button.setEnabled(False)
            
    def generate_report(self):
        log_info("开始生成财务分析报告", "FinancialAnalysisApp")
        try:
            # 更新进度条
            self.progress_bar.setValue(10)
            log_debug("进度更新: 10%", "FinancialAnalysisApp")
            
            # 检查文件存在性
            if not os.path.exists(self.balance_sheet_path):
                raise FileNotFoundError(f"资产负债表文件不存在: {self.balance_sheet_path}")
                
            if not os.path.exists(self.income_statement_path):
                raise FileNotFoundError(f"损益表文件不存在: {self.income_statement_path}")
            
            # 处理Excel文件
            log_info("初始化Excel处理器", "FinancialAnalysisApp")
            excel_processor = ExcelProcessor(
                self.balance_sheet_path, 
                self.income_statement_path,
                template_dir=os.path.join(os.getcwd(), "templates")
            )
            
            # 解析Excel文件
            log_info("处理资产负债表数据", "FinancialAnalysisApp")
            self.progress_bar.setValue(30)
            log_debug("进度更新: 30%", "FinancialAnalysisApp")
            
            try:
                balance_sheet_data = excel_processor.process_balance_sheet()
            except Exception as e:
                log_error(f"处理资产负债表时出错: {e}", e, "FinancialAnalysisApp")
                QMessageBox.warning(self, "警告", f"处理资产负债表时出错: {str(e)}\n将继续尝试生成报告，但结果可能不准确。")
                balance_sheet_data = {'assets': {}, 'liabilities': {}, 'equity': {}, 'dates': []}
            
            log_info("处理损益表数据", "FinancialAnalysisApp")
            self.progress_bar.setValue(50)
            log_debug("进度更新: 50%", "FinancialAnalysisApp")
            
            try:
                income_statement_data = excel_processor.process_income_statement()
            except Exception as e:
                log_error(f"处理损益表时出错: {e}", e, "FinancialAnalysisApp")
                QMessageBox.warning(self, "警告", f"处理损益表时出错: {str(e)}\n将继续尝试生成报告，但结果可能不准确。")
                income_statement_data = {'revenue': {}, 'costs': {}, 'profit': {}, 'dates': []}
            
            # 保存Excel数据捕获日志
            self.progress_bar.setValue(60)
            log_debug("进度更新: 60%", "FinancialAnalysisApp")
            log_info("保存Excel数据捕获日志", "FinancialAnalysisApp")
            
            try:
                log_paths = excel_processor.save_processing_logs()
                log_info(f"Excel数据捕获日志已保存: {log_paths[0]}, {log_paths[1]}", "FinancialAnalysisApp")
            except Exception as e:
                log_error(f"保存Excel数据捕获日志时出错: {e}", e, "FinancialAnalysisApp")
                log_paths = ["日志保存失败", "日志保存失败"]
            
            # 获取处理统计信息
            try:
                processing_stats = excel_processor.get_processing_statistics()
                log_info(f"Excel处理统计: 字段识别率 {processing_stats['capture_stats']['field_recognition_rate']:.2f}%, " +
                        f"数值捕获率 {processing_stats['capture_stats']['value_capture_rate']:.2f}%", 
                        "FinancialAnalysisApp")
            except Exception as e:
                log_error(f"获取处理统计信息时出错: {e}", e, "FinancialAnalysisApp")
                processing_stats = {
                    'capture_stats': {
                        'field_recognition_rate': 0,
                        'value_capture_rate': 0,
                        'anomaly_count': 0,
                        'error_count': 0
                    }
                }
            
            # 生成报告
            log_info("初始化报告生成器", "FinancialAnalysisApp")
            self.progress_bar.setValue(70)
            log_debug("进度更新: 70%", "FinancialAnalysisApp")
            report_generator = ReportGenerator(
                balance_sheet_data,
                income_statement_data,
                self.output_folder
            )
            
            log_info("生成财务分析报告", "FinancialAnalysisApp")
            report_path = report_generator.generate()
            self.progress_bar.setValue(90)
            log_debug("进度更新: 90%", "FinancialAnalysisApp")
            
            # 检查报告文件是否存在
            if not os.path.exists(report_path):
                raise FileNotFoundError(f"报告文件生成失败: {report_path}")
            
            # 完成
            self.progress_bar.setValue(100)
            log_info(f"财务分析报告生成完成: {report_path}", "FinancialAnalysisApp")
            
            # 显示成功消息
            QMessageBox.information(
                self, 
                "处理完成", 
                f"财务分析报告已生成到: {report_path}\n\n" +
                f"数据处理统计:\n" +
                f"- 字段识别率: {processing_stats['capture_stats']['field_recognition_rate']:.2f}%\n" +
                f"- 数值捕获率: {processing_stats['capture_stats']['value_capture_rate']:.2f}%\n" +
                f"- 发现的异常值: {processing_stats['capture_stats']['anomaly_count']}\n" +
                f"- 处理错误: {processing_stats['capture_stats']['error_count']}"
            )
            
        except Exception as e:
            self.progress_bar.setValue(0)
            error_msg = f"生成报告时出错: {str(e)}"
            log_error(error_msg, e, "FinancialAnalysisApp")
            QMessageBox.critical(self, "错误", error_msg)
    
    def show_logs(self):
        log_debug("打开日志查看器", "FinancialAnalysisApp")
        log_viewer = LogViewerDialog(self)
        log_viewer.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FinancialAnalysisApp()
    window.show()
    sys.exit(app.exec()) 