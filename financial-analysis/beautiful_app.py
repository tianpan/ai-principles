#!/usr/bin/env python
"""
财务分析应用程序 - 美化版
实现了现代简约界面、响应式设计和良好用户体验的财务分析工具
"""
import sys
import os

# 将当前目录添加到Python路径中，确保可以正确导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QFileDialog, QWidget,
                           QProgressBar, QMessageBox, QComboBox, QTextEdit, 
                           QTabWidget, QGroupBox, QCheckBox, QFrame, QSizePolicy,
                           QGridLayout, QSpacerItem, QToolTip, QStatusBar)
from PyQt6.QtCore import Qt, QSize, QEasingCurve, QPropertyAnimation, QRect
from PyQt6.QtGui import QFont, QIcon, QPixmap, QColor, QPalette, QCursor

from financial_analysis.processors.excel_processor import ExcelProcessor
from financial_analysis.processors.report_generator import ReportGenerator
from financial_analysis.logger import initialize_logging, log_info, log_debug, log_warning, log_error, LogViewerDialog

# 定义样式常量
MAIN_COLOR = "#2c3e50"  # 深蓝色作为主色调
ACCENT_COLOR = "#3498db"  # 亮蓝色作为强调色
BG_COLOR = "#f5f5f5"  # 浅灰色作为背景色
TEXT_COLOR = "#333333"  # 深灰色作为文本颜色
SUCCESS_COLOR = "#2ecc71"  # 绿色作为成功状态颜色
WARNING_COLOR = "#f39c12"  # 橙色作为警告状态颜色
ERROR_COLOR = "#e74c3c"  # 红色作为错误状态颜色

# 按钮样式
BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {ACCENT_COLOR};
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: #2980b9;
    }}
    QPushButton:pressed {{
        background-color: #1a5276;
    }}
    QPushButton:disabled {{
        background-color: #bdc3c7;
        color: #7f8c8d;
    }}
"""

# 进度条样式
PROGRESS_BAR_STYLE = f"""
    QProgressBar {{
        border: none;
        background-color: #e0e0e0;
        border-radius: 5px;
        text-align: center;
        color: {TEXT_COLOR};
        font-weight: bold;
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT_COLOR};
        border-radius: 5px;
    }}
"""

# 标签样式
TITLE_STYLE = f"""
    color: {MAIN_COLOR};
    font-size: 24px;
    font-weight: bold;
"""

# 分组框样式
GROUP_BOX_STYLE = f"""
    QGroupBox {{
        border: 1px solid #cccccc;
        border-radius: 5px;
        margin-top: 10px;
        font-weight: bold;
        color: {MAIN_COLOR};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }}
"""

# 下拉框样式
COMBOBOX_STYLE = f"""
    QComboBox {{
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 4px 8px;
        background-color: white;
        color: {TEXT_COLOR};
    }}
    QComboBox::drop-down {{
        width: 20px;
        border-left: 1px solid #cccccc;
    }}
    QComboBox:hover {{
        border: 1px solid {ACCENT_COLOR};
    }}
"""

class ModernButton(QPushButton):
    """现代风格按钮，带有悬停和点击效果"""
    def __init__(self, text, parent=None, is_primary=True):
        super().__init__(text, parent)
        self.is_primary = is_primary
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet(BUTTON_STYLE)
        
        if not is_primary:
            self.setStyleSheet(BUTTON_STYLE.replace(ACCENT_COLOR, "#95a5a6"))
            
    def enterEvent(self, event):
        # 鼠标悬停动画效果
        self.setStyleSheet(self.styleSheet().replace("border-radius: 4px", "border-radius: 5px"))
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        # 鼠标离开动画效果
        self.setStyleSheet(self.styleSheet().replace("border-radius: 5px", "border-radius: 4px"))
        super().leaveEvent(event)

class BeautifulFinancialAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.balance_sheet_path = ""
        self.income_statement_path = ""
        self.output_folder = ""
        
        # 初始化日志
        self.logger = initialize_logging()
        log_info("Application initialization started", "FinancialAnalysisApp")
        
        # 设置窗口基本属性
        self.setWindowTitle("港华集团财务分析工具")
        self.setGeometry(100, 100, 900, 700)
        self.setMinimumSize(800, 600)
        
        # 设置应用程序图标
        self.icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pic", "港华logo.png")
        if os.path.exists(self.icon_path):
            self.setWindowIcon(QIcon(self.icon_path))
        
        # 设置状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("准备就绪")
        
        # 初始化UI
        self.init_ui()
        log_info("Application initialization completed", "FinancialAnalysisApp")
        
    def init_ui(self):
        log_debug("Setting up user interface", "FinancialAnalysisApp")
        
        # 创建主窗口部件和布局
        main_widget = QWidget()
        main_widget.setStyleSheet(f"background-color: {BG_COLOR}; color: {TEXT_COLOR};")
        
        # 使用网格布局来实现响应式设计
        main_layout = QGridLayout(main_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 添加标题和Logo区域
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        
        # 添加Logo
        logo_label = QLabel()
        if os.path.exists(self.icon_path):
            logo_pixmap = QPixmap(self.icon_path)
            logo_pixmap = logo_pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
        else:
            log_warning("未找到Logo图片，使用文本替代", "FinancialAnalysisApp")
            logo_label.setText("港华集团")
            logo_label.setStyleSheet("font-weight: bold; color: " + MAIN_COLOR)
        
        logo_label.setMaximumSize(80, 80)
        header_layout.addWidget(logo_label)
        
        # 添加标题
        title_label = QLabel("港华集团财务分析工具")
        title_label.setStyleSheet(TITLE_STYLE)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label, stretch=1)
        
        # 添加间距，确保标题居中
        header_layout.addSpacing(80)
        
        # 将头部添加到主布局
        main_layout.addWidget(header_widget, 0, 0, 1, 2)
        
        # 创建文件选择分组框
        file_group = QGroupBox("文件选择")
        file_group.setStyleSheet(GROUP_BOX_STYLE)
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(15)
        
        # 资产负债表选择
        balance_sheet_layout = QHBoxLayout()
        balance_sheet_label = QLabel("选择资产负债表:")
        self.balance_sheet_path_label = QLabel("未选择文件")
        self.balance_sheet_path_label.setStyleSheet("color: #7f8c8d;")
        balance_sheet_button = ModernButton("浏览...", is_primary=False)
        balance_sheet_button.clicked.connect(self.select_balance_sheet)
        balance_sheet_button.setToolTip("选择资产负债表Excel文件")
        balance_sheet_layout.addWidget(balance_sheet_label)
        balance_sheet_layout.addWidget(self.balance_sheet_path_label, stretch=1)
        balance_sheet_layout.addWidget(balance_sheet_button)
        file_layout.addLayout(balance_sheet_layout)
        
        # 损益表选择
        income_statement_layout = QHBoxLayout()
        income_statement_label = QLabel("选择损益表:")
        self.income_statement_path_label = QLabel("未选择文件")
        self.income_statement_path_label.setStyleSheet("color: #7f8c8d;")
        income_statement_button = ModernButton("浏览...", is_primary=False)
        income_statement_button.clicked.connect(self.select_income_statement)
        income_statement_button.setToolTip("选择损益表Excel文件")
        income_statement_layout.addWidget(income_statement_label)
        income_statement_layout.addWidget(self.income_statement_path_label, stretch=1)
        income_statement_layout.addWidget(income_statement_button)
        file_layout.addLayout(income_statement_layout)
        
        # 输出文件夹选择
        output_folder_layout = QHBoxLayout()
        output_folder_label = QLabel("选择输出文件夹:")
        self.output_folder_label = QLabel("未选择文件夹")
        self.output_folder_label.setStyleSheet("color: #7f8c8d;")
        output_folder_button = ModernButton("浏览...", is_primary=False)
        output_folder_button.clicked.connect(self.select_output_folder)
        output_folder_button.setToolTip("选择报告输出文件夹")
        output_folder_layout.addWidget(output_folder_label)
        output_folder_layout.addWidget(self.output_folder_label, stretch=1)
        output_folder_layout.addWidget(output_folder_button)
        file_layout.addLayout(output_folder_layout)
        
        # 添加文件选择区域到主布局
        main_layout.addWidget(file_group, 1, 0, 1, 1)
        
        # 创建分析选项分组框
        options_group = QGroupBox("分析选项")
        options_group.setStyleSheet(GROUP_BOX_STYLE)
        options_layout = QVBoxLayout(options_group)
        
        # 时间维度选择
        time_dimension_layout = QHBoxLayout()
        time_dimension_label = QLabel("时间维度:")
        self.time_dimension_combo = QComboBox()
        self.time_dimension_combo.addItems(["月度", "季度", "年度"])
        self.time_dimension_combo.setStyleSheet(COMBOBOX_STYLE)
        self.time_dimension_combo.setToolTip("选择财务分析的时间维度")
        time_dimension_layout.addWidget(time_dimension_label)
        time_dimension_layout.addWidget(self.time_dimension_combo, stretch=1)
        options_layout.addLayout(time_dimension_layout)
        
        # 留出空间以便将来添加更多选项
        options_layout.addStretch()
        
        # 添加分析选项区域到主布局
        main_layout.addWidget(options_group, 1, 1, 1, 1)
        
        # 进度区域
        progress_group = QGroupBox("处理进度")
        progress_group.setStyleSheet(GROUP_BOX_STYLE)
        progress_layout = QVBoxLayout(progress_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.progress_bar)
        
        # 将进度区域添加到主布局
        main_layout.addWidget(progress_group, 2, 0, 1, 2)
        
        # 按钮区域
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        # 生成报告按钮
        self.generate_button = ModernButton("生成分析报告")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate_report)
        self.generate_button.setMinimumWidth(150)
        self.generate_button.setToolTip("处理选定的Excel文件并生成财务分析报告")
        
        # 查看日志按钮
        self.view_logs_button = ModernButton("查看日志", is_primary=False)
        self.view_logs_button.clicked.connect(self.show_logs)
        self.view_logs_button.setMinimumWidth(100)
        self.view_logs_button.setToolTip("查看应用程序运行日志")
        
        # 添加按钮到布局
        button_layout.addWidget(self.generate_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.view_logs_button)
        
        # 将按钮区域添加到主布局
        main_layout.addWidget(button_widget, 3, 0, 1, 2)
        
        # 设置网格布局的伸缩因子
        main_layout.setRowStretch(1, 1)
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)
        
        # 设置中央窗口部件
        self.setCentralWidget(main_widget)
        
        # 设置字体
        app_font = QFont("微软雅黑", 10)
        QApplication.setFont(app_font)
        
        # 设置工具提示字体（只设置字体，不设置样式）
        QToolTip.setFont(QFont("微软雅黑", 9))
        
        # 设置窗口为可拖放
        self.setAcceptDrops(True)
        
        log_debug("User interface setup completed", "FinancialAnalysisApp")
        
    def select_balance_sheet(self):
        log_debug("Opening balance sheet file selection dialog", "FinancialAnalysisApp")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择资产负债表", "", "Excel文件 (*.xlsx *.xls)"
        )
        if file_path:
            self.balance_sheet_path = file_path
            file_name = os.path.basename(file_path)
            self.balance_sheet_path_label.setText(file_name)
            self.balance_sheet_path_label.setStyleSheet("color: " + TEXT_COLOR + "; font-weight: bold;")
            self.balance_sheet_path_label.setToolTip(file_path)
            log_info(f"Balance sheet selected: {file_path}", "FinancialAnalysisApp")
            self.statusBar.showMessage(f"已选择资产负债表: {file_name}", 3000)
            self.check_generate_button()
        else:
            log_debug("Balance sheet selection canceled", "FinancialAnalysisApp")
            
    def select_income_statement(self):
        log_debug("Opening income statement file selection dialog", "FinancialAnalysisApp")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择损益表", "", "Excel文件 (*.xlsx *.xls)"
        )
        if file_path:
            self.income_statement_path = file_path
            file_name = os.path.basename(file_path)
            self.income_statement_path_label.setText(file_name)
            self.income_statement_path_label.setStyleSheet("color: " + TEXT_COLOR + "; font-weight: bold;")
            self.income_statement_path_label.setToolTip(file_path)
            log_info(f"Income statement selected: {file_path}", "FinancialAnalysisApp")
            self.statusBar.showMessage(f"已选择损益表: {file_name}", 3000)
            self.check_generate_button()
        else:
            log_debug("Income statement selection canceled", "FinancialAnalysisApp")
            
    def select_output_folder(self):
        log_debug("Opening output folder selection dialog", "FinancialAnalysisApp")
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder_path:
            self.output_folder = folder_path
            self.output_folder_label.setText(folder_path)
            self.output_folder_label.setStyleSheet("color: " + TEXT_COLOR + "; font-weight: bold;")
            self.output_folder_label.setToolTip(folder_path)
            log_info(f"Output folder selected: {folder_path}", "FinancialAnalysisApp")
            self.statusBar.showMessage(f"已选择输出文件夹: {folder_path}", 3000)
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
        log_info("Starting financial report generation", "FinancialAnalysisApp")
        self.statusBar.showMessage("正在生成财务分析报告...", 0)
        try:
            # 更新进度条
            self.progress_bar.setValue(10)
            log_debug("Progress updated: 10%", "FinancialAnalysisApp")
            
            # 处理Excel文件
            log_info("Initializing Excel processor", "FinancialAnalysisApp")
            excel_processor = ExcelProcessor(
                self.balance_sheet_path, 
                self.income_statement_path,
                template_dir=os.path.join(os.getcwd(), "templates")
            )
            
            # 解析Excel文件
            log_info("Processing balance sheet data", "FinancialAnalysisApp")
            self.statusBar.showMessage("正在处理资产负债表数据...", 0)
            self.progress_bar.setValue(30)
            log_debug("Progress updated: 30%", "FinancialAnalysisApp")
            balance_sheet_data = excel_processor.process_balance_sheet()
            
            log_info("Processing income statement data", "FinancialAnalysisApp")
            self.statusBar.showMessage("正在处理损益表数据...", 0)
            self.progress_bar.setValue(50)
            log_debug("Progress updated: 50%", "FinancialAnalysisApp")
            income_statement_data = excel_processor.process_income_statement()
            
            # 生成报告
            log_info("Initializing report generator", "FinancialAnalysisApp")
            self.statusBar.showMessage("正在初始化报告生成器...", 0)
            self.progress_bar.setValue(70)
            log_debug("Progress updated: 70%", "FinancialAnalysisApp")
            report_generator = ReportGenerator(
                balance_sheet_data,
                income_statement_data,
                self.time_dimension_combo.currentText()
            )
            
            # 保存报告
            output_path = os.path.join(
                self.output_folder, 
                f"财务分析报告_{excel_processor.company_name}.docx"
            )
            log_info(f"Generating report file: {output_path}", "FinancialAnalysisApp")
            self.statusBar.showMessage(f"正在生成报告文件: {os.path.basename(output_path)}...", 0)
            self.progress_bar.setValue(90)
            log_debug("Progress updated: 90%", "FinancialAnalysisApp")
            report_generator.generate_report(output_path)
            
            # 完成
            self.progress_bar.setValue(100)
            log_debug("Progress updated: 100%", "FinancialAnalysisApp")
            log_info("Financial report generation completed", "FinancialAnalysisApp")
            self.statusBar.showMessage("报告生成完成", 5000)
            
            # 显示成功消息
            QMessageBox.information(
                self, 
                "完成", 
                f"财务分析报告已生成，保存在：\n{output_path}"
            )
            
        except Exception as e:
            log_error("Error generating report", e, "FinancialAnalysisApp")
            self.statusBar.showMessage("报告生成失败", 5000)
            QMessageBox.critical(self, "错误", f"生成报告时出错：\n{str(e)}")
            self.progress_bar.setValue(0)
    
    def show_logs(self):
        """Display the log viewer dialog"""
        log_debug("Opening log viewer", "FinancialAnalysisApp")
        log_dialog = LogViewerDialog(self)
        log_dialog.exec()
        
    def resizeEvent(self, event):
        """处理窗口大小调整事件，确保界面元素自适应"""
        super().resizeEvent(event)
        # 可以在这里添加响应式设计相关代码
        
    def dragEnterEvent(self, event):
        """处理拖拽进入事件，支持文件拖拽"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        """处理文件拖放事件"""
        urls = event.mimeData().urls()
        if not urls:
            return
            
        file_path = urls[0].toLocalFile()
        
        # 根据文件类型判断处理方式
        if file_path.lower().endswith(('.xlsx', '.xls')):
            # 简单判断是资产负债表还是损益表
            if '资产' in file_path or 'balance' in file_path.lower():
                self.balance_sheet_path = file_path
                self.balance_sheet_path_label.setText(os.path.basename(file_path))
                self.balance_sheet_path_label.setStyleSheet("color: " + TEXT_COLOR + "; font-weight: bold;")
                self.statusBar.showMessage(f"已选择资产负债表: {os.path.basename(file_path)}", 3000)
            elif '损益' in file_path or 'income' in file_path.lower():
                self.income_statement_path = file_path
                self.income_statement_path_label.setText(os.path.basename(file_path))
                self.income_statement_path_label.setStyleSheet("color: " + TEXT_COLOR + "; font-weight: bold;")
                self.statusBar.showMessage(f"已选择损益表: {os.path.basename(file_path)}", 3000)
            else:
                # 如果无法判断，询问用户
                msg = QMessageBox()
                msg.setWindowTitle("选择文件类型")
                msg.setText(f"请选择拖入的文件 '{os.path.basename(file_path)}' 类型:")
                msg.addButton("资产负债表", QMessageBox.ButtonRole.AcceptRole)
                msg.addButton("损益表", QMessageBox.ButtonRole.RejectRole)
                msg.addButton("取消", QMessageBox.ButtonRole.DestructiveRole)
                
                ret = msg.exec()
                
                if ret == 0:  # 资产负债表
                    self.balance_sheet_path = file_path
                    self.balance_sheet_path_label.setText(os.path.basename(file_path))
                    self.balance_sheet_path_label.setStyleSheet("color: " + TEXT_COLOR + "; font-weight: bold;")
                    self.statusBar.showMessage(f"已选择资产负债表: {os.path.basename(file_path)}", 3000)
                elif ret == 1:  # 损益表
                    self.income_statement_path = file_path
                    self.income_statement_path_label.setText(os.path.basename(file_path))
                    self.income_statement_path_label.setStyleSheet("color: " + TEXT_COLOR + "; font-weight: bold;")
                    self.statusBar.showMessage(f"已选择损益表: {os.path.basename(file_path)}", 3000)
        
        # 检查是否可以启用生成按钮
        self.check_generate_button()

def main():
    """应用程序主入口"""
    # 初始化日志
    initialize_logging()
    log_info("财务分析程序启动", "beautiful_app")
    
    # 启动应用程序
    app = QApplication(sys.argv)
    
    # 设置工具提示样式 - 应用到app实例而不是QApplication类
    tooltip_style = f"""
        QToolTip {{
            background-color: {MAIN_COLOR};
            color: white;
            border: none;
            padding: 5px;
        }}
    """
    app.setStyleSheet(tooltip_style)
    
    window = BeautifulFinancialAnalysisApp()
    window.show()
    log_info("应用程序窗口已显示", "beautiful_app")
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 