import os
import sys
import time
import logging
import logging.handlers
from datetime import datetime
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QTextEdit, QComboBox, QLabel, QLineEdit, 
                            QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QSize, pyqtSignal

# Configure logging system
class LoggerConfig:
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    
    LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s] - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_DIR = os.path.join(os.path.expanduser("~"), ".financial_analysis", "logs")
    LOG_FILE = os.path.join(LOG_DIR, "financial_analysis.log")
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT = 5
    
    @classmethod
    def setup(cls):
        # Create logs directory if it doesn't exist
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all levels
        
        # Clear existing handlers to avoid duplicates
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            cls.LOG_FILE,
            maxBytes=cls.MAX_LOG_SIZE,
            backupCount=cls.BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(cls.LOG_FORMAT, cls.DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Stream handler for console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(cls.LOG_FORMAT, cls.DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Log startup information
        logging.info("Financial Analysis application startup")
        logging.debug(f"Log directory: {cls.LOG_DIR}")
        
        return root_logger
    
    @classmethod
    def cleanup_logs(cls, days=30):
        """Clean up log files older than specified days"""
        try:
            now = time.time()
            cutoff_time = now - (days * 24 * 60 * 60)
            log_files = [f for f in os.listdir(cls.LOG_DIR) if f.startswith("financial_analysis.log.")]
            
            for file in log_files:
                file_path = os.path.join(cls.LOG_DIR, file)
                file_mod_time = os.path.getmtime(file_path)
                if file_mod_time < cutoff_time:
                    os.remove(file_path)
                    logging.info(f"Cleaned up old log file: {file}")
        except Exception as e:
            logging.error(f"Error during log cleanup: {str(e)}")


# QTextEdit subclass for log viewer with HTML formatting
class LogViewerTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.level_colors = {
            "DEBUG": "#808080",  # Gray
            "INFO": "#000000",   # Black
            "WARNING": "#FFA500", # Orange
            "ERROR": "#FF0000"   # Red
        }
        
    def append_log_line(self, line):
        """Append a log line with HTML formatting based on log level"""
        if not line.strip():
            return
            
        try:
            # Parse the log line to identify the log level
            parts = line.split("[", 2)
            if len(parts) >= 3:
                timestamp = parts[0].strip()
                level = parts[1].strip().rstrip("]")
                message = "[".join(parts[2:])
                
                color = self.level_colors.get(level, "#000000")
                html_line = f'<span style="color:{color}"><b>{timestamp} [{level}]</b> {message}</span>'
                super().append(html_line)
            else:
                # Fallback for lines that don't match expected format
                super().append(line)
        except Exception:
            # Just append the raw line if parsing fails
            super().append(line)


# Log viewer dialog
class LogViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("日志查看器")
        self.setMinimumSize(800, 600)
        self.setup_ui()
        self.current_filter_level = "ALL"
        self.current_search_text = ""
        self.load_logs()
        
    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout()
        
        # Controls area
        controls_layout = QHBoxLayout()
        
        # Level filter
        level_label = QLabel("日志级别:")
        self.level_combo = QComboBox()
        self.level_combo.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
        self.level_combo.currentTextChanged.connect(self.apply_filters)
        controls_layout.addWidget(level_label)
        controls_layout.addWidget(self.level_combo)
        
        # Search
        search_label = QLabel("搜索:")
        self.search_input = QLineEdit()
        self.search_input.returnPressed.connect(self.apply_filters)
        controls_layout.addWidget(search_label)
        controls_layout.addWidget(self.search_input)
        controls_layout.addStretch()
        
        # Log viewer
        self.log_viewer = LogViewerTextEdit()
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.load_logs)
        
        export_button = QPushButton("导出日志")
        export_button.clicked.connect(self.export_logs)
        
        clear_button = QPushButton("清空筛选")
        clear_button.clicked.connect(self.clear_filters)
        
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        
        buttons_layout.addWidget(refresh_button)
        buttons_layout.addWidget(export_button)
        buttons_layout.addWidget(clear_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(close_button)
        
        # Add all components to main layout
        layout.addLayout(controls_layout)
        layout.addWidget(self.log_viewer)
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        
    def load_logs(self):
        """Load and display logs with current filters applied"""
        self.log_viewer.clear()
        try:
            log_file = LoggerConfig.LOG_FILE
            if not os.path.exists(log_file):
                self.log_viewer.append("日志文件不存在")
                return
                
            with open(log_file, "r", encoding="utf-8") as file:
                lines = file.readlines()
                
            # Apply filters
            for line in lines:
                if self.should_display_line(line):
                    self.log_viewer.append_log_line(line)
                    
            # Scroll to bottom
            self.log_viewer.verticalScrollBar().setValue(
                self.log_viewer.verticalScrollBar().maximum()
            )
        except Exception as e:
            self.log_viewer.append(f"加载日志错误: {str(e)}")
    
    def should_display_line(self, line):
        """Check if the line should be displayed based on current filters"""
        # Level filter
        if self.current_filter_level != "ALL":
            if f"[{self.current_filter_level}]" not in line:
                return False
                
        # Search text filter
        if self.current_search_text and self.current_search_text not in line:
            return False
            
        return True
    
    def apply_filters(self):
        """Apply filters and reload logs"""
        self.current_filter_level = self.level_combo.currentText()
        self.current_search_text = self.search_input.text()
        self.load_logs()
    
    def clear_filters(self):
        """Clear all filters"""
        self.level_combo.setCurrentText("ALL")
        self.search_input.clear()
        self.current_filter_level = "ALL"
        self.current_search_text = ""
        self.load_logs()
    
    def export_logs(self):
        """Export logs to a file"""
        try:
            # Get file path for export
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出日志", "", "Log Files (*.log);;Text Files (*.txt);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Export based on current filters
            log_file = LoggerConfig.LOG_FILE
            with open(log_file, "r", encoding="utf-8") as src:
                lines = src.readlines()
                
            filtered_lines = [line for line in lines if self.should_display_line(line)]
            
            with open(file_path, "w", encoding="utf-8") as dest:
                dest.writelines(filtered_lines)
                
            QMessageBox.information(self, "导出成功", f"日志已成功导出到:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出日志时出错:\n{str(e)}")


# Initialize the logger
def initialize_logging():
    """Initialize the logging system"""
    logger = LoggerConfig.setup()
    # Perform cleanup of old logs
    LoggerConfig.cleanup_logs()
    return logger


# Convenience logging functions
def log_debug(message, module=None):
    """Log a debug message with optional module information"""
    if module:
        logging.debug(f"[{module}] {message}")
    else:
        logging.debug(message)
        
def log_info(message, module=None):
    """Log an info message with optional module information"""
    if module:
        logging.info(f"[{module}] {message}")
    else:
        logging.info(message)
        
def log_warning(message, module=None):
    """Log a warning message with optional module information"""
    if module:
        logging.warning(f"[{module}] {message}")
    else:
        logging.warning(message)
        
def log_error(message, error=None, module=None):
    """Log an error message with optional exception details and module information"""
    if error:
        error_details = f"{message}: {str(error)}"
        if module:
            logging.error(f"[{module}] {error_details}")
        else:
            logging.error(error_details)
    else:
        if module:
            logging.error(f"[{module}] {message}")
        else:
            logging.error(message)


# Helper function to get logger for a specific module
def get_logger(module_name):
    """Get a logger instance for a specific module"""
    return logging.getLogger(module_name) 