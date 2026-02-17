"""
财务分析程序入口模块
"""

from .main import FinancialAnalysisApp
import sys
from PyQt6.QtWidgets import QApplication
# Import logging functions
from .logger import initialize_logging, log_info

def main():
    """主函数"""
    # Initialize logging
    logger = initialize_logging()
    log_info("财务分析程序启动", "main")
    
    # Start application
    app = QApplication(sys.argv)
    window = FinancialAnalysisApp()
    window.show()
    log_info("应用程序窗口已显示", "main")
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 