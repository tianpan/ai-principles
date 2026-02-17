# 港华集团财务分析工具

这是一个用于分析港华集团下属成员企业财务状况的工具，能够处理资产负债表和损益表的Excel文件，生成财务分析报告。

## 功能特点

- 自动识别并处理资产负债表和损益表Excel文件
- 计算关键财务指标（流动比率、资产负债率、净利润率等）
- 生成财务趋势分析图表
- 输出完整的Word格式财务分析报告
- 提供简洁易用的图形界面

## 安装说明

本工具基于Python 3.10开发，使用Poetry进行依赖管理。

1. 克隆项目代码：

```bash
git clone https://github.com/yourusername/financial-analysis.git
cd financial-analysis
```

2. 使用Poetry安装依赖：

```bash
poetry install
```

或者使用pip安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动程序：

```bash
poetry run python financial_analysis/main.py
```

或者：

```bash
python financial_analysis/main.py
```

2. 在图形界面中：
   - 选择资产负债表Excel文件
   - 选择损益表Excel文件
   - 选择报告输出文件夹
   - 设置分析选项
   - 点击"生成分析报告"按钮

3. 程序会自动处理数据并在指定位置生成Word格式的财务分析报告。

## 要求的Excel格式

- 资产负债表和损益表应为标准Excel格式（.xlsx或.xls）
- 表格结构可以灵活多变，程序会尝试自动识别关键财务数据
- 参考样例：ref文件夹中的示例文件

## 贡献指南

欢迎提交问题报告和改进建议！如果您想贡献代码，请遵循以下步骤：

1. Fork本项目
2. 创建您的特性分支 (`git checkout -b feature/amazingfeature`)
3. 提交您的更改 (`git commit -m 'Add some amazingfeature'`)
4. 推送到分支 (`git push origin feature/amazingfeature`)
5. 开启一个Pull Request

## 许可证信息

[MIT](LICENSE) © Your Name