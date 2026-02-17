import pandas as pd

# Try to read the files using pandas, which can handle various encodings
try:
    # Try different encodings for the first file
    for encoding in ['utf-8', 'gbk', 'gb18030', 'latin1']:
        try:
            print(f"Trying to read file 1 with {encoding}...")
            df1 = pd.read_csv('1.板块收集.csv', encoding=encoding)
            print(f"Successfully read file 1 with {encoding}")
            print(f"Columns in file 1: {df1.columns.tolist()}")
            print(f"Sample from file 1:\n{df1.head()}")
            break
        except Exception as e:
            print(f"Error reading file 1 with {encoding}: {e}")
    
    # Try different encodings for the second file
    for encoding in ['utf-8', 'gbk', 'gb18030', 'latin1']:
        try:
            print(f"\nTrying to read file 2 with {encoding}...")
            df2 = pd.read_csv('2.TA800.csv', encoding=encoding)
            print(f"Successfully read file 2 with {encoding}")
            print(f"Columns in file 2: {df2.columns.tolist()}")
            print(f"Sample from file 2:\n{df2.head()}")
            break
        except Exception as e:
            print(f"Error reading file 2 with {encoding}: {e}")
    
    # Extract company names
    try:
        company_names1 = df1.iloc[:, 1].str.strip().tolist()  # Second column
        print(f"\nExtracted {len(company_names1)} company names from file 1")
        print(f"Sample companies from file 1: {company_names1[:5]}")
    except Exception as e:
        print(f"Error extracting company names from file 1: {e}")
        company_names1 = []
    
    try:
        company_names2 = df2.iloc[:, 2].str.strip().tolist()  # Third column
        print(f"Extracted {len(company_names2)} company names from file 2")
        print(f"Sample companies from file 2: {company_names2[:5]}")
    except Exception as e:
        print(f"Error extracting company names from file 2: {e}")
        company_names2 = []
    
    # Find exact matches
    exact_matches = []
    for name1 in company_names1:
        if name1 in company_names2:
            exact_matches.append((name1, name1))
    
    print(f"\nFound {len(exact_matches)} exact matches")
    print("First 10 exact matches:")
    for i, match in enumerate(exact_matches[:10]):
        print(f"{i+1}. {match[0]}")
    
    # Write results to file
    with open('simple_comparison_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== 精确匹配的公司 ===\n")
        for i, (name1, name2) in enumerate(exact_matches):
            f.write(f"{i+1}. {name1}\n")
        
        f.write("\n=== 文件1中的公司 (前20个) ===\n")
        for i, name in enumerate(company_names1[:20]):
            f.write(f"{i+1}. {name}\n")
        
        f.write("\n=== 文件2中的公司 (前20个) ===\n")
        for i, name in enumerate(company_names2[:20]):
            f.write(f"{i+1}. {name}\n")
    
    print("\nResults written to simple_comparison_results.txt")

except Exception as e:
    print(f"Error in script: {e}") 