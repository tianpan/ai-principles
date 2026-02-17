import re
from difflib import SequenceMatcher

def similarity(a, b):
    """Calculate string similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def detect_encoding_and_read(filename):
    """Try to detect encoding and read the file content."""
    encodings = ['utf-8', 'gbk', 'gb18030', 'latin1']
    
    # Read file in binary mode
    with open(filename, 'rb') as f:
        content = f.read()
    
    # Try different encodings
    for encoding in encodings:
        try:
            decoded = content.decode(encoding)
            lines = decoded.split('\n')
            print(f"Successfully decoded {filename} with {encoding}")
            print(f"First 5 lines:")
            for i, line in enumerate(lines[:5]):
                print(f"  {i+1}: {line}")
            return lines, encoding
        except UnicodeDecodeError:
            print(f"Failed to decode {filename} with {encoding}")
    
    print(f"Could not decode {filename} with any of the attempted encodings")
    return None, None

def parse_csv_lines(lines, file_num):
    """Parse CSV lines and extract company names."""
    companies = []
    
    for line in lines[1:]:  # Skip header
        if not line.strip():
            continue
        
        parts = line.split(',')
        if file_num == 1 and len(parts) >= 2:
            company = parts[1].strip()
            if company:
                companies.append(company)
        elif file_num == 2 and len(parts) >= 3:
            company = parts[2].strip()
            if company:
                companies.append(company)
    
    return companies

def clean_name(name):
    """Clean up company names for better matching."""
    name = name.replace('有限公司', '').replace('有限责任公司', '')
    name = name.replace('股份', '').replace('责任', '')
    name = name.replace('(', '').replace(')', '')
    name = name.replace('（', '').replace('）', '')
    name = re.sub(r'\s+', '', name)
    return name

# Main execution
print("Reading file 1...")
lines1, encoding1 = detect_encoding_and_read('1.板块收集.csv')
if not lines1:
    print("Failed to read file 1")
    exit(1)

print("\nReading file 2...")
lines2, encoding2 = detect_encoding_and_read('2.TA800.csv')
if not lines2:
    print("Failed to read file 2")
    exit(1)

# Extract company names
print("\nExtracting company names...")
companies1 = parse_csv_lines(lines1, 1)
companies2 = parse_csv_lines(lines2, 2)

print(f"Extracted {len(companies1)} companies from file 1")
print(f"First 5 companies from file 1: {companies1[:5]}")

print(f"Extracted {len(companies2)} companies from file 2")
print(f"First 5 companies from file 2: {companies2[:5]}")

# Clean company names
clean_companies1 = [clean_name(c) for c in companies1]
clean_companies2 = [clean_name(c) for c in companies2]

# Find exact matches
exact_matches = []
for i, name1 in enumerate(companies1):
    if name1 in companies2:
        exact_matches.append((name1, name1))

print(f"\nFound {len(exact_matches)} exact matches")
if exact_matches:
    print("First 5 exact matches:")
    for i, match in enumerate(exact_matches[:5]):
        print(f"  {i+1}: {match[0]}")

# Find similar matches
threshold = 0.7
similar_matches = []
for i, name1 in enumerate(companies1):
    clean1 = clean_companies1[i]
    for j, name2 in enumerate(companies2):
        clean2 = clean_companies2[j]
        if name1 != name2:  # Skip exact matches
            sim = similarity(clean1, clean2)
            if sim > threshold:
                similar_matches.append((name1, name2, sim))

# Sort by similarity
similar_matches.sort(key=lambda x: x[2], reverse=True)

print(f"\nFound {len(similar_matches)} similar matches")
if similar_matches:
    print("Top 10 similar matches:")
    for i, (name1, name2, sim) in enumerate(similar_matches[:10]):
        print(f"  {i+1}: {sim:.2f} | {name1} - {name2}")

# Write results to file
with open('binary_comparison_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== 精确匹配的公司 ===\n")
    for i, (name1, name2) in enumerate(exact_matches):
        f.write(f"{i+1}. {name1}\n")
    
    f.write("\n=== 相似匹配的公司 (前100个) ===\n")
    for i, (name1, name2, sim) in enumerate(similar_matches[:100]):
        f.write(f"{i+1}. 相似度: {sim:.2f} | 文件1: {name1} | 文件2: {name2}\n")
    
    f.write("\n=== 文件1中的公司 (前20个) ===\n")
    for i, name in enumerate(companies1[:20]):
        f.write(f"{i+1}. {name}\n")
    
    f.write("\n=== 文件2中的公司 (前20个) ===\n")
    for i, name in enumerate(companies2[:20]):
        f.write(f"{i+1}. {name}\n")

print("\nResults written to binary_comparison_results.txt") 