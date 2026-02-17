import csv
import re
from difflib import SequenceMatcher

def similarity(a, b):
    """Calculate string similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()

# Read first file (板块收集.csv)
companies_file1 = []
encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'latin1']

for encoding in encodings_to_try:
    try:
        print(f"Trying to read 1.板块收集.csv with {encoding} encoding...")
        with open('1.板块收集.csv', 'r', encoding=encoding) as f:
            reader = csv.reader(f)
            # Read a few lines to check if it's working
            first_rows = [next(reader) for _ in range(min(5, sum(1 for _ in open('1.板块收集.csv', 'r', encoding=encoding))))]
            print(f"First few rows from file 1 with {encoding}: {first_rows}")
            # Reopen the file to read from beginning
            f.seek(0)
            reader = csv.reader(f)
            next(reader, None)  # Skip header row
            for row in reader:
                if len(row) >= 2:
                    company_name = row[1].strip() if len(row[1].strip()) > 0 else "Unknown"
                    companies_file1.append(company_name)
        print(f"Successfully read {len(companies_file1)} companies from file 1")
        break
    except Exception as e:
        print(f"Error reading first file with {encoding}: {e}")
        companies_file1 = []

if not companies_file1:
    print("Failed to read the first file with any encoding")
    exit(1)

# Read second file (TA800.csv)
companies_file2 = []
for encoding in encodings_to_try:
    try:
        print(f"Trying to read 2.TA800.csv with {encoding} encoding...")
        with open('2.TA800.csv', 'r', encoding=encoding) as f:
            reader = csv.reader(f)
            # Read a few lines to check if it's working
            first_rows = [next(reader) for _ in range(min(5, sum(1 for _ in open('2.TA800.csv', 'r', encoding=encoding))))]
            print(f"First few rows from file 2 with {encoding}: {first_rows}")
            # Reopen the file to read from beginning
            f.seek(0)
            reader = csv.reader(f)
            next(reader, None)  # Skip header row
            for row in reader:
                if len(row) >= 3:
                    company_name = row[2].strip() if len(row[2].strip()) > 0 else "Unknown"
                    companies_file2.append(company_name)
        print(f"Successfully read {len(companies_file2)} companies from file 2")
        break
    except Exception as e:
        print(f"Error reading second file with {encoding}: {e}")
        companies_file2 = []

if not companies_file2:
    print("Failed to read the second file with any encoding")
    exit(1)

# Print out sample of company names to verify
print("\nSample companies from file 1:")
for name in companies_file1[:5]:
    print(f"  - {name}")

print("\nSample companies from file 2:")
for name in companies_file2[:5]:
    print(f"  - {name}")

# Clean up company names
def clean_name(name):
    # Remove common irrelevant terms and normalize
    name = name.replace('有限公司', '').replace('有限责任公司', '')
    name = name.replace('股份', '').replace('责任', '')
    name = name.replace('(', '').replace(')', '')
    name = name.replace('（', '').replace('）', '')
    name = re.sub(r'\s+', '', name)
    return name

# Clean all company names
clean_companies1 = [clean_name(name) for name in companies_file1]
clean_companies2 = [clean_name(name) for name in companies_file2]

# Find exact matches
exact_matches = []
for i, name1 in enumerate(companies_file1):
    for j, name2 in enumerate(companies_file2):
        if name1 == name2:
            exact_matches.append((name1, name2))

# Find high similarity matches
threshold = 0.7  # Adjust as needed
potential_matches = []
for i, name1 in enumerate(companies_file1):
    for j, name2 in enumerate(companies_file2):
        if name1 != name2:  # Skip exact matches
            clean1 = clean_companies1[i]
            clean2 = clean_companies2[j]
            if clean1 and clean2:  # Make sure neither is empty
                sim_ratio = similarity(clean1, clean2)
                if sim_ratio > threshold:
                    potential_matches.append((name1, name2, sim_ratio))

# Sort potential matches by similarity
potential_matches.sort(key=lambda x: x[2], reverse=True)

# Find companies in file1 not matching any in file2
not_matched_file1 = []
for name1 in companies_file1:
    matched = False
    for match in exact_matches:
        if name1 == match[0]:
            matched = True
            break
    
    if not matched:
        for match in potential_matches:
            if name1 == match[0] and match[2] > 0.8:  # High confidence match
                matched = True
                break
    
    if not matched:
        not_matched_file1.append(name1)

# Find companies in file2 not matching any in file1
not_matched_file2 = []
for name2 in companies_file2:
    matched = False
    for match in exact_matches:
        if name2 == match[1]:
            matched = True
            break
    
    if not matched:
        for match in potential_matches:
            if name2 == match[1] and match[2] > 0.8:  # High confidence match
                matched = True
                break
    
    if not matched:
        not_matched_file2.append(name2)

# Output results
print(f"\nTotal companies in file 1: {len(companies_file1)}")
print(f"Total companies in file 2: {len(companies_file2)}")
print(f"Exact matches found: {len(exact_matches)}")
print(f"Potential matches found: {len(potential_matches)}")
print(f"Companies in file 1 not matched: {len(not_matched_file1)}")
print(f"Companies in file 2 not matched: {len(not_matched_file2)}")

# Write results to file
with open('comparison_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== 精确匹配的公司 ===\n")
    for i, (name1, name2) in enumerate(exact_matches):
        f.write(f"{i+1}. 文件1: {name1} | 文件2: {name2}\n")
    
    f.write("\n=== 可能匹配的公司 (相似度排序) ===\n")
    for i, (name1, name2, ratio) in enumerate(potential_matches[:100]):  # Limit to 100 most likely
        f.write(f"{i+1}. 相似度: {ratio:.2f} | 文件1: {name1} | 文件2: {name2}\n")
    
    f.write("\n=== 文件1中未匹配的公司 (前100个) ===\n")
    for i, name in enumerate(not_matched_file1[:100]):
        f.write(f"{i+1}. {name}\n")
    
    f.write("\n=== 文件2中未匹配的公司 (前100个) ===\n")
    for i, name in enumerate(not_matched_file2[:100]):
        f.write(f"{i+1}. {name}\n")

print("Results written to comparison_results.txt") 