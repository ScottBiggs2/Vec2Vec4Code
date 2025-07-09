#!/usr/bin/env python3
"""
Simple Code Collector for Vec2Vec Training Dataset

This script collects Python and C code snippets from publicly available sources
without requiring API keys or complex authentication.
"""

import requests
import time
import re
import os
from typing import List, Dict
from bs4 import BeautifulSoup
import tempfile
import subprocess

class SimpleCodeCollector:
    def __init__(self, output_dir: str = "scraped_code"):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Rate limiting
        self.request_delay = 2.0  # seconds between requests
        
    def clean_code_snippet(self, code: str, language: str) -> str:
        """Clean and validate code snippet"""
        # Remove common artifacts
        code = code.strip()
        
        # Remove code block markers
        code = re.sub(r'^```.*?\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'``` ```', '', code, flags=re.MULTILINE)
        
        # Remove HTML entities
        code = code.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        
        # Remove line numbers
        code = re.sub(r'^\d+\s*', '', code, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
        
        # Language-specific cleaning
        if language == 'python':
            # Remove >>> prompts
            code = re.sub(r'^>>>\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'^\.\.\.\s*', '', code, flags=re.MULTILINE)
            
        elif language == 'c':
            # Ensure basic includes if using standard functions
            if 'printf' in code and '#include' not in code:
                code = '#include <stdio.h>\n' + code
            if any(func in code for func in ['strlen', 'strcpy', 'strcmp']) and '#include <string.h>' not in code:
                code = '#include <string.h>\n' + code
        
        return code.strip()
    
    def validate_code_snippet(self, code: str, language: str) -> bool:
        """Validate that code snippet is reasonable"""
        
        # Basic checks
        if len(code) < 30 or len(code) > 1500:
            return False
        
        if code.count('\n') < 3:
            return False
        
        # Language-specific validation
        if language == 'python':
            # Should have Python patterns
            python_patterns = [
                r'def\s+\w+\s*\(',
                r'class\s+\w+',
                r'for\s+\w+\s+in',
                r'if\s+.+:',
                r'while\s+.+:',
                r'import\s+\w+',
                r'from\s+\w+\s+import'
            ]
            
            if not any(re.search(pattern, code) for pattern in python_patterns):
                return False
            
            # Try basic syntax check
            try:
                compile(code, '<string>', 'exec')
                return True
            except SyntaxError:
                return False
                
        elif language == 'c':
            # Should have C patterns
            c_patterns = [
                r'\w+\s+\w+\s*\([^)]*\)\s*{',
                r'#include\s*<.+>',
                r'int\s+main\s*\(',
                r'printf\s*\(',
                r'for\s*\(',
                r'while\s*\(',
                r'if\s*\('
            ]
            
            if not any(re.search(pattern, code) for pattern in c_patterns):
                return False
            
            return True
        
        return False
    
    def scrape_rosetta_code(self, language: str, max_snippets: int = 50) -> List[str]:
        """Scrape code examples from Rosetta Code"""
        snippets = []
        
        # Rosetta Code has great algorithm examples
        tasks = [
            'Bubble_sort', 'Quick_sort', 'Binary_search', 'Fibonacci_sequence',
            'Factorial', 'Prime_numbers', 'Greatest_common_divisor',
            'Palindrome_detection', 'String_reverse', 'Array_sum',
            'Linear_search', 'Selection_sort', 'Insertion_sort',
            'Matrix_multiplication', 'Towers_of_Hanoi', 'Stack',
            'Queue', 'Hash_table', 'Tree_traversal', 'Graph_algorithms'
        ]
        
        lang_name = 'Python' if language == 'python' else 'C'
        
        for task in tasks:
            if len(snippets) >= max_snippets:
                break
                
            try:
                url = f"https://rosettacode.org/wiki/{task}#{lang_name}"
                print(f"Scraping Rosetta Code: {task}")
                
                response = self.session.get(url)
                time.sleep(self.request_delay)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find code blocks
                    code_elements = soup.find_all(['pre', 'code'])
                    
                    for element in code_elements:
                        if len(snippets) >= max_snippets:
                            break
                        
                        code = element.get_text()
                        cleaned = self.clean_code_snippet(code, language)
                        
                        if self.validate_code_snippet(cleaned, language):
                            snippets.append(cleaned)
                            print(f"  Found valid {language} snippet ({len(cleaned)} chars)")
                            
            except Exception as e:
                print(f"Error scraping {task}: {e}")
                continue
        
        return snippets
    
    def scrape_geeks_for_geeks(self, language: str, max_snippets: int = 30) -> List[str]:
        """Scrape code examples from GeeksforGeeks"""
        snippets = []
        
        if language == 'python':
            topics = [
                'python-programming-examples',
                'python-data-structures',
                'python-algorithms',
                'python-string-programs',
                'python-array-programs'
            ]
        else:
            topics = [
                'c-programming-examples',
                'c-data-structures',
                'c-algorithms',
                'c-string-programs',
                'c-array-programs'
            ]
        
        for topic in topics:
            if len(snippets) >= max_snippets:
                break
                
            try:
                url = f"https://www.geeksforgeeks.org/{topic}/"
                print(f"Scraping GeeksforGeeks: {topic}")
                
                response = self.session.get(url)
                time.sleep(self.request_delay)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find code blocks
                    code_elements = soup.find_all(['pre', 'code'])
                    
                    for element in code_elements:
                        if len(snippets) >= max_snippets:
                            break
                        
                        code = element.get_text()
                        cleaned = self.clean_code_snippet(code, language)
                        
                        if self.validate_code_snippet(cleaned, language):
                            snippets.append(cleaned)
                            print(f"  Found valid {language} snippet ({len(cleaned)} chars)")
                            
            except Exception as e:
                print(f"Error scraping {topic}: {e}")
                continue
        
        return snippets
    
    def scrape_programiz(self, language: str, max_snippets: int = 30) -> List[str]:
        """Scrape code examples from Programiz"""
        snippets = []
        
        if language == 'python':
            base_url = "https://www.programiz.com/python-programming/examples"
        else:
            base_url = "https://www.programiz.com/c-programming/examples"
        
        try:
            print(f"Scraping Programiz for {language}")
            
            response = self.session.get(base_url)
            time.sleep(self.request_delay)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find links to individual examples
                example_links = soup.find_all('a', href=True)
                
                for link in example_links:
                    if len(snippets) >= max_snippets:
                        break
                    
                    href = link['href']
                    if '/examples/' in href:
                        try:
                            if not href.startswith('http'):
                                href = f"https://www.programiz.com{href}"
                            
                            example_response = self.session.get(href)
                            time.sleep(self.request_delay)
                            
                            if example_response.status_code == 200:
                                example_soup = BeautifulSoup(example_response.text, 'html.parser')
                                
                                # Find code blocks
                                code_elements = example_soup.find_all(['pre', 'code'])
                                
                                for element in code_elements:
                                    code = element.get_text()
                                    cleaned = self.clean_code_snippet(code, language)
                                    
                                    if self.validate_code_snippet(cleaned, language):
                                        snippets.append(cleaned)
                                        print(f"  Found valid {language} snippet from {href}")
                                        break
                                        
                        except Exception as e:
                            print(f"Error scraping example {href}: {e}")
                            continue
                            
        except Exception as e:
            print(f"Error scraping Programiz: {e}")
        
        return snippets
    
    def generate_algorithmic_variants(self, language: str, count: int = 20) -> List[str]:
        """Generate variations of common algorithms"""
        snippets = []
        
        if language == 'python':
            templates = [
                # Sorting variations
                "def bubble_sort_variant(arr, ascending=True):\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n-i-1):\n            condition = arr[j] > arr[j+1] if ascending else arr[j] < arr[j+1]\n            if condition:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n                swapped = True\n        if not swapped:\n            break\n    return arr",
                
                "def selection_sort_variant(arr, key=None):\n    n = len(arr)\n    for i in range(n):\n        min_idx = i\n        for j in range(i+1, n):\n            val_j = arr[j] if key is None else key(arr[j])\n            val_min = arr[min_idx] if key is None else key(arr[min_idx])\n            if val_j < val_min:\n                min_idx = j\n        arr[i], arr[min_idx] = arr[min_idx], arr[i]\n    return arr",
                
                # Search variations
                "def binary_search_variant(arr, target, left=0, right=None):\n    if right is None:\n        right = len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                
                # Data structure variations
                "class SimpleQueue:\n    def __init__(self, maxsize=None):\n        self.items = []\n        self.maxsize = maxsize\n    \n    def enqueue(self, item):\n        if self.maxsize and len(self.items) >= self.maxsize:\n            raise Exception('Queue is full')\n        self.items.append(item)\n    \n    def dequeue(self):\n        if not self.items:\n            raise Exception('Queue is empty')\n        return self.items.pop(0)\n    \n    def is_empty(self):\n        return len(self.items) == 0",
                
                # String processing variations
                "def string_utils(text, operation='reverse'):\n    if operation == 'reverse':\n        return text[::-1]\n    elif operation == 'uppercase':\n        return text.upper()\n    elif operation == 'word_count':\n        return len(text.split())\n    elif operation == 'char_frequency':\n        freq = {}\n        for char in text:\n            freq[char] = freq.get(char, 0) + 1\n        return freq\n    return text",
            ]
        else:  # C
            templates = [
                # Sorting variations
                "void bubble_sort_variant(int arr[], int n, int ascending) {\n    for (int i = 0; i < n-1; i++) {\n        int swapped = 0;\n        for (int j = 0; j < n-i-1; j++) {\n            int condition = ascending ? (arr[j] > arr[j+1]) : (arr[j] < arr[j+1]);\n            if (condition) {\n                int temp = arr[j];\n                arr[j] = arr[j+1];\n                arr[j+1] = temp;\n                swapped = 1;\n            }\n        }\n        if (!swapped) break;\n    }\n}",
                
                "void selection_sort_variant(int arr[], int n) {\n    for (int i = 0; i < n-1; i++) {\n        int min_idx = i;\n        for (int j = i+1; j < n; j++) {\n            if (arr[j] < arr[min_idx]) {\n                min_idx = j;\n            }\n        }\n        int temp = arr[min_idx];\n        arr[min_idx] = arr[i];\n        arr[i] = temp;\n    }\n}",
                
                # Search variations
                "int binary_search_variant(int arr[], int n, int target) {\n    int left = 0, right = n - 1;\n    while (left <= right) {\n        int mid = left + (right - left) / 2;\n        if (arr[mid] == target) {\n            return mid;\n        }\n        if (arr[mid] < target) {\n            left = mid + 1;\n        } else {\n            right = mid - 1;\n        }\n    }\n    return -1;\n}",
                
                # Data structure variations
                "#include <stdio.h>\n#include <stdlib.h>\n\ntypedef struct {\n    int* data;\n    int front;\n    int rear;\n    int capacity;\n} Queue;\n\nQueue* create_queue(int capacity) {\n    Queue* q = (Queue*)malloc(sizeof(Queue));\n    q->data = (int*)malloc(capacity * sizeof(int));\n    q->front = 0;\n    q->rear = -1;\n    q->capacity = capacity;\n    return q;\n}\n\nvoid enqueue(Queue* q, int item) {\n    if (q->rear < q->capacity - 1) {\n        q->data[++q->rear] = item;\n    }\n}\n\nint dequeue(Queue* q) {\n    if (q->front <= q->rear) {\n        return q->data[q->front++];\n    }\n    return -1;\n}",
                
                # String processing variations
                "#include <string.h>\n\nvoid string_reverse(char* str) {\n    int len = strlen(str);\n    for (int i = 0; i < len/2; i++) {\n        char temp = str[i];\n        str[i] = str[len-1-i];\n        str[len-1-i] = temp;\n    }\n}\n\nint string_length(char* str) {\n    int count = 0;\n    while (str[count] != '\\0') {\n        count++;\n    }\n    return count;\n}\n\nvoid string_copy(char* dest, char* src) {\n    int i = 0;\n    while (src[i] != '\\0') {\n        dest[i] = src[i];\n        i++;\n    }\n    dest[i] = '\\0';\n}"
            ]
        
        # Add all templates as they are already valid code
        for template in templates:
            if self.validate_code_snippet(template, language):
                snippets.append(template)
        
        return snippets
    
    def save_snippets(self, snippets: List[str], language: str, source: str):
        """Save collected code snippets to files"""
        filename = f"{self.output_dir}/{source}_{language}_snippets.py"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {language.upper()} code snippets from {source}\n")
            f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"{source.upper()}_{language.upper()}_SNIPPETS = [\n")
            for snippet in snippets:
                # Escape quotes and newlines for Python list
                escaped = snippet.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                f.write(f'    "{escaped}",\n')
            f.write("]\n")
        
        print(f"✅ Saved {len(snippets)} {language} snippets from {source} to {filename}")
    
    def collect_all_snippets(self):
        """Collect code snippets from all sources"""
        
        for language in ['python', 'c']:
            print(f"\n🔍 === Collecting {language.upper()} code snippets ===")
            
            all_snippets = []
            
            # Rosetta Code
            print(f"📚 Scraping Rosetta Code for {language} examples...")
            rosetta_snippets = self.scrape_rosetta_code(language, 25)
            all_snippets.extend(rosetta_snippets)
            if rosetta_snippets:
                self.save_snippets(rosetta_snippets, language, 'rosetta_code')
            
            # GeeksforGeeks
            print(f"📚 Scraping GeeksforGeeks for {language} examples...")
            gfg_snippets = self.scrape_geeks_for_geeks(language, 15)
            all_snippets.extend(gfg_snippets)
            if gfg_snippets:
                self.save_snippets(gfg_snippets, language, 'geeksforgeeks')
            
            # Programiz
            print(f"📚 Scraping Programiz for {language} examples...")
            programiz_snippets = self.scrape_programiz(language, 15)
            all_snippets.extend(programiz_snippets)
            if programiz_snippets:
                self.save_snippets(programiz_snippets, language, 'programiz')
            
            # Generate algorithmic variants
            print(f"🔧 Generating algorithmic variants for {language}...")
            variant_snippets = self.generate_algorithmic_variants(language, 10)
            all_snippets.extend(variant_snippets)
            if variant_snippets:
                self.save_snippets(variant_snippets, language, 'variants')
            
            # Remove duplicates and save combined
            unique_snippets = []
            seen = set()
            for snippet in all_snippets:
                if snippet not in seen:
                    seen.add(snippet)
                    unique_snippets.append(snippet)
            
            if unique_snippets:
                self.save_snippets(unique_snippets, language, 'combined')
            
            print(f"✅ Collected {len(unique_snippets)} unique {language} snippets")
    
    def create_final_dataset(self, output_file: str = "large_training_dataset.py"):
        """Create final training dataset file"""
        
        # Load all collected snippets
        all_python_snippets = []
        all_c_snippets = []
        
        # Look for combined files first
        python_combined = f"{self.output_dir}/combined_python_snippets.py"
        c_combined = f"{self.output_dir}/combined_c_snippets.py"
        
        if os.path.exists(python_combined):
            try:
                with open(python_combined, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract the list content
                    start = content.find('COMBINED_PYTHON_SNIPPETS = [')
                    if start != -1:
                        start = content.find('[', start) + 1
                        end = content.rfind(']')
                        if end != -1:
                            list_content = content[start:end]
                            # Parse the snippets (simple approach)
                            import ast
                            snippets_str = '[' + list_content + ']'
                            try:
                                all_python_snippets = ast.literal_eval(snippets_str)
                            except:
                                print("Error parsing Python snippets")
            except Exception as e:
                print(f"Error reading Python combined file: {e}")
        
        if os.path.exists(c_combined):
            try:
                with open(c_combined, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract the list content
                    start = content.find('COMBINED_C_SNIPPETS = [')
                    if start != -1:
                        start = content.find('[', start) + 1
                        end = content.rfind(']')
                        if end != -1:
                            list_content = content[start:end]
                            # Parse the snippets (simple approach)
                            import ast
                            snippets_str = '[' + list_content + ']'
                            try:
                                all_c_snippets = ast.literal_eval(snippets_str)
                            except:
                                print("Error parsing C snippets")
            except Exception as e:
                print(f"Error reading C combined file: {e}")
        
        # Create final dataset file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Large-scale training dataset for Vec2Vec code translation\n")
            f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Contains {len(all_python_snippets)} Python and {len(all_c_snippets)} C snippets\n\n")
            
            f.write("LARGE_PYTHON_SAMPLES = [\n")
            for snippet in all_python_snippets:
                escaped = snippet.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                f.write(f'    "{escaped}",\n')
            f.write("]\n\n")
            
            f.write("LARGE_C_SAMPLES = [\n")
            for snippet in all_c_snippets:
                escaped = snippet.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                f.write(f'    "{escaped}",\n')
            f.write("]\n")
        
        print(f"✅ Created final dataset: {output_file}")
        print(f"📊 Python snippets: {len(all_python_snippets)}")
        print(f"📊 C snippets: {len(all_c_snippets)}")
        print(f"📊 Total: {len(all_python_snippets) + len(all_c_snippets)} code snippets")

def main():
    """Main function to run the code collector"""
    
    collector = SimpleCodeCollector()
    
    print("🚀 Starting code collection for Vec2Vec training dataset...")
    print("⏱️  This may take a while due to rate limiting...")
    
    # Collect snippets from all sources
    collector.collect_all_snippets()
    
    # Create final training dataset
    collector.create_final_dataset()
    
    print("\n🎉 Code collection complete!")
    print("📁 Check the 'scraped_code' directory for individual source files")
    print("📄 Use 'large_training_dataset.py' for your Vec2Vec training")

if __name__ == "__main__":
    main()