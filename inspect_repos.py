"""
Repository Inspector - Helps identify how to integrate each embedding method
Analyzes the cloned repositories to find main functions and APIs
"""

import os
from pathlib import Path
import re


def find_python_files(directory: Path, max_depth: int = 3) -> list:
    """Find all Python files in directory up to max_depth"""
    python_files = []
    
    def search(path, depth):
        if depth > max_depth:
            return
        try:
            for item in path.iterdir():
                if item.is_file() and item.suffix == '.py':
                    python_files.append(item)
                elif item.is_dir() and not item.name.startswith('.'):
                    search(item, depth + 1)
        except PermissionError:
            pass
    
    search(directory, 0)
    return python_files


def find_embedding_functions(file_path: Path) -> list:
    """Search for likely embedding function definitions"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Patterns that might indicate embedding functions
        patterns = [
            r'def\s+(.*embed.*)\s*\(',
            r'def\s+(.*minor.*)\s*\(',
            r'class\s+(.*Embed.*)\s*[\(:]',
            r'class\s+(.*Minor.*)\s*[\(:]',
        ]
        
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, content, re.IGNORECASE)
            matches.extend(found)
        
        return list(set(matches))  # Remove duplicates
        
    except Exception as e:
        return []


def find_main_files(directory: Path) -> list:
    """Find likely main entry point files"""
    main_files = []
    
    # Common names for main files
    main_names = ['main.py', '__main__.py', 'embed.py', 'embedding.py', 
                  'run.py', 'demo.py', 'example.py']
    
    for py_file in directory.rglob('*.py'):
        if py_file.name in main_names:
            main_files.append(py_file)
    
    return main_files


def analyze_repository(repo_path: Path, repo_name: str):
    """Analyze a repository to help with integration"""
    print(f"\n{'=' * 80}")
    print(f"Analyzing: {repo_name}")
    print(f"Path: {repo_path}")
    print('=' * 80)
    
    if not repo_path.exists():
        print(f"❌ Repository not found at {repo_path}")
        print(f"   Run setup_benchmark.py first to clone the repository")
        return
    
    # Find README
    readme_files = list(repo_path.glob('README*'))
    if readme_files:
        print(f"\n📖 README found: {readme_files[0].name}")
        print(f"   Read this file first for API documentation")
    else:
        print(f"\n⚠️  No README found")
    
    # Find requirements
    req_files = list(repo_path.glob('requirements*.txt'))
    if req_files:
        print(f"\n📦 Requirements file: {req_files[0].name}")
        try:
            with open(req_files[0], 'r') as f:
                deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"   Dependencies: {', '.join(deps[:5])}")
            if len(deps) > 5:
                print(f"   ... and {len(deps) - 5} more")
        except:
            pass
    
    # Find setup.py
    setup_file = repo_path / 'setup.py'
    if setup_file.exists():
        print(f"\n⚙️  setup.py found - repository can be installed as package")
        print(f"   Try: pip install -e {repo_path}")
    
    # Find main/entry point files
    print(f"\n🔍 Searching for main entry points...")
    main_files = find_main_files(repo_path)
    if main_files:
        print(f"   Found {len(main_files)} potential entry points:")
        for mf in main_files[:5]:
            print(f"     - {mf.relative_to(repo_path)}")
    else:
        print(f"   No obvious entry points found")
    
    # Find Python files with embedding functions
    print(f"\n🎯 Searching for embedding-related functions...")
    python_files = find_python_files(repo_path)
    print(f"   Scanning {len(python_files)} Python files...")
    
    function_findings = {}
    for py_file in python_files:
        functions = find_embedding_functions(py_file)
        if functions:
            rel_path = py_file.relative_to(repo_path)
            function_findings[rel_path] = functions
    
    if function_findings:
        print(f"\n   Found embedding functions in {len(function_findings)} files:")
        for file_path, functions in list(function_findings.items())[:10]:
            print(f"\n   📄 {file_path}")
            for func in functions[:3]:
                print(f"      → {func}()")
    else:
        print(f"   No embedding functions found automatically")
        print(f"   You'll need to manually inspect the repository")
    
    # Look for examples
    example_dirs = [repo_path / 'examples', repo_path / 'demos', 
                    repo_path / 'tests', repo_path / 'tutorial']
    example_files = []
    
    for ex_dir in example_dirs:
        if ex_dir.exists():
            example_files.extend(list(ex_dir.glob('*.py')))
    
    if example_files:
        print(f"\n💡 Example files found:")
        for ex_file in example_files[:5]:
            print(f"   - {ex_file.relative_to(repo_path)}")
        print(f"   Check these for usage examples!")
    
    # Integration suggestions
    print(f"\n" + "─" * 80)
    print(f"📝 Integration Checklist for {repo_name}:")
    print(f"─" * 80)
    print(f"1. Read the README: {readme_files[0] if readme_files else 'N/A'}")
    print(f"2. Check examples directory for usage patterns")
    print(f"3. Look for main embedding function (likely in files above)")
    print(f"4. Understand input format (NetworkX graph? Adjacency dict?)")
    print(f"5. Understand output format (dict? custom object?)")
    print(f"6. Test the method independently before integrating")
    print(f"7. Write integration code in integration_template.py")
    

def main():
    """Main inspector routine"""
    print("=" * 80)
    print("Repository Inspector for Minor Embedding Methods")
    print("=" * 80)
    print("\nThis tool analyzes the cloned repositories to help you integrate them")
    print("into the benchmark framework.\n")
    
    repos = [
        ("implementations/atom", "ATOM"),
        ("implementations/charme", "CHARME"),
        ("implementations/oct_based", "OCT-Based"),
    ]
    
    for repo_path_str, repo_name in repos:
        repo_path = Path(repo_path_str)
        analyze_repository(repo_path, repo_name)
    
    print(f"\n" + "=" * 80)
    print("Inspection Complete!")
    print("=" * 80)
    print("""
Next Steps:
1. Review the analysis above for each repository
2. Read the README and example files for each method
3. Identify the main embedding function and its API
4. Edit integration_template.py with the correct function calls
5. Test each integration individually before running full benchmark

Pro tip: Start with the method that has the clearest documentation!
    """)


if __name__ == "__main__":
    main()
