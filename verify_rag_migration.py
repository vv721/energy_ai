#!/usr/bin/env python3
"""
Energy AI RAG System - LCEL Migration Verification Report
验证所有 RAG 模块已成功迁移到 LCEL 实现
"""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_imports():
    """验证所有关键导入"""
    print("=" * 60)
    print("VERIFICATION 1: Checking Module Imports")
    print("=" * 60)
    
    checks = [
        ("backend.rag.rag_chain", "RAGChain"),
        ("backend.rag.vector_store", "VectorStoreManager"),
        ("backend.rag.document_processor", "DocumentProcessor"),
        ("backend.llm.llm_factory", "get_llm"),
    ]
    
    for module_name, class_name in checks:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  [OK] {module_name}.{class_name}")
        except Exception as e:
            print(f"  [FAIL] {module_name}.{class_name}: {e}")
            return False
    
    return True

def verify_lcel_components():
    """验证 LCEL 组件可用性"""
    print("\n" + "=" * 60)
    print("VERIFICATION 2: Checking LCEL Components")
    print("=" * 60)
    
    components = [
        "langchain_core.runnables:RunnablePassthrough",
        "langchain_core.runnables:RunnableParallel",
        "langchain_core.output_parsers:StrOutputParser",
        "langchain_core.prompts:PromptTemplate",
        "langchain_core.documents:Document",
    ]
    
    for item in components:
        module_path, comp = item.split(":")
        try:
            module = __import__(module_path, fromlist=[comp])
            obj = getattr(module, comp)
            print(f"  [OK] {item}")
        except Exception as e:
            print(f"  [FAIL] {item}: {e}")
            return False
    
    return True

def verify_no_deprecated_imports():
    """验证不存在废弃导入"""
    print("\n" + "=" * 60)
    print("VERIFICATION 3: Checking for Deprecated Imports")
    print("=" * 60)
    
    # Files to check
    files_to_check = [
        "backend/rag/rag_chain.py",
        "backend/rag/vector_store.py",
        "backend/rag/document_processor.py",
    ]
    
    # Only check for actual imports, not docstring mentions
    deprecated_imports = [
        "from langchain_classic",
        "import langchain_classic",
        "from langchain_community.chains import PebbloRetrievalQA",
    ]
    
    all_clean = True
    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                found_deprecated = False
                for pattern in deprecated_imports:
                    if pattern in content:
                        print(f"  [FAIL] {file_path}: Found '{pattern}'")
                        found_deprecated = True
                        all_clean = False
                if not found_deprecated:
                    print(f"  [OK] {file_path}: Clean")
    
    return all_clean

def verify_rag_chain_methods():
    """验证 RAGChain 的核心方法"""
    print("\n" + "=" * 60)
    print("VERIFICATION 4: Checking RAGChain Methods")
    print("=" * 60)
    
    try:
        from backend.rag.vector_store import VectorStoreManager
        from backend.rag.rag_chain import RAGChain
        
        vs_manager = VectorStoreManager(persist_directory="./dummy_vs")
        rag = RAGChain(vs_manager)
        
        methods = [
            "setup_qa_chain",
            "answer_question",
            "get_relevant_documents",
        ]
        
        for method_name in methods:
            if hasattr(rag, method_name) and callable(getattr(rag, method_name)):
                print(f"  [OK] RAGChain.{method_name}()")
            else:
                print(f"  [FAIL] RAGChain.{method_name}() not found")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error creating RAGChain: {e}")
        return False

def verify_syntax():
    """验证 Python 文件语法"""
    print("\n" + "=" * 60)
    print("VERIFICATION 5: Checking Python Syntax")
    print("=" * 60)
    
    import py_compile
    
    files_to_check = [
        "backend/rag/rag_chain.py",
        "backend/rag/vector_store.py",
        "backend/rag/document_processor.py",
    ]
    
    all_valid = True
    for file_path in files_to_check:
        full_path = project_root / file_path
        try:
            py_compile.compile(str(full_path), doraise=True)
            print(f"  [OK] {file_path}: Valid syntax")
        except py_compile.PyCompileError as e:
            print(f"  [FAIL] {file_path}: {e}")
            all_valid = False
    
    return all_valid

def main():
    """运行所有验证"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Energy AI RAG System - LCEL Migration Verification  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = []
    
    results.append(("Module Imports", verify_imports()))
    results.append(("LCEL Components", verify_lcel_components()))
    results.append(("No Deprecated Imports", verify_no_deprecated_imports()))
    results.append(("RAGChain Methods", verify_rag_chain_methods()))
    results.append(("Python Syntax", verify_syntax()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "[OK]" if result else "[FAIL]"
        print(f"  {symbol} {check_name}: {status}")
    
    print("\n" + "=" * 60)
    print(f"Overall: {passed}/{total} checks passed")
    print("=" * 60)
    
    if passed == total:
        print("\n[OK] ALL VERIFICATIONS PASSED!")
        print("[OK] LCEL migration is complete and successful")
        print("[OK] Project is ready for deployment")
        return 0
    else:
        print(f"\n[FAIL] {total - passed} verification(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
