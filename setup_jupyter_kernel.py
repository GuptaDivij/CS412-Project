"""
配置 Jupyter Notebook 使用与命令行相同的 Python 环境
"""

import sys
import subprocess
import os

def main():
    print("="*60)
    print("配置 Jupyter Notebook 使用当前 Python 环境")
    print("="*60)
    
    # 获取当前 Python 路径
    python_path = sys.executable
    python_version = sys.version
    
    print(f"\n当前 Python 环境:")
    print(f"  路径: {python_path}")
    print(f"  版本: {python_version.split()[0]}")
    
    # 检查是否安装了 ipykernel
    try:
        import ipykernel
        print(f"\n✓ ipykernel 已安装: {ipykernel.__version__}")
    except ImportError:
        print("\n✗ ipykernel 未安装，正在安装...")
        try:
            subprocess.check_call([python_path, "-m", "pip", "install", "ipykernel", "-q"])
            print("✓ ipykernel 安装成功")
        except Exception as e:
            print(f"✗ 安装失败: {e}")
            print("\n请手动运行: python -m pip install ipykernel")
            return
    
    # 获取 kernel 名称
    kernel_name = "python3"
    
    # 检查是否已经注册了 kernel
    try:
        result = subprocess.run(
            [python_path, "-m", "ipykernel", "list"],
            capture_output=True,
            text=True
        )
        
        if python_path in result.stdout:
            print(f"\n✓ Kernel '{kernel_name}' 已经注册到当前 Python 环境")
        else:
            print(f"\n正在注册 kernel '{kernel_name}'...")
            subprocess.check_call([
                python_path, "-m", "ipykernel", "install", 
                "--user", 
                "--name", kernel_name,
                "--display-name", f"Python {sys.version_info.major}.{sys.version_info.minor} ({python_path})"
            ])
            print(f"✓ Kernel '{kernel_name}' 注册成功")
    except Exception as e:
        print(f"\n✗ 注册 kernel 失败: {e}")
        print("\n请手动运行:")
        print(f'  python -m ipykernel install --user --name python3 --display-name "Python {sys.version_info.major}.{sys.version_info.minor}"')
        return
    
    print("\n" + "="*60)
    print("配置完成！")
    print("="*60)
    print("\n使用说明:")
    print("1. 打开 Jupyter Notebook")
    print("2. 创建新 notebook 时，选择 kernel: 'Python 3'")
    print("3. 或者在现有 notebook 中: Kernel -> Change Kernel -> 选择 'Python 3'")
    print("\n验证方法（在 Jupyter Notebook 中运行）:")
    print("  import sys")
    print("  print(sys.executable)")
    print("  应该显示:", python_path)

if __name__ == "__main__":
    main()





