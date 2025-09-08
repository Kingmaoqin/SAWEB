# # texgisa.py
# Legacy shim: keep old entrypoint name but route to new MySA implementation.
from .mysa import run_mysa as run_texgisa

# import pandas as pd
# import subprocess
# import json
# import os

# def run_texgisa(data: pd.DataFrame, config: dict) -> dict:
#     """
#     Wrapper function to run the TEXGISA model via best.py script.
#     """
#     # --- 1. 准备输入数据 ---
#     # 将用户上传的内存中的数据保存到一个临时的CSV文件中
#     temp_input_path = 'temp_input_for_best.csv'
#     data.to_csv(temp_input_path, index=False)
    
#     # --- 2. 定义脚本路径和输出路径 ---
#     # 这是您提供的best.py的绝对路径
#     script_path = '/home/xqin5/SAonline/ICDM-SA/FLCHAIN/best.py'
#     # 假设您的脚本会生成一个名为 "results.json" 的结果文件
#     # **您可能需要根据 best.py 的实际行为修改这里**
#     temp_output_path = 'results.json'

#     # --- 3. 执行您的脚本 ---
#     try:
#         print("Executing best.py script...")
#         # 使用subprocess来运行 "python /path/to/best.py"
#         # 我们设置工作目录，以防脚本需要读取像flchain.csv这样的相对路径文件
#         working_directory = os.path.dirname(script_path)
#         subprocess.run(
#             ['python', script_path],
#             check=True,
#             capture_output=True,
#             text=True,
#             cwd=working_directory # 设置工作目录
#         )
#         print("best.py script finished successfully.")

#         # --- 4. 读取并解析结果 ---
#         # **重要**: 请确保您的 best.py 脚本会生成一个包含结果的 JSON 文件
#         # 如果您的脚本输出结果的方式不同，需要修改这部分代码
#         if os.path.exists(temp_output_path):
#             with open(temp_output_path, 'r') as f:
#                 results = json.load(f)
#             print(f"Loaded results: {results}")
#         else:
#             # 如果脚本不生成文件，而是打印结果，我们可以捕获它
#             # 在这种情况下，您需要修改best.py让它打印JSON格式的结果
#             results = {"error": f"Result file '{temp_output_path}' not found. Please check the best.py script."}

#     except subprocess.CalledProcessError as e:
#         print(f"Error executing best.py: {e}")
#         print(f"Stderr: {e.stderr}")
#         return {"error": f"Failed to run best.py. Details: {e.stderr}"}
#     except Exception as e:
#         return {"error": f"An unexpected error occurred: {str(e)}"}
#     finally:
#         # --- 5. 清理临时文件 ---
#         if os.path.exists(temp_input_path):
#             os.remove(temp_input_path)
#         if os.path.exists(temp_output_path):
#             os.remove(temp_output_path)

#     return results