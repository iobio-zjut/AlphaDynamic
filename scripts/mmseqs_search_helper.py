import argparse
import pandas as pd
import subprocess
import os
import shutil  # 引入 shutil 用于更高级的文件操作，如移动目录

# --- 解析命令行参数 ---
parser = argparse.ArgumentParser(description='批量读取多蛋白质序列并一次性运行mmseqs搜索，然后根据序列名称拆分结果。')
parser.add_argument('--base_dir', type=str, required=True,
                    help='包含所有蛋白质子文件夹的基础输入目录。')
parser.add_argument('--protein_list_file', type=str, required=True,
                    help='包含需要处理的蛋白质名称（每行一个）的txt文件路径。')
parser.add_argument('--db_dir', type=str, default='./dbbase',
                    help='MMseqs2数据库的路径。')
args = parser.parse_args()

# --- 配置临时文件和目录 ---
# 使用固定的临时路径，因为现在只运行一次搜索
TOTAL_TMP_FASTA = '/tmp/total_merged_queries.fasta'
TOTAL_TMP_OUTDIR = '/tmp/total_mmseqs_search_results'


def run_batch_mmseqs_search(protein_list, base_dir, db_dir):
    """
    批量处理所有蛋白质的序列，运行一次搜索，然后拆分结果。
    """
    all_data = []
    final_outdirs = {}  # 存储最终的输出目录 {protein_name: path}

    # ----------------------------------------------------
    # 步骤 1 & 2: 合并输入数据并创建总 FASTA 文件
    # ----------------------------------------------------
    print("\n--- 1/4: 正在合并输入序列... ---")

    for protein_name in protein_list:
        input_file = os.path.join(base_dir, protein_name, 'msas_cluster', 'msas_cluster.csv')
        outdir = os.path.join(base_dir, protein_name, 'msas_cluster_search')

        # 存储最终输出目录，并确保其存在
        final_outdirs[protein_name] = outdir
        os.makedirs(outdir, exist_ok=True)

        if not os.path.exists(input_file):
            print(f"警告：找不到 {protein_name} 的输入文件，跳过该蛋白质。")
            continue

        try:
            df = pd.read_csv(input_file)
            # 添加一列用于区分蛋白质
            df['protein_name'] = protein_name
            all_data.append(df)
            print(f"已读取 {protein_name} ({len(df)} 条序列)")
        except Exception as e:
            print(f"错误：读取 {input_file} 失败: {e}，跳过该蛋白质。")
            continue

    if not all_data:
        print("错误：没有可用的序列数据进行搜索。")
        return

    # 合并所有 DataFrame
    merged_df = pd.concat(all_data, ignore_index=True)
    total_sequences = len(merged_df)
    print(f"\n成功合并 {len(protein_list)} 个蛋白质，总序列数: {total_sequences}")

    # 创建总 FASTA 文件
    with open(TOTAL_TMP_FASTA, 'w') as f:
        for _, row in merged_df.iterrows():
            # 'name' 列已经是唯一的 ID (e.g., 1AKZ_A_M000)
            seq_name = str(row['name'])
            seq = str(row.seqres).lstrip("'").rstrip("'")
            f.write(f'>{seq_name}\n{seq}\n')

    print(f"已创建总临时FASTA文件: {TOTAL_TMP_FASTA}")

    # ----------------------------------------------------
    # 步骤 3: 运行单次 MMseqs 搜索
    # ----------------------------------------------------
    print("\n--- 2/4: 正在运行 MMseqs 搜索... ---")
    os.makedirs(TOTAL_TMP_OUTDIR, exist_ok=True)

    cmd = f'python -m scripts.mmseqs_search {TOTAL_TMP_FASTA} {db_dir} {TOTAL_TMP_OUTDIR}'
    print(f"运行命令: {cmd}")
    os.system(cmd)

    # ----------------------------------------------------
    # 步骤 4: 拆分和分发输出文件
    # ----------------------------------------------------
    print("\n--- 3/4: 正在拆分和分发输出文件... ---")

    # 遍历总临时输出目录中的所有 .a3m 文件
    processed_count = 0
    for name in os.listdir(TOTAL_TMP_OUTDIR):
        if not name.endswith('.a3m'):
            continue

        a3m_path = os.path.join(TOTAL_TMP_OUTDIR, name)

        # 读取 a3m 文件的第一行获取查询名称
        with open(a3m_path) as f:
            try:
                # 第一行形如 >1AKZ_A_M000
                full_query_name = next(f).strip()[1:]
            except StopIteration:
                print(f"警告：a3m文件 {name} 为空，跳过。")
                continue
            except Exception as e:
                print(f"读取文件 {name} 的第一行出错: {e}，跳过。")
                continue

        # 提取蛋白质名称 (假设 protein_name 是 name 的前几个字符，直到找到第一个下划线)
        # 例如 '1AKZ_A_M000' -> '1AKZ_A'
        try:
            # 这里的逻辑是获取第一个下划线前的部分。如果你的 protein name 中有下划线，
            # 且 cluster name 中也有，请根据实际情况调整。
            protein_name = full_query_name.split('_')[0] + '_' + full_query_name.split('_')[1]
            # 假设 protein name 总是 PDB_CHAIN (e.g. 1AKZ_A)
        except IndexError:
            # 如果名称格式不符合预期，跳过
            print(f"警告: 无法从查询名 {full_query_name} 中解析出蛋白质名称, 跳过。")
            continue

        if protein_name not in final_outdirs:
            print(f"警告：查询名称 {full_query_name} 对应的蛋白质 {protein_name} 不在原始列表中，跳过。")
            continue

        # 目标路径: {base_dir}/{protein_name}/msas_cluster_search/{full_query_name}.a3m
        target_dir = final_outdirs[protein_name]
        target_path = os.path.join(target_dir, f'{full_query_name}.a3m')

        # 移动并重命名文件
        try:
            shutil.move(a3m_path, target_path)
            processed_count += 1
        except Exception as e:
            print(f"错误：移动文件 {name} 到 {target_path} 失败: {e}")

    print(f"已成功分发 {processed_count} 个 .a3m 文件。")

    # ----------------------------------------------------
    # 步骤 5: 清理临时文件和目录
    # ----------------------------------------------------
    print("\n--- 4/4: 正在清理临时文件... ---")

    try:
        os.remove(TOTAL_TMP_FASTA)
    except Exception as e:
        print(f"清理临时 FASTA 文件失败: {e}")

    try:
        # 移除整个临时输出目录
        shutil.rmtree(TOTAL_TMP_OUTDIR)
    except Exception as e:
        print(f"清理临时输出目录失败: {e}")

    print("\n所有蛋白质的MMseqs搜索任务已优化完成。")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 读取蛋白质列表
    try:
        with open(args.protein_list_file, 'r') as f:
            protein_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：找不到蛋白质列表文件: {args.protein_list_file}")
        exit(1)

    if not protein_list:
        print("警告：蛋白质列表文件为空，没有需要处理的蛋白质。")
        exit(0)

    # 2. 运行批量处理
    run_batch_mmseqs_search(protein_list, args.base_dir, args.db_dir)