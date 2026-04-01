"""
VTBench — 从 Cloudflare R2 下载 chart_images/（云端训练用）
============================================================

每个数据集是一个独立的 tar.gz，下载后自动解压到 chart_images/。

用法:
    # 安装依赖
    pip install boto3

    # 设置环境变量
    export R2_ACCOUNT_ID="..."
    export R2_ACCESS_KEY_ID="..."
    export R2_SECRET_ACCESS_KEY="..."
    export R2_BUCKET_NAME="..."

    # 查看远端有哪些数据集
    python scripts/download_from_r2.py --list

    # 下载指定数据集（云端训练常用）
    python scripts/download_from_r2.py --only GunPoint ECG5000 FordA

    # 下载全部
    python scripts/download_from_r2.py

    # 下载到自定义路径
    python scripts/download_from_r2.py --output /data/chart_images

    # 只下载不解压（保留tar.gz）
    python scripts/download_from_r2.py --only GunPoint --no-extract
"""

import os
import sys
import time
import tarfile
import argparse
from pathlib import Path

try:
    import boto3
    from botocore.config import Config
except ImportError:
    print("错误: 需要安装 boto3")
    print("运行: pip install boto3")
    sys.exit(1)


R2_PREFIX = "chart_images"


def get_r2_client():
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")

    missing = []
    if not account_id:
        missing.append("R2_ACCOUNT_ID")
    if not access_key:
        missing.append("R2_ACCESS_KEY_ID")
    if not secret_key:
        missing.append("R2_SECRET_ACCESS_KEY")

    if missing:
        print(f"错误: 缺少环境变量: {', '.join(missing)}")
        sys.exit(1)

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=50,
        ),
    )


def get_bucket_name():
    bucket = os.environ.get("R2_BUCKET_NAME")
    if not bucket:
        print("错误: 缺少环境变量 R2_BUCKET_NAME")
        sys.exit(1)
    return bucket


def list_remote_tar_files(client, bucket):
    """列出远端所有 tar.gz 及其大小"""
    files = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{R2_PREFIX}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tar.gz"):
                files.append({"key": obj["Key"], "size": obj["Size"]})
    return files


def format_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_with_progress(client, bucket, r2_key, local_path):
    """下载文件并显示进度"""
    # 先获取文件大小
    head = client.head_object(Bucket=bucket, Key=r2_key)
    file_size = head["ContentLength"]

    downloaded = 0
    last_print = 0

    def progress_callback(bytes_transferred):
        nonlocal downloaded, last_print
        downloaded += bytes_transferred
        now = time.time()
        if now - last_print >= 2:
            pct = 100 * downloaded / file_size if file_size > 0 else 0
            print(
                f"    下载中: {format_size(downloaded)}/{format_size(file_size)} "
                f"({pct:.1f}%)",
                flush=True,
            )
            last_print = now

    transfer_config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=4,
    )

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client.download_file(
        bucket, r2_key, str(local_path),
        Callback=progress_callback,
        Config=transfer_config,
    )
    return file_size


def extract_tar_gz(tar_path, output_dir):
    """解压 tar.gz 到指定目录"""
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir)


def main():
    parser = argparse.ArgumentParser(description="从 R2 下载 chart_images (tar.gz)")
    parser.add_argument("--only", nargs="+", help="只下载指定数据集 (如 GunPoint ECG5000)")
    parser.add_argument("--output", type=str, default="chart_images",
                        help="解压到的本地目录 (默认 chart_images)")
    parser.add_argument("--list", action="store_true", help="只列出远端数据集，不下载")
    parser.add_argument("--resume", action="store_true",
                        help="跳过本地已存在的数据集目录")
    parser.add_argument("--no-extract", action="store_true",
                        help="只下载 tar.gz，不自动解压")
    args = parser.parse_args()

    client = get_r2_client()
    bucket = get_bucket_name()
    output_dir = Path(args.output)

    # 列出远端 tar.gz
    print("正在扫描 R2...")
    remote_files = list_remote_tar_files(client, bucket)
    total_size = sum(f["size"] for f in remote_files)
    print(f"  远端共 {len(remote_files)} 个数据集, {format_size(total_size)}")

    if not remote_files:
        print("R2 上没有找到任何 tar.gz 文件。")
        return

    # --list 模式
    if args.list:
        print(f"\n远端数据集 ({len(remote_files)} 个):")
        for f in sorted(remote_files, key=lambda x: x["key"]):
            ds_name = os.path.basename(f["key"]).replace(".tar.gz", "")
            print(f"  {ds_name}: {format_size(f['size'])}")
        print(f"\n总计: {format_size(total_size)}")
        return

    # 过滤数据集
    if args.only:
        filtered = []
        for f in remote_files:
            ds_name = os.path.basename(f["key"]).replace(".tar.gz", "").replace("_images", "")
            if ds_name in args.only:
                filtered.append(f)
        remote_files = filtered
        print(f"  过滤后: {len(remote_files)} 个数据集")

    if not remote_files:
        print("没有匹配的数据集。")
        return

    # 逐个下载 + 解压
    completed = 0
    skipped = 0
    failed = 0
    total_start = time.time()

    for i, rf in enumerate(sorted(remote_files, key=lambda x: x["key"])):
        ds_name = os.path.basename(rf["key"]).replace(".tar.gz", "")
        tar_path = output_dir / f"{ds_name}.tar.gz"
        ds_dir = output_dir / ds_name

        print(f"\n[{i+1}/{len(remote_files)}] {ds_name} ({format_size(rf['size'])})")

        # Resume: 跳过已存在的目录
        if args.resume and ds_dir.exists():
            print(f"  跳过 (本地目录已存在)")
            skipped += 1
            continue

        # 下载
        try:
            dl_start = time.time()
            download_with_progress(client, bucket, rf["key"], str(tar_path))
            dl_time = time.time() - dl_start
            speed = (rf["size"] / 1024 / 1024) / dl_time if dl_time > 0 else 0
            print(f"  下载完成: {dl_time:.0f} 秒 ({speed:.1f} MB/s)")
        except Exception as e:
            print(f"  下载失败: {e}")
            failed += 1
            continue

        # 解压
        if not args.no_extract:
            print(f"  解压中...")
            try:
                ext_start = time.time()
                extract_tar_gz(str(tar_path), str(output_dir))
                ext_time = time.time() - ext_start
                print(f"  解压完成: {ext_time:.0f} 秒")
                # 删除tar
                tar_path.unlink()
            except Exception as e:
                print(f"  解压失败: {e}")
                print(f"  tar.gz 保留在: {tar_path}")
                failed += 1
                continue

        completed += 1

    total_time = time.time() - total_start
    print()
    print("=" * 60)
    print(f"下载完成！")
    print(f"  成功: {completed}")
    print(f"  跳过: {skipped}")
    print(f"  失败: {failed}")
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print("=" * 60)


if __name__ == "__main__":
    main()
