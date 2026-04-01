"""
VTBench — 从 R2 下载 UCRArchive_2018.zip 并解压
=================================================

用法:
    export R2_ACCOUNT_ID="..."
    export R2_ACCESS_KEY_ID="..."
    export R2_SECRET_ACCESS_KEY="..."
    export R2_BUCKET_NAME="..."

    python scripts/download_ucr_from_r2.py                    # 下载并解压
    python scripts/download_ucr_from_r2.py --no-extract       # 只下载不解压
    python scripts/download_ucr_from_r2.py --output-dir /data # 指定输出目录
"""

import os
import sys
import time
import zipfile
import argparse
from pathlib import Path

try:
    import boto3
    from botocore.config import Config
except ImportError:
    print("错误: 需要安装 boto3")
    print("运行: pip install boto3")
    sys.exit(1)


R2_KEY = "UCRArchive_2018.zip"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent


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

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def get_bucket_name():
    bucket = os.environ.get("R2_BUCKET_NAME")
    if not bucket:
        print("错误: 缺少环境变量 R2_BUCKET_NAME")
        sys.exit(1)
    return bucket


def format_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="从 R2 下载 UCRArchive_2018.zip")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--no-extract", action="store_true", help="只下载不解压")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT
    zip_path = output_dir / "UCRArchive_2018.zip"
    extract_dir = output_dir / "UCRArchive_2018"

    # 如果已经解压过，跳过
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"UCRArchive_2018/ 目录已存在且非空，跳过下载。")
        print(f"如需重新下载，请先删除 {extract_dir}")
        return

    client = get_r2_client()
    bucket = get_bucket_name()

    # 获取文件大小
    try:
        meta = client.head_object(Bucket=bucket, Key=R2_KEY)
        file_size = meta["ContentLength"]
        print(f"R2 文件: {R2_KEY} ({format_size(file_size)})")
    except client.exceptions.ClientError:
        print(f"错误: R2 上找不到 {R2_KEY}")
        print("请先运行 upload_ucr_to_r2.py 上传数据。")
        sys.exit(1)

    # 如果 zip 已存在且大小一致，跳过下载
    if zip_path.exists() and zip_path.stat().st_size == file_size:
        print(f"本地已有 {zip_path.name}，大小匹配，跳过下载。")
    else:
        # 下载
        downloaded = 0
        last_print = 0

        def progress_callback(bytes_transferred):
            nonlocal downloaded, last_print
            downloaded += bytes_transferred
            now = time.time()
            if now - last_print >= 2:
                pct = 100 * downloaded / file_size
                speed = downloaded / (now - start) / 1024 / 1024
                print(
                    f"  下载中: {format_size(downloaded)}/{format_size(file_size)} "
                    f"({pct:.1f}%) {speed:.1f} MB/s",
                    flush=True,
                )
                last_print = now

        transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=4,
        )

        print(f"下载到: {zip_path}")
        start = time.time()

        client.download_file(
            bucket,
            R2_KEY,
            str(zip_path),
            Callback=progress_callback,
            Config=transfer_config,
        )

        elapsed = time.time() - start
        speed = (file_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
        print(f"下载完成: {elapsed:.0f} 秒 ({speed:.1f} MB/s)")

    # 解压
    if not args.no_extract:
        print(f"\n解压到: {extract_dir}")
        start = time.time()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        elapsed = time.time() - start
        print(f"解压完成: {elapsed:.0f} 秒")

        # 验证
        n_dirs = sum(1 for d in extract_dir.iterdir() if d.is_dir())
        print(f"UCRArchive_2018/ 包含 {n_dirs} 个数据集目录")
    else:
        print("跳过解压 (--no-extract)")

    print("\n完成！")


if __name__ == "__main__":
    main()
