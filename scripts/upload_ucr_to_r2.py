"""
VTBench — 上传 UCRArchive_2018.zip 到 Cloudflare R2
====================================================

UCR Archive 只有一个 302MB 的 zip 文件，直接上传即可。

用法:
    # 设置环境变量（同 upload_to_r2.py）
    $env:R2_ACCOUNT_ID = "你的account_id"
    $env:R2_ACCESS_KEY_ID = "你的access_key_id"
    $env:R2_SECRET_ACCESS_KEY = "你的secret_access_key"
    $env:R2_BUCKET_NAME = "你的bucket名称"

    # 运行
    python scripts/upload_ucr_to_r2.py                          # 上传 UCRArchive_2018.zip
    python scripts/upload_ucr_to_r2.py --file path/to/file.zip  # 指定文件路径
    python scripts/upload_ucr_to_r2.py --dry-run                # 只看计划
"""

import os
import sys
import time
import argparse
from pathlib import Path

try:
    import boto3
    from botocore.config import Config
except ImportError:
    print("错误: 需要安装 boto3")
    print("运行: pip install boto3")
    sys.exit(1)


DEFAULT_ZIP = Path(__file__).resolve().parent.parent / "UCRArchive_2018.zip"
R2_KEY = "UCRArchive_2018.zip"


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
    parser = argparse.ArgumentParser(description="上传 UCRArchive_2018.zip 到 R2")
    parser.add_argument("--file", type=str, default=None, help="zip 文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只显示计划")
    args = parser.parse_args()

    zip_path = Path(args.file) if args.file else DEFAULT_ZIP

    if not zip_path.exists():
        print(f"错误: 找不到文件 {zip_path}")
        sys.exit(1)

    file_size = zip_path.stat().st_size
    print(f"文件: {zip_path}")
    print(f"大小: {format_size(file_size)}")
    print(f"R2 目标: {R2_KEY}")

    if args.dry_run:
        print("\n[Dry Run] 将上传以上文件到 R2")
        return

    client = get_r2_client()
    bucket = get_bucket_name()

    # 检查连接
    try:
        client.head_bucket(Bucket=bucket)
        print(f"已连接到 R2 bucket: {bucket}")
    except Exception as e:
        print(f"错误: 无法访问 bucket '{bucket}': {e}")
        sys.exit(1)

    # 检查是否已存在
    try:
        existing = client.head_object(Bucket=bucket, Key=R2_KEY)
        existing_size = existing["ContentLength"]
        if existing_size == file_size:
            print(f"\nR2 上已有相同大小的文件 ({format_size(existing_size)})，跳过上传。")
            print("如需强制重新上传，请先在 R2 上删除该文件。")
            return
        else:
            print(f"\nR2 上有同名文件但大小不同 (R2: {format_size(existing_size)}, 本地: {format_size(file_size)})")
            print("将覆盖上传...")
    except client.exceptions.ClientError:
        pass  # 文件不存在，正常上传

    # 上传
    uploaded = 0
    last_print = 0

    def progress_callback(bytes_transferred):
        nonlocal uploaded, last_print
        uploaded += bytes_transferred
        now = time.time()
        if now - last_print >= 2:
            pct = 100 * uploaded / file_size
            print(f"  上传中: {format_size(uploaded)}/{format_size(file_size)} ({pct:.1f}%)", flush=True)
            last_print = now

    transfer_config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=4,
    )

    print(f"\n开始上传...")
    start = time.time()

    client.upload_file(
        str(zip_path),
        bucket,
        R2_KEY,
        ExtraArgs={"ContentType": "application/zip"},
        Callback=progress_callback,
        Config=transfer_config,
    )

    elapsed = time.time() - start
    speed = (file_size / 1024 / 1024) / elapsed if elapsed > 0 else 0

    print(f"\n上传完成！")
    print(f"  耗时: {elapsed:.0f} 秒")
    print(f"  速度: {speed:.1f} MB/s")
    print(f"  R2 路径: {R2_KEY}")


if __name__ == "__main__":
    main()
