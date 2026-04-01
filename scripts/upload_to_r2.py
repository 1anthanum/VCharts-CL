"""
VTBench — 按数据集打包上传 chart_images/ 到 Cloudflare R2
=========================================================

策略: 每个数据集目录打包成一个独立的 tar.gz，上传单个大文件。
      比逐文件上传快 50-100 倍（540万文件 → 149个tar.gz）。

用法:
    # 第一步：安装依赖（只需一次）
    pip install boto3

    # 第二步：设置环境变量
    # Windows PowerShell:
    $env:R2_ACCOUNT_ID = "你的account_id"
    $env:R2_ACCESS_KEY_ID = "你的access_key_id"
    $env:R2_SECRET_ACCESS_KEY = "你的secret_access_key"
    $env:R2_BUCKET_NAME = "你的bucket名称"

    # Linux/Mac:
    export R2_ACCOUNT_ID="你的account_id"
    export R2_ACCESS_KEY_ID="你的access_key_id"
    export R2_SECRET_ACCESS_KEY="你的secret_access_key"
    export R2_BUCKET_NAME="你的bucket名称"

    # 第三步：运行
    python scripts/upload_to_r2.py                          # 上传全部
    python scripts/upload_to_r2.py --only GunPoint ECG5000  # 只传指定数据集
    python scripts/upload_to_r2.py --dry-run                # 只看计划不实际传
    python scripts/upload_to_r2.py --resume                 # 跳过已上传的数据集
    python scripts/upload_to_r2.py --tar-dir D:\\temp\\tar  # 指定tar临时目录
    python scripts/upload_to_r2.py --keep-tar               # 上传后不删除tar文件

环境变量获取方式:
    1. 登录 Cloudflare Dashboard → R2 → Overview
    2. 右上角 "Manage R2 API Tokens" → 创建 API Token
    3. 权限选 "Object Read & Write"
    4. 复制 Access Key ID 和 Secret Access Key
    5. Account ID 在 Cloudflare Dashboard 右侧栏可以看到
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


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

CHART_IMAGES_DIR = Path(__file__).resolve().parent.parent / "chart_images"
R2_PREFIX = "chart_images"  # R2 中的路径前缀
TAR_DIR = Path(__file__).resolve().parent.parent / "_tar_staging"


def get_r2_client():
    """创建 R2 客户端（兼容 S3 API）"""
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
        print()
        print("请设置以下环境变量:")
        print('  $env:R2_ACCOUNT_ID = "..."        # PowerShell')
        print('  $env:R2_ACCESS_KEY_ID = "..."')
        print('  $env:R2_SECRET_ACCESS_KEY = "..."')
        print('  $env:R2_BUCKET_NAME = "..."')
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


def list_remote_tar_keys(client, bucket):
    """列出 R2 上已有的 tar.gz 文件"""
    keys = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{R2_PREFIX}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tar.gz"):
                keys.add(obj["Key"])
    return keys


def get_dataset_dirs(base_dir, only_datasets=None):
    """获取所有数据集目录"""
    dirs = sorted(
        (d for d in os.scandir(base_dir) if d.is_dir()),
        key=lambda d: d.name,
    )
    if only_datasets:
        dirs = [
            d for d in dirs
            if d.name.replace("_images", "") in only_datasets
        ]
    return dirs


def count_files_fast(directory):
    """快速计算目录下的文件数"""
    count = 0
    for _, _, filenames in os.walk(directory):
        count += len(filenames)
    return count


def create_tar_gz(dataset_dir, tar_path):
    """将一个数据集目录打包成 tar.gz，显示进度"""
    dataset_name = os.path.basename(dataset_dir)
    file_count = 0

    with tarfile.open(tar_path, "w:gz", compresslevel=1) as tar:
        # compresslevel=1: 最快压缩，PNG本身已经压缩过，高压缩率没意义
        for dirpath, _, filenames in os.walk(dataset_dir):
            for fname in filenames:
                filepath = os.path.join(dirpath, fname)
                arcname = os.path.relpath(filepath, os.path.dirname(dataset_dir))
                tar.add(filepath, arcname=arcname)
                file_count += 1
                if file_count % 5000 == 0:
                    print(f"    打包中: {file_count:,} 文件...", flush=True)

    return file_count


def upload_with_progress(client, bucket, tar_path, r2_key):
    """上传文件并显示进度"""
    file_size = os.path.getsize(tar_path)
    uploaded = 0
    last_print = 0

    def progress_callback(bytes_transferred):
        nonlocal uploaded, last_print
        uploaded += bytes_transferred
        now = time.time()
        if now - last_print >= 2:  # 每2秒打印一次
            pct = 100 * uploaded / file_size if file_size > 0 else 0
            print(
                f"    上传中: {format_size(uploaded)}/{format_size(file_size)} "
                f"({pct:.1f}%)",
                flush=True,
            )
            last_print = now

    # 大文件用 multipart upload（boto3自动处理）
    transfer_config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=64 * 1024 * 1024,   # 64MB 以上分片
        multipart_chunksize=64 * 1024 * 1024,    # 每片 64MB
        max_concurrency=4,
    )

    client.upload_file(
        str(tar_path),
        bucket,
        r2_key,
        ExtraArgs={"ContentType": "application/gzip"},
        Callback=progress_callback,
        Config=transfer_config,
    )


def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="按数据集打包上传 chart_images 到 R2")
    parser.add_argument("--only", nargs="+", help="只上传指定数据集 (如 GunPoint ECG5000)")
    parser.add_argument("--dry-run", action="store_true", help="只显示计划，不实际操作")
    parser.add_argument("--resume", action="store_true", help="跳过 R2 上已存在的数据集")
    parser.add_argument("--chart-dir", type=str, default=None, help="chart_images 目录路径")
    parser.add_argument("--tar-dir", type=str, default=None, help="tar.gz 临时存放目录")
    parser.add_argument("--keep-tar", action="store_true", help="上传后保留 tar.gz 文件")
    args = parser.parse_args()

    base_dir = Path(args.chart_dir) if args.chart_dir else CHART_IMAGES_DIR
    tar_dir = Path(args.tar_dir) if args.tar_dir else TAR_DIR

    if not base_dir.exists():
        print(f"错误: 找不到目录 {base_dir}")
        sys.exit(1)

    tar_dir.mkdir(parents=True, exist_ok=True)

    # 扫描数据集目录
    print(f"扫描 {base_dir} ...")
    dataset_dirs = get_dataset_dirs(base_dir, only_datasets=args.only)
    print(f"  找到 {len(dataset_dirs)} 个数据集目录")

    if not dataset_dirs:
        print("没有找到需要上传的数据集。")
        return

    # Dry-run 模式
    if args.dry_run:
        print(f"\n=== Dry Run 模式 ===")
        for d in dataset_dirs:
            n_files = count_files_fast(d.path)
            print(f"  {d.name}: ~{n_files:,} 文件 → {d.name}.tar.gz")
        print(f"\n总计: {len(dataset_dirs)} 个 tar.gz 将上传到 R2")
        return

    # 连接 R2
    client = get_r2_client()
    bucket = get_bucket_name()

    try:
        client.head_bucket(Bucket=bucket)
        print(f"已连接到 R2 bucket: {bucket}")
    except Exception as e:
        print(f"错误: 无法访问 bucket '{bucket}': {e}")
        sys.exit(1)

    # Resume 模式
    existing_keys = set()
    if args.resume:
        print("正在检查已上传的数据集...")
        existing_keys = list_remote_tar_keys(client, bucket)
        print(f"  R2 上已有 {len(existing_keys)} 个 tar.gz 文件")

    # 逐个数据集：打包 → 上传 → 删除tar
    total_start = time.time()
    completed = 0
    failed = 0
    skipped = 0

    for i, ds_entry in enumerate(dataset_dirs):
        ds_name = ds_entry.name
        r2_key = f"{R2_PREFIX}/{ds_name}.tar.gz"
        tar_path = tar_dir / f"{ds_name}.tar.gz"

        print(f"\n[{i+1}/{len(dataset_dirs)}] {ds_name}")

        # Resume: 跳过已上传的
        if args.resume and r2_key in existing_keys:
            print(f"  跳过 (已存在于 R2)")
            skipped += 1
            continue

        # 如果tar已存在（之前上传中断），直接上传
        if tar_path.exists():
            print(f"  发现已有 tar.gz ({format_size(tar_path.stat().st_size)}), 直接上传")
        else:
            # 打包
            print(f"  打包中...")
            pack_start = time.time()
            try:
                n_files = create_tar_gz(ds_entry.path, str(tar_path))
                pack_time = time.time() - pack_start
                tar_size = tar_path.stat().st_size
                print(
                    f"  打包完成: {n_files:,} 文件 → {format_size(tar_size)}, "
                    f"耗时 {pack_time:.0f} 秒"
                )
            except Exception as e:
                print(f"  打包失败: {e}")
                failed += 1
                continue

        # 上传
        print(f"  上传到 R2: {r2_key}")
        upload_start = time.time()
        try:
            upload_with_progress(client, bucket, str(tar_path), r2_key)
            upload_time = time.time() - upload_start
            tar_size = tar_path.stat().st_size
            speed_mbps = (tar_size / 1024 / 1024) / upload_time if upload_time > 0 else 0
            print(f"  上传完成: {upload_time:.0f} 秒 ({speed_mbps:.1f} MB/s)")
            completed += 1
        except Exception as e:
            print(f"  上传失败: {e}")
            print(f"  tar.gz 保留在: {tar_path} (下次 --resume 会跳过已上传的)")
            failed += 1
            continue

        # 删除临时tar（除非 --keep-tar）
        if not args.keep_tar:
            tar_path.unlink()
            print(f"  已删除临时文件")

    total_time = time.time() - total_start

    print()
    print("=" * 60)
    print(f"上传完成！")
    print(f"  成功: {completed}")
    print(f"  跳过: {skipped}")
    print(f"  失败: {failed}")
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print("=" * 60)

    if failed > 0:
        print("\n提示: 使用 --resume 重新运行可以跳过已上传的数据集，只重试失败的")


if __name__ == "__main__":
    main()
