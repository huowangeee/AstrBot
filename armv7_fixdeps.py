import os
import re
import subprocess
import sys

"""
构建期依赖自愈。

为什么要它
----------
实测：镜像构建成功、容器一起来就崩，日志是
    File "/AstrBot/astrbot/core/log.py", line 20, in <module>
        from loguru import logger as _raw_loguru_logger
    ModuleNotFoundError: No module named 'loguru'

原因不是我们漏装，而是**上游的依赖清单本身漏了**：loguru 在代码里是直接
import 的，但 requirements.txt 和 pyproject.toml 的 dependencies 里都没有它，
清单里 40 多个包也没有任何一个传递依赖 loguru（我用 PyPI 元数据逐个核对过）。
从源码自建的镜像因此必然缺它。

与其在玩客云上等一次 30 分钟的 CI 才发现下一个漏项，不如在构建时把
`import astrbot.core` 真跑一遍：缺什么就装什么，装不上就把 traceback 打出来
让构建失败（fail fast），而不是做出一个起来就无限重启的镜像。

退出码
------
0  依赖齐（或者导入失败但不是缺依赖 —— 那种情况只警告，不阻断构建）
1  有模块装不上，或补齐轮次用尽
"""

APP_DIR = os.environ.get("ASTRBOT_APP_DIR", "/AstrBot")
PROBE = "import astrbot.core"
MAX_ROUNDS = 10

# 模块名 -> PyPI 包名（两者不一致的常见项）
ALIAS = {
    "yaml": "PyYAML",
    "dateutil": "python-dateutil",
    "PIL": "Pillow",
    "Crypto": "pycryptodome",
    "dotenv": "python-dotenv",
    "jwt": "PyJWT",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python-headless",
    "OpenSSL": "pyOpenSSL",
    "socks": "PySocks",
    "magic": "python-magic",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "serial": "pyserial",
    "jose": "python-jose",
    "multipart": "python-multipart",
    "prometheus_client": "prometheus-client",
    "sqlalchemy": "SQLAlchemy",
}


def probe():
    return subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=APP_DIR,
        capture_output=True,
        text=True,
        timeout=900,
    )


def main():
    installed = []
    for rnd in range(1, MAX_ROUNDS + 1):
        result = probe()
        if result.returncode == 0:
            print(
                "[armv7] 依赖自检通过（第 %d 轮，累计补齐: %s）"
                % (rnd, ", ".join(installed) or "无")
            )
            return 0

        match = re.search(r"No module named '([^']+)'", result.stderr)
        if not match:
            # 不是缺依赖导致的导入失败（例如需要在特定环境/目录下才能 import）。
            # 这种情况只警告，别把构建搞挂 —— 真正的运行时验证在设备上。
            print("[armv7] 警告: astrbot.core 导入失败，但不是缺依赖，跳过自愈")
            print("\n".join(result.stderr.splitlines()[-15:]))
            return 0

        module = match.group(1).split(".")[0]
        package = ALIAS.get(module, module)
        print("[armv7] 缺少模块 %s -> pip install %s" % (module, package))
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", package],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if install.returncode != 0:
            print("[armv7] 安装 %s 失败，日志尾部：" % package)
            print("\n".join(install.stdout.splitlines()[-20:]))
            print(install.stderr[-3000:])
            return 1
        installed.append(package)

    print("[armv7] %d 轮仍未补齐依赖，最后一次错误：" % MAX_ROUNDS)
    print(probe().stderr[-3000:])
    return 1


if __name__ == "__main__":
    sys.exit(main())
