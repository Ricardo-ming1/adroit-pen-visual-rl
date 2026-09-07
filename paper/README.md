# Adroit Pen 中文 LaTeX 技术报告

`main.tex` 是论文源码入口，实验仓库是内容的事实源。报告由 17 个主体章节、附录、8 张结果图和一张 TikZ 架构图组成；所有数值均来自仓库内冻结的 V1--V6.1 结果，没有在制本文档时重新运行训练、confirmation 或 frozen test。

## 编译与打包

推荐使用本机持久化中文 Overleaf 镜像，确保与网页端环境一致：

```bash
cd paper
sg docker -c 'make docker-pdf'
make zip
```

若主机已经安装完整 TeX Live 2026，也可以运行 `make pdf`。清理中间文件使用 `make clean`；这只删除 `paper/build/` 内的 LaTeX 中间文件以及 `paper/dist/` 中可重建的 PDF/ZIP。

图表由仓库中已核对的 CSV/JSON 摘要生成：

```bash
make figures
```

正文模块由已审核的中文 Markdown 重排而来；若需要重新同步正文，运行：

```bash
python3 scripts/convert_verified_markdown.py
```

日常修改建议直接编辑 `sections/*.tex`。重新运行转换脚本会覆盖这些自动生成章节，因此只应在明确要从 Markdown 重新同步时执行。

## 导入本地 Overleaf

1. 运行 `make zip` 生成 `dist/adroit_pen_overleaf.zip`。
2. 浏览器打开本地 Overleaf，选择 **New Project → Upload Project**。
3. 上传 ZIP 后，在项目菜单中把编译器设为 **XeLaTeX**。
4. `main.tex` 位于 ZIP 根目录，Biber 会由 `latexmk` 自动执行。

若网页提示 `This compile didn’t produce a PDF`，先在项目左上角 **Menu → Settings → Compiler** 中选择 **XeLaTeX** 再重新编译；旧项目会保留最初导入时的 `pdfLaTeX` 元数据。当前 ZIP 也包含兼容配置，重新导入为新项目时可自动转交 XeLaTeX。

本地 Toolkit 固定信息：

- Toolkit 目录：`/path/to/adroit-overleaf-toolkit`
- Toolkit commit：`5221893829154a0c87d6b945a6952dd3f27e01fb`
- Overleaf CE：`sharelatex/sharelatex:6.3.0`
- 中文镜像：`local/adroit-overleaf-ce:6.3.0`
- TeX Live：2026，完整 `scheme-full`
- MongoDB：`8.0.4`（固定版本；兼容本机 kernel 7.0.0）
- 本地地址：`http://localhost:8080`
- 管理员初始化：`http://localhost:8080/launchpad`

## 服务维护

当前终端若尚未继承新加入的 `docker` group，请保留 `sg docker -c` 包装；重新登录后可以直接执行 Toolkit 命令。
首次部署或修改 Compose/Mongo 配置后使用 `sg docker -c 'bin/up -d'`，由 Toolkit 初始化 Mongo replica set；日常启停使用下列 `bin/start` 和 `bin/stop`。

```bash
cd /path/to/adroit-overleaf-toolkit
sg docker -c 'docker build -f config/Dockerfile.adroit-texlive -t local/adroit-overleaf-ce:6.3.0 .'
sg docker -c 'bin/start'                  # 启动
sg docker -c 'bin/stop'                   # 停止
sg docker -c 'bin/logs -f sharelatex'     # 查看日志
sg docker -c 'bin/docker-compose ps'      # 查看状态
sg docker -c 'bin/doctor'                 # 配置诊断
bin/backup-config -m zip "$HOME/overleaf-config-backup.zip"
```

完整数据备份应在停止服务后复制 Toolkit 的 `data/` 与 `config/`。这些目录可能包含账号和项目数据，备份文件不要加入 Git：

```bash
cd /path/to/adroit-overleaf-toolkit
sg docker -c 'bin/stop'
sudo tar --xattrs --acls -czf "$HOME/overleaf-data-backup.tar.gz" -C "$PWD" data config
sg docker -c 'bin/start'
```

## Overleaf 与 Git 的往返协作

Git 仓库始终是源码事实源。网页修改后，从 Overleaf 下载项目源码 ZIP，在临时目录解压并审阅差异；不要直接挂载或编辑 Overleaf 内部 MongoDB/项目数据目录。

```bash
git status --short
tmp_dir="$(mktemp -d)"
unzip ~/Downloads/adroit_pen_overleaf.zip -d "$tmp_dir"
diff -ru --exclude build --exclude dist paper "$tmp_dir"
```

合并前先提交、stash 或保留双方未提交修改，再逐文件复制需要的 `.tex`、图表和 BibTeX 变更。不要用 `git reset --hard`、破坏性 checkout 或覆盖整个 `paper/` 目录。

## 结果边界

最终 E2 以 25 条 human demonstrations 训练的 offline AWAC 为起点，随后还使用 544 条仿真轨迹、108,800 个时序样本和 privileged-state/action supervision。部署输入只有 RGB、proprioception 与动作历史；项目纯仿真，无 sim-to-real 或真实硬件结果。最终 checkpoint SHA-256：

```text
cc35a4130cb33df92fb635e58f125a219da76a2a89fd4109ba7e9267a3080b22
```
