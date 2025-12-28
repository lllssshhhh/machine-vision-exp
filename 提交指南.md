# Git和GitHub使用完整指南

本文档详细介绍如何使用Git工具将本地项目上传到GitHub，让其他人可以访问你的项目。

## 目录
- [前期准备](#前期准备)
- [第一步：初始化本地Git仓库](#第一步初始化本地git仓库)
- [第二步：在GitHub上创建远程仓库](#第二步在github上创建远程仓库)
- [第三步：连接本地仓库与远程仓库](#第三步连接本地仓库与远程仓库)
- [第四步：推送代码到GitHub](#第四步推送代码到github)
- [日常使用：更新代码到GitHub](#日常使用更新代码到github)

---

## 前期准备

### 1. 安装Git
- **Windows**: 访问 [git-scm.com](https://git-scm.com/) 下载安装包
- **Mac**: 使用 `brew install git` 或从官网下载
- **Linux**: `sudo apt-get install git` (Ubuntu/Debian)

验证安装：
```bash
git --version
```

### 2. 配置Git用户信息
首次使用Git需要配置用户名和邮箱：
```bash
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱"
```

**推荐**：如果要上传到GitHub，建议使用GitHub的no-reply邮箱：
```bash
git config --global user.email "你的GitHub用户名@users.noreply.github.com"
```

### 3. 注册GitHub账号
访问 [github.com](https://github.com) 注册一个免费账号。

---

## 第一步：初始化本地Git仓库

在你的项目文件夹中执行以下操作：

### 1.1 进入项目目录
```bash
cd /path/to/your/project
```

### 1.2 初始化Git仓库
```bash
git init
```
这会在项目根目录创建一个 `.git` 文件夹，项目就变成了一个Git仓库。

### 1.3 创建 .gitignore 文件
`.gitignore` 文件用于指定哪些文件不需要上传到GitHub（如临时文件、敏感信息、大型数据文件等）。

创建 `.gitignore` 文件并添加以下内容：
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/

# 数据文件
*.csv
*.xlsx
*.npy
*.pth
*.h5

# IDE
.vscode/
.idea/
*.swp

# 系统文件
.DS_Store
Thumbs.db

# 其他
*.log
.env
```

### 1.4 添加文件到暂存区
```bash
# 添加所有文件
git add .

# 或者添加特定文件
git add filename1 filename2
```

### 1.5 提交到本地仓库
```bash
git commit -m "Initial commit: 项目初始化"
```

💡 **提示**：提交信息应该简洁明了地描述本次提交的内容。

---

## 第二步：在GitHub上创建远程仓库

### 2.1 登录GitHub
访问 [github.com](https://github.com) 并登录你的账号。

### 2.2 创建新仓库
1. 点击右上角的 **"+"** 按钮
2. 选择 **"New repository"**

### 2.3 填写仓库信息
- **Repository name**（仓库名称）：
  - 建议使用英文和连字符，如 `my-project` 或 `CV-Experiments`
  - 避免使用中文（可能导致URL编码问题）
  
- **Description**（描述）：可选，简短描述项目内容

- **Public / Private**（公开/私有）：
  - **Public**：任何人都可以访问（推荐用于开源项目）
  - **Private**：只有你和授权的人可以访问

- **重要选项**：
  - **不要勾选** "Add a README file"
  - **不要勾选** "Add .gitignore"
  - **不要勾选** "Choose a license"
  
  *(因为我们已经在本地创建了项目，勾选这些会导致冲突)*

### 2.4 创建仓库
点击 **"Create repository"** 按钮。

### 2.5 复制仓库URL
创建完成后，GitHub会显示一个HTTPS链接，类似：
```
https://github.com/你的用户名/仓库名.git
```
复制这个链接，后续会用到。

---

## 第三步：连接本地仓库与远程仓库

### 3.1 添加远程仓库
在本地项目目录执行：
```bash
git remote add origin https://github.com/你的用户名/仓库名.git
```

`origin` 是远程仓库的默认名称（可以自定义，但一般使用origin）。

### 3.2 验证远程仓库配置
```bash
git remote -v
```
应该显示：
```
origin  https://github.com/你的用户名/仓库名.git (fetch)
origin  https://github.com/你的用户名/仓库名.git (push)
```

---

## 第四步：推送代码到GitHub

### 4.1 推送代码
```bash
git push -u origin main
```
或者（如果你的主分支名是其他名称）：
```bash
git push -u origin 分支名
```

**常见分支名**：`main`, `master`, `CV_Exp` 等

### 4.2 输入GitHub凭据
首次推送时，会提示输入GitHub用户名和密码：
- **用户名**：你的GitHub用户名
- **密码**：需要使用 **Personal Access Token**（个人访问令牌），而不是账号密码

#### 如何获取Personal Access Token：
1. 访问 GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 点击 "Generate new token (classic)"
3. 设置权限，至少勾选 `repo` 相关权限
4. 生成并复制token（只显示一次，务必保存）

### 4.3 验证上传成功
访问你的GitHub仓库页面：
```
https://github.com/你的用户名/仓库名
```
应该能看到你上传的所有文件。

---

## 日常使用：更新代码到GitHub

当你在本地修改了代码，想要更新到GitHub时：

### 1. 查看修改状态
```bash
git status
```

### 2. 添加修改的文件
```bash
# 添加所有修改
git add .

# 或添加指定文件
git add 文件名
```

### 3. 提交修改
```bash
git commit -m "描述本次修改的内容"
```

提交信息示例：
- `"修复了图像处理bug"`
- `"添加了新的实验代码"`
- `"更新了README文档"`

### 4. 推送到GitHub
```bash
git push
```

### 5. 完整流程（三步走）
```bash
git add .
git commit -m "更新说明"
git push
```

---

## 常用Git命令速查

| 命令 | 说明 |
|------|------|
| `git init` | 初始化Git仓库 |
| `git status` | 查看仓库状态 |
| `git add .` | 添加所有文件到暂存区 |
| `git add 文件名` | 添加指定文件到暂存区 |
| `git commit -m "消息"` | 提交到本地仓库 |
| `git push` | 推送到远程仓库 |
| `git pull` | 从远程仓库拉取更新 |
| `git clone URL` | 克隆远程仓库到本地 |
| `git branch` | 查看分支 |
| `git checkout -b 分支名` | 创建并切换分支 |
| `git log` | 查看提交历史 |
| `git remote -v` | 查看远程仓库配置 |

---

## 分支管理基础

### 创建新分支
```bash
git checkout -b 新分支名
```

### 切换分支
```bash
git checkout 分支名
```

### 查看所有分支
```bash
git branch -a
```

### 合并分支
```bash
# 切换到主分支
git checkout main

# 合并其他分支
git merge 分支名
```

---

## 总结

完整流程回顾：

1. **本地初始化**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **GitHub创建仓库**
   - 访问 github.com
   - 新建仓库（不勾选任何初始化选项）

3. **连接并推送**
   ```bash
   git remote add origin https://github.com/用户名/仓库名.git
   git push -u origin main
   ```

4. **日常更新**
   ```bash
   git add .
   git commit -m "更新说明"
   git push
   ```

---


