---
title: Feishu2GithubPages
date: 2025-09-08
categories:
  - build_your_web
tags:
  - github
  - 教程
---

# Feishu2GithubPages

这篇教程讲的是把飞书文档下载为 Markdown，然后发布到 GitHub Pages。

## 配置

先通过 `feishu2md config --appId your_id --appSecret your_secret` 生成配置文件。

## 下载

下载单个文档使用 `feishu2md dl "文档链接"`。
批量下载文件夹可以使用 `feishu2md dl --batch -o output_directory "文件夹链接"`。
