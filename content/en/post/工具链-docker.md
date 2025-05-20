---
date: 2025-05-16T04:00:59-07:00
description: "Docker 是一个用于开发、发布和运行应用的开放平台。Docker 使你能够将应用与基础设施分离，从而快速交付软件。"
featured_image: "/images/docker/tomo.jpg"
tags: ["tool"]
title: "工具链 - docker"
---

虚拟机的缺点：OS 太重、慢。

![1](/images/docker/1.JPG)

容器技术：**只隔离应用程序的运行时环境，但容器之间可共享同一操作系统。**

![2](/images/docker/2.JPG)

容器技术的代表：docker

<!--more-->

&nbsp;

程序的表现只和集装箱(容器)有关系，和集装箱放在哪个货船或者哪个港口(操作系统)没有关系。

#### 如何使用 docker

1. 在 **dockerfile** 中指定需要哪些程序、依赖什么样的配置；
2. **docker build命令**：把 dockerfile 交给“编译器” docker 进行“编译”，生成的可执行程序 **image**；
3. **docker run命令**：运行 image，image运行起来后就是docker **container**。

&nbsp;

#### docker 如何工作

+ **client-server 模式**

  **client**：负责处理用户输入的各种命令，比如docker build、docker run。

  **server**(docker demon)：负责工作。

  注：docker client 和docker demon 可运行在同一台机器上。

![6](/images/docker/6.png)&nbsp;

1. **docker build**

   写完 dockerfile 交给 docker “编译”时使用该命令

   ![3](/images/docker/3.png)

   client 接收请求后转发给 docker daemon，docker daemon 根据 dockerfile 创建出“可执行程序” image。

2. **docker run**

   得到“可执行程序” image 后使用命令 docker run

   ![4](/images/docker/4.png)

   docker daemon 接收命令后找到具体的 image，然后加载到内存开始执行。image 执行起来就是所谓的container。

3. **docker pull**

   docker pull 命令可下载到别人编写好的image。

   ![5](/images/docker/5.png)

   用户通过 docker client 发送命令，docker daemon 接收到命令后向 docker registry 发送 image 下载请求，下载后存放在本地使用。

   docker registry(docker Hub)：用来存放可以供任何人下载 image 的公共仓库。

&nbsp;

#### docker 的底层实现

docker 基于 Linux 内核提供两项功能实现：

- **NameSpace**

  一种资源隔离方案，在该机制下资源不再是全局的，而是属于某个特定的 NameSpace，各个 NameSpace 下的资源互不干扰。

- **Control groups**

  控制容器中进程对系统资源的访问。比如可以限制某个容器使用内存的上限、可以在哪些 CPU 上运行等等。





mac 部署 RAGFlow：https://jishuzhan.net/article/1889102793436827650