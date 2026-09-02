# 资源索引

每日文件中 `Gxx` 表示三维/机器人资源，`Mxx` 表示互信息/多模态资源。优先使用官方文档、公开课、原论文和作者代码；中文社区材料只用于辅助直觉。

## G：三维重建、机器人与仿真

后文用 `Rxx` 引用，避免每天重复长链接。先使用这些一手材料；知乎/B站只用于辅助理解，不作为公式和实现细节的最终依据。

### 数学与编程

- **G01 Python 官方教程**：[Python Tutorial](https://docs.python.org/3/tutorial/index.html?gdm=GetApp)
- **G02 NumPy 入门**：[NumPy absolute basics](https://numpy.org/doc/stable/user/absolute_beginners)
- **G03 Git 中文/英文免费书**：[Pro Git](https://git-scm.com/book/en/v2)
- **G04 CMake 官方教程**：[CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- **G05 线性代数直觉视频**：[3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- **G06 线性代数系统课**：[MIT 18.06SC Linear Algebra](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)
- **G07 PyTorch 官方入门**：[Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro)

### 三维视觉与重建

- **G08 OpenCV 官方教程**：[OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- **G09 TUM 多视图几何公开课**：[Computer Vision II: Multiple View Geometry](https://www.vision.in.tum.de/teaching/online/mvg)
- **G10 COLMAP 官方教程/源码**：[COLMAP image-based reconstruction tutorial](https://github.com/colmap/colmap/blob/main/doc/tutorial.rst)
- **G11 Open3D 官方教程**：[Open3D Tutorials](https://www.open3d.org/docs/latest/tutorial/)
- **G12 Open3D 完整 RGB-D 重建管线**：[Reconstruction System](https://www.open3d.org/docs/release/tutorial/reconstruction_system/index.html)
- **G13 Nerfstudio**：[官方文档](https://docs.nerf.studio/)；[第一个 NeRF](https://docs.nerf.studio/quickstart/first_nerf.html)
- **G14 3D Gaussian Splatting 原作者实现**：[graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

### 机器人、SLAM 与仿真

- **G15 Modern Robotics 免费视频与章节**：[Book Pages](https://modernrobotics.northwestern.edu/nu-gm-book-resource/)；[代码库](https://github.com/nxrlab/modernrobotics)
- **G16《视觉 SLAM 十四讲》第二版代码**：[gaoxiang12/slambook2](https://github.com/gaoxiang12/slambook2)
- **G17 GTSAM 因子图教程**：[Factor Graphs and GTSAM](https://gtsam.org/tutorials/intro.html)
- **G18 ORB-SLAM3 原作者仓库**：[UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/README.md?plain=1)
- **G19 ROS 2 文档**：[ROS Developer Documentation](https://docs.ros.org/index.html)；[Jazzy CLI 入门](https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools.html)
- **G20 Gazebo/ROS 2**：[Gazebo Harmonic 入门](https://gazebosim.org/docs/harmonic/getstarted/)；[ROS 2 集成](https://gazebosim.org/docs/harmonic/ros2_integration/)
- **G21 Isaac Sim（选修，需要合适 NVIDIA GPU）**：[ROS 2 Tutorials](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/index.html)

## M：互信息与多模态学习

### 数学、概率和深度学习

| 编号 | 资源 | 推荐用途 |
| --- | --- | --- |
| M1 | [3Blue1Brown《线性代数的本质》官方双语版](https://www.bilibili.com/list/ml1017712180?bvid=BV1ys411472E&oid=6731067) | 建立向量、矩阵和线性变换直觉 |
| M2 | [Harvard Stat 110 概率论公开视频](https://stat110.hsites.harvard.edu/youtube) | 系统学习条件概率、随机变量和联合分布[^1] |
| M3 | [MIT 6.012 概率论短视频](https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/resources/lecture-videos/) | 按知识点补充概率论 |
| M4 | [Deep Learning Book：概率与信息论](https://www.deeplearningbook.org/contents/prob.html) | 连接概率、信息论和深度学习[^2] |
| M5 | [《动手学深度学习》中文 GitHub](https://github.com/d2l-ai/d2l-zh) | PyTorch、神经网络与可运行 Notebook[^3] |
| M6 | [《动手学深度学习》中文课程](https://courses.d2l.ai/zh-v2/) | 配合 M5 观看视频 |

### 信息论

| 编号 | 资源 | 推荐用途 |
| --- | --- | --- |
| M7 | [西安电子科技大学《信息论基础》](https://www.bilibili.com/video/BV1tx411d7Rc/) | 中文学习自信息、熵、互信息和数据处理定理 |
| M8 | [MIT 6.441 Information Theory](https://ocw.mit.edu/courses/6-441-information-theory-spring-2010/) | 系统教材、讲义与习题[^4] |
| M9 | [Stanford EE376A Information Theory](https://web.stanford.edu/class/ee376a/) | 信息度量及其统计和机器学习联系[^5] |
| M10 | [Princeton 信息论基础单讲](https://www.youtube.com/watch?v=bkLHszLlH34) | 快速复习熵、KL 和互信息 |

### 多模态学习

| 编号 | 资源 | 推荐用途 |
| --- | --- | --- |
| M11 | [CMU 11-777 多模态机器学习课程表](https://cmu-multicomp-lab.github.io/mmml-course/fall2022/schedule/) | 完整视频、讲义和论文路径[^6] |
| M12 | [CMU 多模态课程第一讲](https://www.youtube.com/watch?v=6YsbpYSO_QM) | 建立多模态问题全景 |
| M13 | [Awesome Multimodal ML](https://github.com/pliang279/awesome-multimodal-ml) | 按主题检索课程和论文[^7] |

### 互信息代码

| 编号 | 资源 | 推荐用途 |
| --- | --- | --- |
| M14 | [Google Research：Variational Bounds on Mutual Information](https://github.com/google-research/google-research/tree/master/vbmi) | 比较 BA、NWJ、MINE、InfoNCE 等变分界[^8] |
| M15 | [torch-mist](https://github.com/mfederici/torch-mist) | PyTorch 互信息估计工具箱 |
| M16 | [MINE PyTorch 教学实现](https://github.com/gtegner/mine-pytorch) | 理解和复现 MINE |
| M17 | [Contrastive Multiview Coding 官方代码](https://github.com/HobbitLong/CMC) | 多视图对比学习实践[^9] |

### 多模态医学

| 编号 | 资源 | 推荐用途 |
| --- | --- | --- |
| M18 | [SimpleITK 官方教程](https://simpleitk.org/TUTORIAL/) | 医学图像处理与配准 Notebook、视频[^10] |
| M19 | [SimpleITK 互信息配准示例](https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethod2_2ImageRegistrationMethod2_8py-example.html) | 直接运行联合直方图 MI 配准 |
| M20 | [医学图像配准字幕课程](https://www.bilibili.com/video/BV1NZ4y1p7RT/) | 第 6–9 节学习联合熵、MI 和优化 |
| M21 | [多模态医学论文清单](https://github.com/czifan/Multimodal-Medicine-AI) | 检索多模态医学综述、模型和数据集 |

### 中文补充材料

| 编号 | 资源 | 推荐用途 |
| --- | --- | --- |
| M22 | [知乎“互信息”专题](https://www.zhihu.com/topic/20362476/intro) | 辅助建立直觉 |
| M23 | [知乎 NLP 课程中的信息论与互信息](https://www.zhihu.com/education/video-course/1564218549538607104?section_id=1566029318404378625) | 中文快速复习 |

> ⚠️ **使用建议：** 知乎适合辅助建立直觉；公式推导、论文结论和实验方法应回到教材、原论文或官方代码核对。
