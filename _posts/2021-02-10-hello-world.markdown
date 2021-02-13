---
layout: post
title: Wav2Lip
date: 2021-02-13 13:44:24.000000000 +09:00
---

### **A Lip Sync Expert Is All You Need for Speech to Lip GenerationIn The Wild**

- 提出了一种新的唇形同步网络—**Wav2Lip**，它比之前的工作在野外用任意语音对任意说话的人脸视频进行唇形同步的准确率要高得多。
- 提出了一个新的评估框架，包括新的基准和指标，以使在无约束视频中对唇形同步进行公平的判断。
- 收集并发布了**ReSyncED**，这是一个真实世界的lip-Sync Evaluation数据集，用于在完全没看过的野生视频上对lip-sync模型的性能进行基准测试。
- Wav2Lip是第一个独立于说话者的模型，可以生成与真实同步视频匹配的lip-sync精度的视频。人类的评估表明，在90%以上的情况下，Wav2Lip生成的视频优于现有的方法和未同步的版本。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="/assets/images/wav2lip输入输出.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">fg1.Wav2Lip网络的输入输出</div>
</center>