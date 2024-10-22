# 初衷


![](https://p1.ssl.qhmsg.com/dr/270_500_/t01498c7b69f6323278.jpg?size=375x472)

传统机器学习和数据科学方面的基础书前前后后也涉猎不少中外版本，Jordan、Hastie、Bishop、周志华等大拿的名著很多人都耳熟能详，但个人最喜欢的还是李航博士的《统计学习方法》和Simon Rogers的《机器学习基础教程》，两位大拿的书都不厚，在有限篇幅把算法写清楚很考验作者的水平，李航博士的书最喜欢的就是EM和HMM两章，是自己看过所有版本里最好的，这两章不仅在传统机器学习，在强化学习中都起到很基础也很重要的作用。另外，Rogers书中线性模型写的很好，不少人觉得线性模型太简单，其实里面东西深入下去还是有很多的，很多非线性问题也是转换为线型问题进行求解，另外Rogers写的贝叶斯方法及推理也很棒，特别是里面的公式推导。

![](https://p.ssl.qhimg.com/dmsmfl/120_75_/t016c6de1c7c7fcf6db.png?size=591x400&phash=6897457096129765387)

另外，Julia语言是目前接触的科学计算语言里最喜欢的，学过Matlab、R和Python的同学如果学Julia的话会看到很多相似的地方，但相比前三者Julia快很多！去年底，Julia自2012年开始经过6年发展终于出了1.0版，所以自己打算用Julia写李航博士书里的算法，除一些很基础的包外，尽量不调用第三方包写核心算法。目前将书中感知机、k近邻法、朴素贝叶斯法、决策树、支持向量机和隐马尔可夫模型（含EM）六个核心算法写完，由于时间有限，还没有写打算年后有空再写的有逻辑斯蒂回归与最大熵模型和AdaBoost算法。过程中，几个算法的测试数据和决策树的树结构设计参考了[WenDesi](https://github.com/WenDesi/lihang_book_algorithm)的Python代码，支持向量机参考了[fengdu78](https://github.com/fengdu78/lihang-code)的Python代码，一并感谢！

注：代码中希腊字母Unicod在Github中无法显示。

