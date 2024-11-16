# Dynamic asset allocation

## 主要思路

主要思路来源于 [__Dynamic Asset Allocation Using Machine Learning: Seeing the Forest for the Trees__](https://www.pm-research.com/content/iijpormgmt/50/5/132)

1. 构建Growth factor增长变 Inflation factor通胀变量 Policy factor政策变量 Risk Appetite Indicator风险变量
2. 观测不同时期，资产未来12个月的表现
3. 构建随机森林模型
4. 回测

## 过程

1. 变量构建
   
   1.1 各种高频变量全部输入
   
   1.2 按照文献中，分别做Z-score组合
2. 分别使用1个月2个月3个月资产表现进行标签
3. 构建随机森林和Xgboost模型

