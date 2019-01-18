# encoding = utf-8
# Author: Liu Jing
# Date: 2019-01-14
# Email: zhulincaowu@gmail.com
# Last modified by: Liu Jing
# Last modified time: 2019-01-18

# Julia Version 1.0.2

using  DataFrames, CSV,   NPZ ,Pkg,LinearAlgebra,Random
features = npzread("features.npy")
raw_data = CSV.read("train_binary.csv")

labels = raw_data[1]; # if use `raw_data[:,1]` will copy the data

"""
   train_test_split(data; train_per, rng)

将数据集分割为训练集和测试集
...
# Arguments
- `data::Array{T, N}`: the input data to split

# Return
- 第一个是训练集，第二个是测试集
...

# Examples
```jldoctest
julia> train_features, test_features = train_test_split(features) # split the features
(Float32[0.0 0.0 … 0.019473 0.0867527; ...])

julia> train_labels, test_labels = train_test_split(labels) # split the labels
(Union{Missing, Int64}[1; 0; … ; 1; 1], Union{Missing, Int64}[1; 1; … ; 1; 1])
```
"""
function train_test_split(data :: Array{T, N}; train_per = 0.67, rng = MersenneTwister(1234)) where {T, N}
    n = size(data,1)
    ind = shuffle(rng, collect(1:n))
    train_ind = ind[1:floor(Int, n * train_per)]
    test_ind = ind[(floor(Int, n * train_per) + 1):n]
    data[train_ind, :], data[test_ind, :]
end

train_labels, test_labels = train_test_split(labels)
train_features, test_features = train_test_split(features)

train_labels = reshape(train_labels, (length(train_labels),))

# 读取图片二值化后的数据
train_bina_data = npzread("train_bina_data.npy")
test_bina_data = npzread("test_bina_data.npy")


"""
   cal_dim(train_features, train_labels)

计算训练数据的一些维度
...
# Arguments
- `train_features`: 训练集特征向量
- `train_labels`: 训练集标签

# Return
- `class_num`: 标签类的取值
- `len_class_num`: 标签类有多少种取值
- `features_num`: 训练集的特征数
...

# Examples
```jldoctest
julia> class_num, len_class_num, features_num = cal_dim(train_features, train_labels)
(Union{Missing, Int64}[1, 0], 2, 28140, 324)
```
"""
function cal_dim(train_features, train_labels)
    class = unique(train_labels) # 输出的种类
    class_num = length(class)
    sample_num = length(train_labels) # 样本容量
    features_num = size(train_features,2) # 训练集实例的特征数
    return class, class_num, sample_num, features_num
end
class, class_num, sample_num, features_num = cal_dim(train_features, train_labels)


"""
   empirical_entropy(D)

计算数据集的经验熵，计算熵只依赖D的分布，而与D的取值无关
...
# Arguments
- `D`: 训练数据集

# Return
- `H_D`: 经验熵
...

# Examples
```jldoctest
julia> empirical_entropy(train_labels)
0.46435992201638326
```
"""
function empirical_entropy(D)
    H_D = 0.0
    for i = 1:class_num
        ck_sample_num = sum(D .== class[i]) # 属于类C_k的样本个数，公式5.7
        if ck_sample_num == 0 # 令0log0=0
            continue
        end
        H_D += -(ck_sample_num/sample_num) * log2(ck_sample_num/sample_num)
    end
    return H_D
end

"""
   conditional_entropy(D, feature)

计算数据集的经验条件熵
...
# Arguments
- `D`: 训练数据集
- `feature`: 训练数据集某一特征向量

# Return
- `H_D_A`: 经验条件熵
...

# Examples
```jldoctest
julia> conditional_entropy(train_bina_data, 400)
2.0319072257698375e14
```
"""
function conditional_entropy(D, feature)
    D_feature = D[:, feature] # 选取训练数据集的某一特征向量列
    feature_val = unique(D_feature) # 该特征向量的取值
    feature_val_len = length(feature_val)

    H_D_A = 0.0
    for i = 1:feature_val_len
        di = D_feature[D_feature .== feature_val[i]] #根据feature将D划分为feature_val_len个子集， 即D1,D2,...Dn
        di_num = length(di) # 计算|D_i|
        H_D_A += (di_num/sample_num) * empirical_entropy(di) # 公式5.8
    end
    return H_D_A
end

"""
   information_gain(D, feature, train_labels)

计算信息增益和信息增益比
...
# Arguments
- `D`: 训练数据集
- `feature`: 训练数据集某一特征向量
- `train_labels`: 训练标签集

# Return
- `ig`: 信息增益
- `igr`: 信息增益比
...

# Examples
```jldoctest
julia> information_gain(train_bina_data, 400, train_labels)
(infor_gain = 0.10731016009704736, infor_gain_ratio = 0.1326054196030962)
```
"""
function information_gain(D, feature, train_labels)
    ig = empirical_entropy(train_labels) - conditional_entropy(D, feature)
    igr = ig/empirical_entropy(D[:,feature])
    return (infor_gain = ig, infor_gain_ratio = igr)
end

# 树的结构参考博文：https://blog.csdn.net/wds2006sdo/article/details/52849400
mutable struct Tree
    node_type::String
    dict::Dict
    label#::Union{Missing, Int64}
    feature
    #Tree(node_type::String, dict::Dict, label, feature) = new(node_type, dict, label, feature)
    function Tree(;node_type::String, label = 1, feature = 1)
        node = new() 
        node.node_type = node_type
        node.dict = Dict()
        node.label = label
        node.feature = feature
        node
    end
end

# 给内节点添加树
function add_tree(tree::Tree, ind, sub_tree)
    tree.dict[ind] = sub_tree
end

# 给定实例进行预测
function predict(tree::Tree, features)
    if tree.node_type == "leaf_node"
        return tree.label
    end

    tree = tree.dict[features[tree.feature]]
    return predict(tree, features)
end


"""
   ID3_train(D, features, train_labels; ϵ = 0.1, method = "ig")

基于ID3算法进行模型训练
...
# Arguments
- `D`: 训练数据集
- `features`: 选取的特征向量集
- `train_labels`: 训练标签集
- `ϵ`: 阀值
- `method`: 默认为采用信息增益（"ig"），如果设置`method = "igr"`，则该用信息增益比

# Return
- `ig`: 信息增益
- `igr`: 信息增益比
...

# Examples
```jldoctest
julia> information_gain(train_bina_data, 400, train_labels)
(infor_gain = 0.10731016009704736, infor_gain_ratio = 0.1326054196030962)
```
"""
function ID3_train(D, features, labels; ϵ = 0.1, method = "ig") # features 取值1:size(D, 1)
    Leaf = "leaf_node" # 叶结点
    Internal = "internal" # 内结点

    class_ = unique(labels) # 输出的种类
    class_num_ = length(class_)

    if class_num_ == 1 # 对应算法5.2 （1）
        return Tree(node_type = Leaf, label = class_) #dict, , features
    end

    #----------------------------------------------
    tmp = findmax([(i, length(filter(x->x .== i, labels))) for i in class_])
    max_class, max_len = tmp[1][1], tmp[1][2]
    # Or
    # max_class = [sum(labels .== i) for i in class_] |> findmax |>x -> x[2] |> x-> class_[x]

    if length(features) == 0
        return Tree(node_type = Leaf, label = max_class)
    end

    # 针对最大增益小于ϵ阀值的时
    max_gain, ind = [information_gain(D, feature, labels).infor_gain for feature in features] |> findmax
    max_feature = features[ind]

    max_feature = 0
    max_gain = 0
    for feature in features
        gain = information_gain(D, feature, labels).infor_gain

        if gain > max_gain
            max_gain, max_feature = gain, feature
        end
    end

    if max_gain < ϵ # 对应算法5.2 （4）
        return Tree(node_type = Leaf, label = max_class)# dict, , features
    end

    D_feature = D[:, max_feature] # 选取此时训练数据集结点的最大信息增益特征向量列
    feature_val = unique(D_feature) # 该特征向量的取值
    feature_val_len = length(feature_val)

    # 构造树
    tree = Tree(node_type = Internal,  feature = max_feature)
    # 递归
    features_sub = setdiff(features, max_feature)

    @simd for i in feature_val
        ind = []
        for i in 1:length(labels)
            if D[i,:][max_feature] == i
                push!(ind, i)
            end
        end
        D_sub = D[ind, :] #根据最大信息增益特征将D划分为feature_val_len个子集， 即D1,D2,...Dn       feature_val[i]
        labels_sub = labels[ind]
        sub_tree = ID3_train(D_sub, features_sub, labels_sub)
        add_tree(tree, i, sub_tree)
    end
    return tree
end
tree = ID3_train(train_bina_data, 1:size(train_features,2) ,train_labels, ϵ=0.1)



function fit(testset, tree::Tree)
    label = []
    @simd for i = 1:size(testset,1)
        sample = testset[i,:]
        predict(tree, sample) |> x -> push!(label, x)
    end
    return label
end

# extract result and test_labels to `Array{Any,1}``, because the former two array are `Array{Union{Missing, Int64},1}``
result1 = []
for i in fit(test_bina_data, tree)
    push!(result1, i[1])
end

result2 = []
for i in test_labels
    push!(result2, i[1])
end
println("The precision of the model is ", sum(result1 .== result2) / length(result2))
