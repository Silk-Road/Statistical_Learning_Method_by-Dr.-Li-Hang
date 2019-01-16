# encoding = utf-8
# Author: Liu Jing
# Date: 2019-01-08
# Email: zhulincaowu@gmail.com
# Last modified by: Liu Jing
# Last modified time: 2019-01-09

# Julia Version 1.0.2

using Pkg,DataFrames,CSV,  NPZ, LinearAlgebra, Random

features = npzread("features.npy")
raw_data = CSV.read("Desktop/julia_temp/train_binary.csv")
labels = raw_data[1]; # if use `raw_data[:,1]` will copy the data
features = raw_data[2:end]; # # if use `raw_data[:,2:end]` will copy the data

"""
   train_test_split(data; train_per, rng)

将数据集分割为训练集和测试集
...
# Arguments
- `data`: the input data to split

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
function train_test_split(data; train_per = 0.67, rng = MersenneTwister(1234))
    n = size(data,1)
    ind = shuffle(rng, collect(1:n))
    train_ind = ind[1:floor(Int, n * train_per)]
    test_ind = ind[(floor(Int, n * train_per) + 1):n]
    data[train_ind, :], data[test_ind, :]
end

train_labels, test_labels = train_test_split(labels)
train_features, test_features = train_test_split(features)
train_labels = reshape(train_labels, (length(train_labels),))

"""
   Predict(train_set, test_set, train_labels; k)

基于《统计学习方法》第三章“k近邻法”中的算法3.1（k近邻法），因为k近邻法不具有显式的学习过程，只是利用训练数据集对特征向量空间进行划分
所以主程序只有`Predict`函数，而没有`train`函数进行训练
...
# Arguments
- `train_set::Array{T,N}`: 训练集特征向量
- `test_set::Array{T,N}`: 测试集特征向量
- `train_labels::Array{Union{Missing, Int},N}`: 训练集类别
- `k`: 代表最邻近k个实例
- `p`: 用于计算Minkowski距离
- `method`: 特征空间的距离度量方法，默认为Euclidian距离(p=2), 其它可供选择的距离有Minkowski, Manhattan(p=1), Lp(p=Inf)

# Return
- `predict`: 测试集中实例所属的类
...

# Examples
```jldoctest
julia> pred = Predict(train_features, test_features, train_labels)
```
"""
# Julia 1.0代码
@inline function Predict(train_set::Array{T,N}, test_set::Array{T,N}, train_labels::Vector{Union{Missing, Int}};
        k = 10, p = 3, method = "Euclidean") where {T,N}
    predict = [] # 初始化输出向量
    count = 0
    label_uni = unique(train_labels)

    @simd for i in 1:size(test_set, 1) # 遍历测试集
        test_vec = test_set[i,:]
        count += 1

        knn_list = [] # 用于存储k个最近邻的距离和其相应的类
        max_index = k # 初始化k个最近邻中最远点的索引
        max_dist = 0  # 初始化k个最近邻中最远点的距离

        # 用训练集的前k个实例初始化填充knn_list，knn_list为k个具名元组组成的一维向量，每个具名元组有距离和索引
        @simd for j = 1:k
            label = train_labels[j]
            train_vec = train_set[j, :]

            if method == "Minkowski"
                dist = Minkowski(train_vec, test_vec, p)
            elseif method == "Manhattan"
                dist = Manhattan(train_vec, test_vec)
            elseif method == "Lp"
                dist = Lp(train_vec, test_vec)
            else
                dist = Euclidean(train_vec, test_vec)
            end

            push!(knn_list, (distance = dist, label = label))
        end

        ind = 0 # 初始化该for循环中测试集实例对应的类

        # 因为之前初始化的knn_list仅仅是训练集前k个实例，并不是for循环中测试集中实例真正的k个最近邻，因此遍历训练集中
        # 剩下的实例，计算剩下的每个训练集实例同该测试集实例的距离
        @simd for u = (k+1):length(train_labels)
            label = train_labels[u]
            train_vec = train_set[u, :]

            dist = Euclidean(train_vec, test_vec) # 计算两个实例间的欧氏距离(Euclidian distance)

            # 在n-1循环加入新训练集实例并替换掉原来k最近邻的最远边界后，需要重新计算这10个最近邻最大边界所在，
            # 所以这段代码必须要在下面`if dist < max_dist`之前
            if max_index == k
                for m = 1:k
                    if max_dist < knn_list[m].distance
                        max_index = m
                        max_dist = knn_list[max_index].distance
                    end
                end
            end

            if dist < max_dist #如果新实例同测试集实例距离小于原先的最大距离，则替换之，这样不断缩小k最近邻的范围
                knn_list[max_index] = (distance = dist, label = label) #更新最大边界所在
                max_index = k # 在更新完k最近邻后，重新将最大距离实例索引归为默认值，即最大索引，以便计算下一个训练集实例
                max_dist = 0 # 在更新完k最近邻后，重新将最大距离归为默认值，即0，以便计算下一个训练集实例
            end
        end

        # 初始化统计投票的“票箱”
        class_total = length(label_uni)
        class_count = zeros(class_total) # 票箱

        # 统计投票
        @simd for i in knn_list
            label = i.label + 1 # 因为有些label的类是0，而Julia的起始索引同R一样是从1开始，不同于Python
            class_count[label] += 1
        end

        ind = findmax(class_count)[2] # 寻找票数最大的类，`findmax`函数返回向量的最大值和相应索引组成的元组
        push!(predict, ind-1)
    end
    return predict
end

function Euclidean(x::T, y::T) where T <: Array{Float32,1}
    x .- y |> x->x.^2 |> sum |> sqrt
end

function Minkowski(x::T, y::T, p) where T <: Array{Float32,1}
    x .- y |> x->abs.(x) |> x-> x.^p |> sum |> x-> x^(1/p)
end

function Manhattan(x::T, y::T) where T <: Array{Float32,1}
    x .- y |> x->abs.(x) |> sum
end

function Lp(x::T, y::T) where T <: Array{Float32,1}
    x .- y |> maximum
end

train_test = train_features[1:100,:]
test_test = test_features[1:100,:]
labels_test = train_labels[1:100]
pred = Predict(train_test, test_test, labels_test, method="Minkowski")
sum(pred .== test_labels[1:100])/ length(test_labels[1:100])

@time pred = Predict(train_features, test_features, train_labels, method = "Minkowski")
# 计算预测精度
p = sum(pred .== test_labels)/ length(test_labels)
