# encoding = utf-8
# Author: Liu Jing
# Date: 2019-01-10
# Email: zhulincaowu@gmail.com
# Last modified by: Liu Jing
# Last modified time: 2019-01-13

# Julia Version 1.0.2

using  DataFrames, CSV,   NPZ ,Pkg,LinearAlgebra,Random
features = npzread("./features.npy")
raw_data = CSV.read("./train_binary.csv")

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

# benchmark精度，这个MINIST数据失衡是很严重的，如果预测在这个精度附近，说明是没有多大作用的
sum(test_labels) / size(test_labels,1)


"""
   cal_dim(train_features, train_labels)

计算训练数据的一些维度
...
# Arguments
- `train_features`: 训练集特征向量
- `train_labels`: 训练集标签

# Return
- `class_num`: 标签类的取值
- `len_class_num`: 标签类有多少中取值
- `features_num`: 训练集的特征数
...

# Examples
```jldoctest
julia> class_num, len_class_num, features_num = cal_dim(train_features, train_labels)
(Union{Missing, Int64}[1, 0], 2, 28140, 324)
```
"""
function cal_dim(train_features, train_labels)
    class_num = unique(train_labels)
    len_class_num = length(class_num)
    features_num = size(train_features,2)
    return class_num, len_class_num, features_num
end
class_num, len_class_num, features_num = cal_dim(train_features, train_labels)

"""
   prior_prob(train_labels)

基于P51拉普拉斯平滑（Laplace smoothing,λ=1）贝叶斯估计计算先验概率
...
# Arguments
- `train_labels`: 训练集的类标签

# Return
- `prior_tuple`: 具名元组存储的先验概率
...

# Examples
```jldoctest
julia> prior_probability = prior_prob(train_labels)
(p0 = 0.09857153009736337, p1 = 0.9014284699026366)
```
"""

function prior_prob(train_labels)

    ## 初始化先验概率
    prior_probability = DataFrame(label = zeros(Int, len_class_num), probability = zeros(Float64, len_class_num))

    # 计算先验概率
    @simd for i in 1:len_class_num
        @inbounds prior_probability[i, 1] = class_num[i]
        @inbounds prior_probability[i, 2] = (train_labels .== class_num[i]) |> sum |> x -> x+1.0 |> x -> x/(length(train_labels) + len_class_num)
    end

    p1 = prior_probability[prior_probability.label .== 1, :].probability[1]
    p0 = prior_probability[prior_probability.label .== 0, :].probability[1]
    prior_tuple = (p0 = p0, p1 = p1)

    return prior_tuple # prior_probability 如果需要的话可以输出数据框结构的先验概率，这里输出的是具名元组
end

prior_probability = prior_prob(train_labels)

# 读取图片二值化后的数据
train_bina_data = npzread("train_bina_data.npy")
test_bina_data = npzread("test_bina_data.npy")

"""
   conditional_prob(train_bina_data)

基于P51拉普拉斯平滑（Laplace smoothing,λ=1）贝叶斯估计计算后验概率
...
# Arguments
- `train_bina_data`: 训练集，在此例中为二值化预处理后的训练集，这样可以确保每个特征向量有相同的2种取值，大大简化编程的难度

# Return
- `conditional_probability`: 后验概率
...

# Examples
```jldoctest
julia> conditional_probability = conditional_prob(train_bina_data)
2×324×2 Array{Float64,3}:
[:, :, 1] =
 0.00153846  0.00153846  0.00153846  …   1.54     1.57077   1.75231
 0.00153846  0.00153846  0.00153846     14.5538  14.8831   16.3862

[:, :, 2] =
  4.26769   4.26769   4.26769   4.26769  …   2.72923   2.69846   2.51692
 39.0277   39.0277   39.0277   39.0277      24.4754   24.1462   22.6431
```
"""
function conditional_prob(train_bina_data)
    # prior_probability = zeros(len_class_num) # 另一种计算先验概率的方法
    conditional_probability = zeros(len_class_num, features_num, 2)
    # 计算后验概率
    # 其计算有3个要素：类标签、具体哪个特征向量，该特征向量里哪个取值
    @simd for i in 1:size(train_bina_data,1)
        img = train_bina_data[i,:]
        label = train_labels[i] + 1 # +1是因为测试数据的类只有0和1，Julia从1开始索引

        #prior_probability[label] += 1

        for j in 1:features_num
            k = img[j] + UInt(1) # +UInt(1) 原因同上
            @inbounds conditional_probability[label, j, k] += 1
        end
    end

    @simd for i in 1:len_class_num
        len_tmp = length(conditional_probability[i,:,:]) # 计算P51（4.10）中的sum(I(y_i = c_k))，即该类所以实例数
        for j in 1:features_num
            # 二值化预处理后图像只有0，1两种取值
            # 具体详见JNingWei的博文https://blog.csdn.net/JNingWei/article/details/77747959
            xyac = conditional_probability[i,j,:] # 计算P51（4.10）中的sum(I(x_ij = a_ij,y_i = c_k))，即该类所以实例数
            len_tmp2 = length(xyac) # 计算P51（4.10）中的S_j，即给定类和特征向量后，该特征向量有多少种取值
            # 计算P51（4.10）条件概率
            @inbounds conditional_probability[i,j,:] = (xyac .+ 1)/ (len_tmp + len_tmp2 )
        end
    end

    return conditional_probability
end

conditional_probability = conditional_prob(train_bina_data)

"""
   predict(bina_img)

对给定的实例预测其类别，由于模型训练时（计算后验概率）为二值化预处理后的训练集，所以这里同样为二值化预处理后的实例
...
# Arguments
- `bina_img`: 测试或需预测的实例，

# Return
- 实例的类别标签
...

# Examples
```jldoctest
julia> predict(test_bina_data[1,:])
1
```
"""
function predict(bina_img)
    posterior_prob = []

    @simd for i = 1:len_class_num
        p = prior_probability[i]

        for j = 1:features_num
            k = bina_img[j] + UInt(1)
            p *= conditional_probability[i, j, k]
        end
        push!(posterior_prob, p)
    end
    return (findmax(posterior_prob)[2]-1) # 因为之前为了Julia索引，统一+1，这里最后再-1
end

# 模型精度
@time precise = sum([predict(test_bina_data[i,:]) for i in 1:size(test_bina_data,1)].==test_labels) /size(test_bina_data,1)
@show precise
