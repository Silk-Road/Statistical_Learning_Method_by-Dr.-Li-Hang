# encoding = utf-8
# Author: Liu Jing
# Date: 2019-01-21
# Email: zhulincaowu@gmail.com
# Last modified by: Liu Jing
# Last modified time: 2019-01-23

# Julia Version 1.0.2

# 可能部分Unicod在Github上无法显示

using RDatasets, LinearAlgebra, Random
# 调取R中iris数据集
iris = dataset("datasets", "iris") # 默认数据类型是DataFrame

iris = convert(Array,iris)
train_data = iris[1:100, 1:2]
tmp = iris[1:100, 5]
label = zeros(length(tmp))

# 为了和书中保持一致，将标签“setosa”转为1，”versicolor“转为-1
for (i, l) in enumerate(tmp)
    if l == "setosa"
        label[i] = 1
    else
        label[i] = -1 # "versicolor"
    end
end

```
using PyCall
Pkg.build("PyCall")

@pyimport matplotlib.pyplot as plt

plt.scatter(train_data[1:50,1], train_data[1:50,2], label = '0')
plt.scatter(train_data[51:100,1], train_data[51:100,2], label = '1')
plt.legend()
plt.show()
```

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
function train_test_split(data; train_per = 0.67, rng = MersenneTwister(1234))  #:: Array{T, N} where {T, N}
    n = size(data,1)
    ind = shuffle(rng, collect(1:n))
    train_ind = ind[1:floor(Int, n * train_per)]
    test_ind = ind[(floor(Int, n * train_per) + 1):n]
    data[train_ind, :], data[test_ind, :]
end

train_labels, test_labels = train_test_split(label)
train_features, test_features = train_test_split(train_data)

train_labels = reshape(train_labels, (length(train_labels),))

#-------------------------------------------------------

mutable struct SVM
    max_iter::Int
    kernel_name::String
    features::Array{Any, 2}
    labels::Array{Real, 1}
    m::Int
    n::Int
    b::Float64
    alpha::Array{Real, 1}
    C::Float64 # 松弛变量
    gamma::Float64 # 高斯核函数参数
    p::Int # 多项式核函数参数

    function SVM(;max_iter = 100, kernel_name = "linear", features = train_data,
         labels = labels, b = 0.0, C = 1.0, gamma = 0.1, p = 2)
        svm = new()
        svm.max_iter = max_iter
        svm.kernel_name = kernel_name
        svm.features = features
        svm.labels = labels
        svm.m, svm.n = size(features)
        svm.b = b
        svm.C = C
        svm.alpha = ones(svm.m)
        svm.gamma = gamma
        svm.p = p
        svm
    end
end

# 7.3.3 常用核函数
function kernel(svm::SVM, xᵢ, xⱼ)
    if svm.kernel_name == "linear"
        return dot(xᵢ, xⱼ) # 线性核函数
    elseif svm.kernel_name == "gaussian"
        l2 = (xᵢ .- xⱼ).^2 |> sum |> sqrt
        return exp(-l2/(2*svm.gamma^2)) # 高斯核函数
    elseif svm.kernel_name == "polynomial"
        return (dot(xᵢ, xⱼ) + 1)^svm.p # 多项式核函数
    elseif !(svm.kernel_name in ["linear", "gaussian", "polynomial"])
        throw("Your kernel function is not defined")
    end
end

# 公式（7.104) g(x)=∑αᵢyᵢK(xᵢ, x)+b
function g(svm::SVM, i)
    gₓ = svm.b
    @simd for j = 1:svm.m
        gₓ += svm.alpha[j] * svm.labels[j] * kernel(svm,svm.features[i,:], svm.features[j,:])
    end
    return  gₓ
end

# 公式（7.105）
function E_cal(svm::SVM, i)
    return g(svm, i) - svm.labels[i]
end

# 7.4.2 KKT条件 公式（7.111）～ （7.113）
function kkt_condition(svm::SVM, i)
    yg = g(svm, i) * svm.labels[i]
    if svm.alpha[i] == 0
        return yg  >=1
    elseif 0 < svm.alpha[i] < svm.C
        return yg == 1
    elseif svm.alpha[i] == svm.C
        return yg <= 1
    end
end

# L与H是α₂所在的对角线段端点的界, P127 公式（7.108）
function compare(_alpha, L, H)
    if _alpha > H
        return H
    elseif _alpha < L
        return L
    else
        return _alpha
    end
end

# 7.4.2 变量选择的方法
function var_select(svm::SVM)
    # 第一个变量选择，外层循环，在训练样本中选取违反KKT条件最严重的样本点，并将其对应的变量
    # 作为第1个变量
    satisfy_ind = [i for i in 1:svm.m if 0 < svm.alpha[i] < svm.C]
    nonsatisfy_ind = setdiff(1:svm.m, satisfy_ind)
    # Or
    # nonsatisfy_ind = [i for i in 1:m if ∋(statisfy_ind, i)]
    append!(satisfy_ind, nonsatisfy_ind)

    E = [E_cal(svm, i) for i in satisfy_ind]
    # 内层循环，在外层循环已经找到第1个变量α₁后，寻找使得α₂尽量大变化的第二个变量
    for i in satisfy_ind
        if kkt_condition(svm, i)
            continue
        end

        E1 = E_cal(svm, i)
        # 如果E₁是正的，选择最小的Eᵢ作为E₂;如果E₁是负的，那么选择最大的Eᵢ作为E₂
        #j = E1 >= 0 ? min(collect(1:svm.m)...) : max(collect(1:svm.m)...)
        j = E1 >= 0 ? argmin(E) : argmax(E)
        return i, j # 返回α1, α2的索引
    end
end

E = zeros(length(train_labels))

function fit(svm::SVM)
    for t in 1:svm.max_iter
        # train
        i1, i2 = var_select(svm)

        # 边界计算，P126
        if svm.labels[i1] == svm.labels[i2] # 图7.8右图
            L = max(0, svm.alpha[i2] + svm.alpha[i1] - svm.C)
            H = min(svm.C, svm.alpha[i2] + svm.alpha[i1])
        else # 图7.8左图
            L = max(0, svm.alpha[i2] - svm.alpha[i1])
            H = min(svm.C, svm.C + svm.alpha[i2] - svm.alpha[i1])
        end

        E1 = E_cal(svm, i1)
        E2 = E_cal(svm, i2)

        # 计算P127 （7.107): η = K₁₁ + K₂₂ - 2K₁₂
        η = kernel(svm, svm.features[i1,:], svm.features[i1,:]) +
                kernel(svm, svm.features[i2,:], svm.features[i2,:]) -
                2 * kernel(svm, svm.features[i1,:], svm.features[i2,:])
        if η <= 0
            continue
        end

        # P128 alpha_2^{new, unc}公式
        α2_new_unc = svm.alpha[i2] + svm.features[i2]*(E1 - E2)/η
        α2_new = compare(α2_new_unc, L, H)
        # P127 公式(109)
        α1_new = svm.alpha[i1] + svm.labels[i1] * svm.labels[i2] * (svm.alpha[i2] - α2_new)

        # P130 公式(7.115)
        b1_new = -E1 - svm.features[i1] *
            kernel(svm, svm.features[i1,:], svm.features[i1,:]) * (α1_new - svm.alpha[i1]) -
            svm.features[i2] * kernel(svm, svm.features[i2,:], svm.features[i1,:]) *
            (α2_new - svm.alpha[i2]) + svm.b
        # P130 公式(7.116)
        b2_new = -E2 - svm.features[i1] *
            kernel(svm, svm.features[i1,:], svm.features[i2,:]) * (α1_new - svm.alpha[i1]) -
            svm.features[i2] * kernel(svm, svm.features[i2,:], svm.features[i2,:]) *
            (α2_new - svm.alpha[i2]) + svm.b

        # P130 b_new更新方法
        if 0 < α1_new < svm.C && 0 < α2_new < svm.C
            b_new = b2_new
        elseif  α1_new == 0 || α1_new == svm.C || α2_new == 0 || α2_new == svm.C
        # b1_new和b2_new以及它们之间数符合KKT条件的阀值，这时选择它们的中点来更新
            b_new = (b1_new + b2_new) / 2
        end

        svm.alpha[i1] = α1_new
        svm.alpha[i2] = α2_new
        svm.b = b_new

        E[i1] = E1
        E[i2] = E2
    end
    return svm, E
end

function predict(svm::SVM, data)
    r = svm.b
    for i in 1:svm.m
        r += svm.alpha[i] * svm.labels[i] * kernel(svm, data, svm.features[i,:])
    end
    return r>0 ? 1 : -1
end

function _precision(svm, test_features, test_labels)
    count = 0
    for i in 1:size(test_features,1)
        result =  predict(svm, test_features[i, :])
        if result == test_labels[i]
            count += 1
        end
    end
    return count / size(test_features, 1)
end


linear_svm = SVM(features = train_features, labels = train_labels, kernel_name = "linear")
gaussian_svm = SVM(features = train_features, labels = train_labels, kernel_name = "gaussian")
polynomial_svm = SVM(features = train_features, labels = train_labels, kernel_name = "polynomial")

svm = [linear_svm, gaussian_svm, polynomial_svm]
[println("the $name kernel precision is: $ans") for (name, ans) in zip(["linear", "gaussian", "polynomial"],[(fit(s); _precision(s, test_features, test_labels)) for s in svm])]
