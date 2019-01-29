# encoding = utf-8
# Author: Silk-Road
# Date: 2019-01-27
# Email: zhulincaowu@gmail.com
# Last modified by: Silk-Road
# Last modified time: 2019-01-29

# Julia Version 1.0.2
using Random, Distributions, LinearAlgebra

#-----------------------------------

mutable struct HMM
    N::Int # 状态数
    M::Int # 观测数 length(unique(O))
    A::Matrix{Float64} # 状态转移概率矩阵 size:[N, N]
    B::NamedTuple # 观测概率具名元组，每一个元素对应书中B的一列 size:[N, M]
    Pi::Vector{Float64} # 初始状态概率向量 length:[N]
    O::Union{Vector{String}, Nothing} # 观测序列 length:[T]
    T::Int
    Alpha::Matrix{Float64} # size:[T, N]
    Beta::Matrix{Float64} # size: [T, N]
    Pos::Float64 # 观测序列概率，即P(O|λ)
    function HMM(n,o,a,b,p_i)#, a = , b = zeros(Float64, n, m), p_i =rand(Float64, n))
        hmm = new()
        hmm.N = n
        hmm.O = o
        hmm.M = length(unique(o))
        hmm.A = a #zeros(Float64, n, n)
        hmm.B = b #nothing #zeros(Float64, n, length(unique(o)))
        hmm.Pi = p_i#rand(Float64, n)
        hmm.T = length(o)
        hmm.Alpha = zeros(Float64, length(o),n)
        hmm.Beta = zeros(Float64, length(o),n)
        hmm.Pos = 0.0 #missing
        hmm
    end
end

#-------------------------------------------
#               §10.2 概率计算算法
#-------------------------------------------
"""
   forward_probability(hmm::HMM)

基于前向算法计算观测序列概率P(O|λ)
...
# Arguments
- `hmm :: HMM`: 隐马尔可夫实例

# Return
- `hmm.Pos` : 观测序列概率P(O|λ)
...

# Examples
```jldoctest
julia> forward_probability(hmm).Pos
0.129318
```
"""
function forward_probability(hmm::HMM)
    # 初值, 公式(10.15)
    ind_1 = hmm.O[1]
    hmm.Alpha[1,:] = hmm.Pi .* hmm.B[Meta.parse(ind_1)]
    # 递推，公式（10.16）
    @simd for t in 2:hmm.T
        ind = hmm.O[t] # 该观测实例将作为索引，用于定位B中的元组
        @simd for i in 1:hmm.N
            @inbounds hmm.Alpha[t, i] = dot(hmm.Alpha[t-1, :], hmm.A[i,:]) * hmm.B[Meta.parse(ind)][i]
        end
    end
    # 终止，公式（10.17）
    hmm.Pos = sum(hmm.Alpha[end,:])
    return hmm
end

"""
   backward_probability(hmm::HMM)

基于后向算法计算观测序列概率P(O|λ)
...
# Arguments
- `hmm :: HMM`: 隐马尔可夫实例

# Return
- `hmm.Pos` : 观测序列概率P(O|λ)
...

# Examples
```jldoctest
julia> backward_probability(hmm).Pos
0.133758
```
"""
function backward_probability(hmm::HMM)
    # 初值 公式（10.19）
    hmm.Beta[end,:] = ones(Float64, hmm.N)
    # 递推 公式（10.20）
    @simd for t in (hmm.T-1) : -1 : 1
        ind = hmm.O[t]
        @simd for i in 1:hmm.N
            @inbounds hmm.Beta[t, i] = [hmm.A[i, j] * hmm.B[Meta.parse(ind)][j] * hmm.Beta[t+1,j] for j in 1:hmm.N] |> sum
        end
    end
    hmm.Pos = [hmm.Pi[i] * hmm.B[Meta.parse(hmm.O[1])][i] * hmm.Beta[1,i] for i in 1:hmm.N] |> sum
    return hmm
end


# 10.2节书中例题
o = ["红", "白", "红"]
a = [0.5, 0.3, 0.2,
     0.2, 0.5, 0.3,
     0.3,0.2, 0.5]

a = reshape(a, (3,3))
b = (红 = [0.5,0.4,0.7], 白 = [0.5,0.6,0.3])
p_i = [0.2,0.4,0.4]
hmm = HMM(3, o, a, b, p_i)

forward_probability(hmm).Pos
backward_probability(hmm).Pos


#-------------------------------------------
#                §10.3 学习算法
#-------------------------------------------
# 公式（10.24）
# 使用多重分派计算gamma，返回整个gamma矩阵或单个gamma值
function gamma(hmm::HMM)
    gamma = zeros(Float64, hmm.T, hmm.N)
    alpha = forward_probability(hmm).Alpha
    beta = backward_probability(hmm).Beta
    @simd for t in 1:hmm.T
        @simd for i in 1:hmm.N
            @inbounds gamma[t,i] = alpha[t,i]*beta[t,i] / sum([alpha[t,j] * beta[t,j] for j in 1:hmm.N])
        end
    end
    return gamma
end

function gamma(hmm::HMM, t, i)
    alpha = forward_probability(hmm).Alpha
    beta = backward_probability(hmm).Beta
    numer = alpha[t, i] * beta[t, i]
    denom = [alpha[t, j] * beta[t, j] for j in 1:hmm.N] |> sum
    return numer/denom
end

# 公式（10.26）
function xi(hmm, t, i, j)
    alpha = forward_probability(hmm).Alpha
    beta = backward_probability(hmm).Beta
    numer = alpha[t, i] * hmm.A[i,j] * hmm.B[Meta.parse(hmm.O[t+1])][j] * beta[t+1, j]
    denom = 0.0
    @simd for i in 1:hmm.N
        @simd for j in 1:hmm.N
            @inbounds denom += alpha[t, i] * hmm.A[i,j] * hmm.B[Meta.parse(hmm.O[t+1])][j] * beta[t+1, j]
        end
    end
    return numer/denom
end

# EM算法初始化
function hmm_init(n,observations; rng = MersenneTwister(1234)) # e.g. observations = ["红", "白", "红"]
    a = rand(rng, 1:10, n, n) |> x-> x./sum(x,dims=2)
    uni_obser = unique(observations)
    m = length(uni_obser)
    b_tmp = rand(rng, 1:10, n, m) |> x-> x./sum(x,dims=2)
    t = tuple(Meta.parse.(unique(observations))...) # e.g. (:红, :白)
    b = NamedTuple{t, Tuple{Array{Float64,1},Array{Float64,1}}} # e.g. NamedTuple{(:红, :白),Tuple{Array{Float64,1},Array{Float64,1}}}
    b = b((b_tmp[:,1], b_tmp[:,2])) # 类型b的构造函数，产生具名数组. 注意根据类型b的定义，在构造时里面是元组，
                                    # 在初始化时这个代码还需要手工，目前还没有找到好的方法
    p_i = rand(rng, 1:10, n) |> x-> x./sum(x)
    hmm = HMM(n, observations, a, b, p_i)
    return hmm
end
# test:
# hmm = hmm_init(3,  ["红", "白", "红"])

"""
   baum_welch(hmm::HMM)

基于非监督学习算法Baum-Welch算法（EM算法）估计模型λ(A,B,π)参数
...
# Arguments
- `hmm :: HMM`: 隐马尔可夫实例

# Return
- `A, B, π` : 模型参数
...
"""
# 使用EM算法估计参数
function baum_welch(hmm::HMM)
    a, b, p_i = hmm.A, hmm.B, hmm.Pi
    for i in 1:hmm.N
        for j in 1:hmm.N
            a[i,j] = [xi(hmm, t, i, j) for t in 1: (hmm.T-1)] |> sum # 公式（10.39）
        end
        p_i[i] = gamma(hmm,1,i) # 公式（10.41）
    end

    for j in 1:hmm.N
        for k in 1:hmm.M
            ind = [t for t in 1:hmm.T if hmm.O[t] == unique(hmm.O)[k]] # hmm.B[Meta.parse(hmm.O[t])]
            numer = [gamma(hmm, t, j) for t in ind] |> sum
            denom = [gamma(hmm, t, j) for t in 1:hmm.T] |> sum
            b[k][j] =  numer/denom # 公式（10.40）
        end
    end
    return a, b, p_i
end

# 递推
function iter(n, observations; maxstep = 100, ϵ = 0.001)
    hmm = hmm_init(n, observations)# e.g. hmm_init(3,  ["红", "白", "红"])
    old_a, old_b, old_p_i = hmm.A, hmm.B, hmm.Pi
    step = 0
    done = false
    while !done
            step +=1
            a, b, p_i = baum_welch(hmm)
            hmm = HMM(n,observations, a, b, p_i)
            done = (step > maxstep) || (abs.(a .- old_a) |> sum) < ϵ ||
            (abs.(b .- old_b) |> sum) < ϵ || (abs.(p_i .- old_p_i) |> sum) < ϵ
    end
    return hmm
end

#-------------------------------------------
#               §10.4 预测算法
#-------------------------------------------
"""
   viterbi(hmm::HMM)

基于维特比算法求最优状态序列
...
# Arguments
- `hmm :: HMM`: 隐马尔可夫实例

# Return
- `I` : 最优状态序列
...

# Examples
```jldoctest
julia> viterbi(hmm)
最有状态序列I为 [3, 3, 3]
最优路径的概率P⃰为 0.014699999999999998
δ为 [0.1 0.16 0.28; 0.028 0.0504 0.042; 0.00756 0.01008 0.0147]
ψ为 [0.0 0.0 0.0; 3.0 3.0 3.0; 2.0 2.0 3.0]
```
"""
function viterbi(hmm::HMM)
    # 初始化
    δ = zeros(Float64, hmm.T, hmm.N)
    ψ = zeros(Float64, hmm.T, hmm.N)
    ind_1 = hmm.O[1]
    I = zeros(Int, hmm.N)
    for i in 1:hmm.N
        δ[1,i] = hmm.Pi[i] * hmm.B[Meta.parse(ind_1)][i]
    end

    # 递推
    for t in 2:hmm.T
        ind = hmm.O[t]
        for i in 1:hmm.N
            delta_a = [δ[t-1, j] * hmm.A[j, i] for j in 1:hmm.N]
            tmp = findmax(delta_a)
            δ[t,i] = tmp[1] * hmm.B[Meta.parse(ind)][i]
            ψ[t,i] = tmp[2]
        end
    end

    # 最优路径回溯
    tmp = findmax(δ[end,:])
    P⃰ = tmp[1]
    I[end] = tmp[2]
    for i in (hmm.N-1):-1:1
        I[i] = ψ[i+1, i+1]
    end

    println("最优状态序列I为 $I")
    println("最优路径的概率P⃰为 $P⃰")
    println("δ为 $δ")
    println("ψ为 $ψ")
    return I
end

# 书中例10.3
o = ["红", "白", "红"]
a = [0.5, 0.3, 0.2,
     0.2, 0.5, 0.3,
     0.3,0.2, 0.5]

a = reshape(a, (3,3))
b = (红 = [0.5,0.4,0.7], 白 = [0.5,0.6,0.3])
p_i = [0.2,0.4,0.4]

hmm = HMM(3, o, a, b, p_i) # n = 3 是状态数
viterbi(hmm)
