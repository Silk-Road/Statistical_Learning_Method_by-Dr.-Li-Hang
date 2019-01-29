# encoding = utf-8
# Author: Silk-Road
# Date: 2019-01-03
# Email: zhulincaowu@gmail.com
# Last modified by: Silk-Road
# Last modified time: 2019-01-04

# Julia Version 1.0.2

using  DataFrames, CSV,   NPZ ,Pkg,LinearAlgebra,Random
features = npzread("features.npy")
raw_data = CSV.read("./train_binary.csv")

labels = raw_data[1]; # if use `raw_data[:,1]` will copy the data
imgs = raw_data[2:end]; # # if use `raw_data[:,2:end]` will copy the data

"""
   train_test_split(data; train_per, rng)

Split the original dataset into trainset and testset based on specified percentage
...
# Arguments
- `data :: Array{T, N}`: the input data to split

# Return
- two data set, first is train set, the second is test set
...

# Examples
```jldoctest
julia> train_features, test_features = train_test_split(features) # split the features
(Float32[0.0 0.0 … 0.019473 0.0867527; ...])

julia> train_labels, test_labels = train_test_split(labels) # split the labels
(Union{Missing, Int64}[1; 0; … ; 1; 1], Union{Missing, Int64}[1; 1; … ; 1; 1])
```
"""
function train_test_split(data :: Array{T, N}; train_per = 0.7, rng = MersenneTwister(1234)) where {T, N}
    n = size(data,1)
    ind = shuffle(rng, collect(1:n))
    train_ind = ind[1:floor(Int, n * train_per)]
    test_ind = ind[(floor(Int, n * train_per) + 1):n]
    data[train_ind, :], data[test_ind, :]
end

train_labels, test_labels = train_test_split(labels)
train_features, test_features = train_test_split(features)

"""
   train(train_set, train_labels; learning_rate, learning_total, object_num)

Based on perceptron algorithm, use features and labels of train set to train the model
...
# Arguments
- `train_set::Array{T,N}`: features of train set
- `train_labels::Array{Union{Missing, Int},N}`: labels of train set
- `learning_rate`: the learning rate of stochastic gradient dedcent
- `learning_total`: the maximum number to learn
- `object_num`: the number to classify

# Return
- `w`: weight vector of perceptron model
- `b`: bias of perceptron model
...

# Examples
```jldoctest
julia> w, b = train(train_features, train_labels; object_num = 0)
([0.000247955, 0.00132109, ..., 0.000309731], -0.00030000000000000003)
```
"""
function train(train_set::Array{T,N}, train_labels::Array{Union{Missing, Int},N}; learning_rate = 0.0001, learning_total = 10000, object_num = 0) where {T,N}
    labels_capacity = size(train_labels,1)
    features_number = size(train_features,2)
    features_capacity = size(train_features, 1)
    labels_capacity == features_capacity || throw(DimensionMismatch("feature capacity should be same as labels'"))

    # initial w and b
    w = zeros(Float64,features_number)
    b = 0.0 # not "0" for type stability

    study_count = 0                 # where classification is wrong, this variable will +1
    nochange_count = 0              # cumsum of continuous correct classification, it will reset to 0 when one classification is wrong
    nochange_upper_limit = 100000   # bound of continuous correct classification, break training when `nochange_count` is larger than this

    while true
        nochange_count += 1
        if nochange_count > nochange_upper_limit
            break
        end

        # select one sample randomly
        ind = rand(1:labels_capacity)
        img = train_set[ind,:]
        label = train_labels[ind]

        # calculate the classification criterion, if it is larger than 0, it means the classification is correct, and there is no need to update w and b parameters
        yi = (label == object_num) ? -1 : 1
        criterion = yi * (dot(img, w) + b)
        if criterion <= 0
            w += learning_rate * yi * img
            b += learning_rate * yi

            study_count += 1
            if study_count > learning_total
                println("the total study number is larger than default")
                break
            end
            nochange_count = 0
        end
    end
    return w, b
end

"""
   predict(test_set, w, b)

Use the features of test data set to predict the labels based on the trained model,
then compare it with the true labels of test data set.
...
# Arguments
- `test_set::Array{T,N}`: features of test data set
- `w`: the output of `train` function
- `b`: the output of `train` function

# Return
- `predict`: the predicted labels
...

# Examples
```jldoctest
julia> pre_val = predict(test_features, w, b)
12601-element Array{Any,1}:
 1
 1
 .
 .
 .
 1
```
"""
function predict(test_set::Array{T,N}, w, b) where {T, N} # if there is missing in train or test dataset, set `test_test::Array{Union{Missing, T},N}`
    predict = []
    @simd for i = 1:size(test_set, 1)
        @inbounds img = test_features[i, :]
        result = dot(img, w) + b
        result = result > 0 ? 1 : 0
        append!(predict, result)
    end
    return predict
end

@time map([0]) do x
           w, b = train(train_features, train_labels; object_num = x)
           pre_val = predict(test_features, w, b);
           _precision = sum(pre_val .== test_labels) / size(test_labels,1)
           println("the precision with respect to $x is ", _precision)
        end
