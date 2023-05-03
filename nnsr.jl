using MLDataUtils, CSV, DataFrames, CategoricalArrays, Random, Plots, JLD, TransferEntropy, BSON
using IterTools, Statistics, StatsBase, Flux, Evolutionary, Distributions, SymbolicRegression

function preprocess(data)
    trainingData, testData = splitobs(data; at = 0.7)
    trainX, trainY, testX, testY = trainingData[!, 1:end-1], trainingData[!,end], testData[!, 1:end-1], testData[!, end]
    trainX = Matrix{Float32}(trainX)
    testX  = Matrix{Float32}(testX)
    
    trainData = []
    for i in range(start=1, stop=size(trainX, 1), step=1)
        push!(trainData, (trainX[i, :], [trainY[i]]))
    end
    testData = []
    for i in range(start=1, stop=size(testX, 1), step=1)
        push!(testData, (testX[i, :], [testY[i]]))
    end
    
    return trainData, testData, trainX, trainY, testX, testY
end

function backprops(trainData::Vector{Any}, testData::Vector{Any}, trainX::Matrix{Float32}, trainY::Any, testX::Matrix{Float32}, testY::Any)
    features = size(trainData[1][1], 1)
    num_hiddens = [1, 1, 2, 2, 3]
    fdict = Dict(1 => ([[4], [5], [6, 3], [8, 5], [6, 4, 2]]),
                 2 => ([[4], [6], [10, 6], [10, 4], [8, 5, 2]]),
                 3 => ([[6], [5], [10, 6], [10, 4], [8, 5, 2]]),
                 4 => ([[8], [10], [12, 6], [14, 7], [10, 7, 3]]),
                 5 => ([[10], [12], [14, 7], [14, 3], [10, 4, 2]]),
                 6 => ([[10], [12], [14, 7], [14, 3], [10, 4, 2]]),
                 7 => ([[12], [14], [14, 7], [14, 3], [12, 4, 2]]),
                 8 => ([[14], [16], [16, 4], [18, 5], [14, 4, 2]]),
                 9 => ([[16], [18], [20, 3], [20, 7], [16, 5, 2]])
                 )
    train_losses = []
    test_losses = []
    models = []
    
    hidden_neurons_arr = fdict[features]
    
    for i in 1:5
        println("Training model " * string(i))
        num_hidden = num_hiddens[i]
        hidden_neurons = hidden_neurons_arr[i]
        layers = [Dense(features, hidden_neurons[1], Flux.relu)]
        for i in 1:num_hidden
            if i == num_hidden
                push!(layers, Dense(hidden_neurons[i], 1, Flux.relu))
            else
                push!(layers, Dense(hidden_neurons[i], hidden_neurons[i+1], Flux.relu))
            end
        end
        model(xx) = foldl((xx, m) -> m(xx), layers, init = xx)
        loss(x, y) = sqrt(Flux.mse(model(x), y))
        optimizer = ADAM()
        for epoch in 1:500
            Flux.train!(loss, Flux.params(model), trainData, optimizer)
        end
        push!(train_losses, loss(transpose(trainX), reshape(trainY, (1, size(trainY, 1)))))
        push!(test_losses, loss(transpose(testX), reshape(testY, (1, size(testY, 1)))))
        push!(models, model)
    end
    return train_losses, test_losses, models
end

function symreg(X, y)
    options = SymbolicRegression.Options(
        binary_operators=(+, *, /, -),
        unary_operators=(exp, safe_sqrt, square, sin, cos),
        batching=true,
        npop=40,
        ncycles_per_iteration=500,
        enable_autodiff=true,
        skip_mutation_failures=false,
        progress=false

    )
    hallOfFame = EquationSearch(X, y, niterations=50, options=options)
    dominating = calculate_pareto_frontier(X, y, hallOfFame, options)
    best_score_eq, all_eqs, best_loss_eq = get_best(X, y; hof=hallOfFame, options)
    return best_score_eq, all_eqs, best_loss_eq
end

function get_best(X, y; hof::HallOfFame{T,L}, options) where {T,L}
    dominating = calculate_pareto_frontier(X, y, hof, options)

    df = DataFrame(;
        tree=[m.tree for m in dominating],
        loss=[m.loss for m in dominating],
        complexity=[compute_complexity(m.tree, options) for m in dominating],
    )

    df[!, :score] = vcat(
        [L(0.0)],
        -1 .* log.(df.loss[2:end] ./ df.loss[1:(end - 1)]) ./
        (df.complexity[2:end] .- df.complexity[1:(end - 1)]),
    )

    min_loss = min(df.loss...)

    best_idx = argmax(df.score .* (df.loss .<= (2 * min_loss)))

    return df.tree[best_idx], df, df.tree[argmin(df.loss)]
end

function correlation(X::Vector{T}, Y::Vector{T}) where T<:Real
    n = size(X, 1)

    mean_X = mean(X)
    mean_Y = mean(Y)

    std_X = std(X)
    std_Y = std(Y)

    cov_XY = sum((X .- mean_X) .* (Y .- mean_Y)) / (n - 1)

    corr_XY = cov_XY / (std_X * std_Y)
    if corr_XY == NaN 
        return 0
    end
    return corr_XY
end

files = readdir(pwd()*"/Feynman_with_units")
files = shuffle(files)
#println(files)
for file in files
    println(file)
    if file in readdir("searchresults/")
        continue
    end
    data = CSV.read(pwd()*"/Feynman_with_units/"*file, DataFrame, delim=" ")
    data = data[1:10000, 1:end-1]
    
    mkdir("searchresults/"*file)    
    
    trainData, testData, trainX, trainY, testX, testY = preprocess(data)
    datadict = Dict("trainData" => trainData, "testData" => testData, "trainX" => trainX, "trainY" => trainY, "testX" => testX, "testY" => testY)
    save("searchresults/"*file*"/data.jld", datadict)
    
    train_losses, test_losses, models = backprops(trainData, testData, trainX, trainY, testX, testY)
    modeldict = Dict("train_losses" => train_losses, "test_losses" => test_losses) 
#                        "model_1" => Flux.params(models[1]),
#                        "model_2" => Flux.params(models[2]),
#                        "model_3" => Flux.params(models[3]),
#                        "model_4" => Flux.params(models[4]),
#                        "model_5" => Flux.params(models[5])
#                        "model_6" => Flux.params(models[6]),
#                        "model_7" => Flux.params(models[7]),
#                        "model_8" => Flux.params(models[8]),
#                        "model_9" => Flux.params(models[9]),
#                        "model_10" => Flux.params(models[10]))

    save("searchresults/"*file*"/nn.jld", modeldict)
#    @save "searchresults/"*file*"/model_1.bson" models[1]
#    @save "searchresults/"*file*"/model_2.bson" models[2]
#    @save "searchresults/"*file*"/model_3.bson" models[3]
#    @save "searchresults/"*file*"/model_4.bson" models[4]
#    @save "searchresults/"*file*"/model_5.bson" models[5]

    X = Matrix{Float64}(transpose(vcat(trainX, testX)))
    y = vcat(trainY, testY)
    options = SymbolicRegression.Options(
        binary_operators=(+, *, /, -),
        unary_operators=(exp, safe_sqrt, square, sin, cos),
        batching=true,
        npop=40,
        ncycles_per_iteration=500,
        enable_autodiff=true,
        skip_mutation_failures=false,
        progress=false
    )
    cd("searchresults/"*file) do
        best_score_eq_sr, all_eqs_sr, best_loss_eq_sr = symreg(X, y)
        srdict = Dict("best_score_eq_sr"=>best_score_eq_sr, "all_eqs_sr"=>all_eqs_sr, "best_loss_eq_sr"=>best_loss_eq_sr)
        save("sr.jld", srdict)
        for i in 1:size(models, 1)
            println("NNSR: "*string(i))
            best_score_eq_nnsr, all_eqs_nnsr, best_loss_eq_nnsr = symreg(X, Vector{Float64}(reshape(models[i](X), (10000,))))
            nnsrdict = Dict("best_score_eq_nnsr"=>best_score_eq_nnsr, "all_eqs_nnsr"=>all_eqs_nnsr, "best_loss_eq_nnsr"=>best_loss_eq_nnsr)
            save("nnsr_model_"*string(i)*".jld", nnsrdict)
            
            b = RectangularBinning(1)
            est = VisitationFrequency(b)
            
            x_labels = [string(j) for j in 1:size(X, 1)]
            y_values = [correlation(y, reshape(X[i, :], (10000,))) for i in 1:size(X, 1)]
            p1 = plot(x_labels, y_values, legend=false, xlabel="Features", ylabel="Correlation", title="Truth")
            y_values = [correlation(eval_tree_array(best_score_eq_sr, X, options)[1], reshape(X[i, :], (10000,))) for i in 1:size(X, 1)]
            p2 = plot(x_labels, y_values, legend=false, xlabel="Features", ylabel="Correlation", title="Symbolic Regressor")
            y_values = [correlation(Vector{Float64}(reshape(models[i](X), (10000,))), reshape(X[it, :], (10000,))) for it in 1:size(X, 1)]
            p3 = plot(x_labels, y_values, legend=false, xlabel="Features", ylabel="Correlation", title="NN")
            a1 = Vector{Float64}(reshape(eval_tree_array(best_score_eq_nnsr, X, options)[1], (10000,)))
            a2 = [reshape(X[it, :], (10000,)) for it in 1:size(X, 1)]
            y_values = [correlation(a1, k) for k in a2]
            p4 = plot(x_labels, y_values, legend=false, xlabel="Features", ylabel="Correlation", title="NN+SR")
            plot(p1, p2, p3, p4, layout=(2,2))
            savefig("model_"*string(i)*"_corr.png")
            
            try
                x_labels = [string(j) for j in 1:size(X, 1)]
                y_values = [mutualinfo(y, reshape(X[j, :], (10000,)), est::VisitationFrequency{RectangularBinning{Int64}}; base = 2, q = 1) for j in 1:size(X, 1)]
                p1 = plot(x_labels, y_values, legend=false, xlabel="Features", ylabel="Mutual Info.", title="Truth")
                y_values = [mutualinfo(eval_tree_array(best_score_eq_sr, X, options)[1], reshape(X[it, :], (10000,)), est::VisitationFrequency{RectangularBinning{Int64}}; base = 2, q = 1) for it in 1:size(X, 1)]
                p2 = plot(x_labels, y_values, legend=false, xlabel="Features", ylabel="Mutual Info.", title="Symbolic Regressor")
                minny = Vector{Float64}(reshape(models[i](X), (10000,)))
                minny = replace(minny, NaN => 0.0)
                minny2 = [reshape(X[it, :], (10000,)) for it in 1:size(X, 1)]
                y_values = [mutualinfo(minny, k, est::VisitationFrequency{RectangularBinning{Int64}}; base = 2, q = 1) for k in minny2]
                p3 = plot(x_labels, y_values, legend=false, xlabel="Features", ylabel="Mutual Info.", title="NN")
                p4a1 = Vector{Float64}(replace(reshape(eval_tree_array(best_score_eq_nnsr, X, options)[1], (10000,)), NaN => 0.0))
                y_values = [mutualinfo(p4a1, reshape(X[it, :], (10000,)), est::VisitationFrequency{RectangularBinning{Int64}}; base = 2, q = 1) for it in 1:size(X, 1)]
                p4 = plot(x_labels, y_values, legend=false, xlabel="Features", ylabel="Mutual Info.", title="NN+SR")
                plot(p1, p2, p3, p4, layout=(2,2))
                savefig("model_"*string(i)*"_mi.png")
            catch ex
                if ex isa InexactError
                    continue
                else
                    rethrow(ex)
                end
            end
        end
    end
    
end
