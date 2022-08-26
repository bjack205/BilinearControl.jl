
function train_model(datafile, outfile; 
        epochs=100, alpha=0.5, hidden=32, use_relu=false, verbose=false)
    mainfile = joinpath(@__DIR__, "main.py")
    outbase,ext = splitext(outfile)
    outfile_jac = outbase * "_jacobian" * ext
    relu = use_relu ? "--relu" : `` 
    verbose = verbose ? "--verbose" : `` 
    cmd = `python3 $mainfile $datafile -o $outfile --epochs $epochs --alpha $alpha --hidden $hidden $relu $verbose`
    cmd_jac = `python3 $mainfile $datafile -o $outfile_jac --jacobian --epochs $epochs --alpha $alpha --hidden $hidden $relu $verbose`
    run(cmd)
    run(cmd_jac)
end
