using Plots

fn = ["O2_Rayleigh.png", "O2_VRS.png", "O2_RRSRVRS.png", "N2_Rayleigh.png", "N2_VRS.png", "N2_RRSRVRS.png"]

γ_O₂_Rayl = 0.02885
γ_O₂_VRS  = 0.16951
γ_N₂_Rayl = 0.01058
γ_N₂_VRS  = 0.08497
γ_RRS     = 0.75
γ = [γ_O₂_Rayl, γ_O₂_VRS, γ_RRS, γ_N₂_Rayl, γ_N₂_VRS, γ_RRS]
vlabels = ["O₂ Rayleigh: f₁₁" "O₂ Rayleigh: f₂₂" "O₂ Rayleigh: f₃₃" "O₂ Rayleigh: f₄₄" "O₂ Rayleigh: f₁₂";
           "O₂ VRS: f₁₁" "O₂ VRS: f₂₂" "O₂ VRS: f₃₃" "O₂ VRS: f₄₄" "O₂ VRS: f₁₂";
           "O₂ RRS/RVRS: f₁₁" "O₂ RRS/RVRS: f₂₂" "O₂ RRS/RVRS: f₃₃" "O₂ RRS/RVRS: f₄₄" "O₂ RRS/RVRS: f₁₂";
           "N₂ Rayleigh: f₁₁" "N₂ Rayleigh: f₂₂" "N₂ Rayleigh: f₃₃" "N₂ Rayleigh: f₄₄" "N₂ Rayleigh: f₁₂";
           "N₂ VRS: f₁₁" "N₂ VRS: f₂₂" "N₂ VRS: f₃₃" "N₂ VRS: f₄₄" "N₂ VRS: f₁₂";
           "N₂ RRS/RVRS: f₁₁" "N₂ RRS/RVRS: f₂₂" "N₂ RRS/RVRS: f₃₃" "N₂ RRS/RVRS: f₄₄" "N₂ RRS/RVRS: f₁₂"]
vls = [:solid, :dash, :dashdot, :dashdotdot, :dot]
vcolor = [:purple, :red, :blue, :purple, :red, :blue]
function compute_f(θ, γ)
    f₁₁ = (1/8π)*(3/(1+2γ))*((cosd.(θ)).^2 .+ (sind.(θ)).^2*γ)
    f₂₂ = (1/8π)*(3/(1+2γ))*θ./θ
    f₃₃ = (1/8π)*(3/(1+2γ))*(1-γ)*cosd.(θ)
    f₄₄ = (1/8π)*(3/(1+2γ))*(1-3γ)*cosd.(θ)
    f₁₂ = (1/8π)*(3γ/(1+2γ))*θ./θ

    return f₁₁, f₂₂, f₃₃, f₄₄, f₁₂
end

θ = 0:0.5:360.
d2r = π/180.
for i = 1:6
    ff₁₁, ff₂₂, ff₃₃, ff₄₄, ff₁₂ = compute_f(θ, γ[i])
    ff = [ff₁₁ ff₂₂ ff₃₃ ff₄₄ ff₁₂]
    plt = plot(θ*d2r, (0.5/π)*(θ./θ), seriestype = :steppre,
           linestyle = :solid,
           linealpha = 0.5,
           linewidth = 1,
           linecolor = :black, label = "", proj=:polar)
    plot!(θ*d2r, (0.25/π)*(θ./θ), seriestype = :steppre,
           linestyle = :solid,
           linealpha = 0.5,
           linewidth = 1,
           linecolor = :black, label = "", proj=:polar)
    for j = 1:5
        plot!(θ*d2r, ff[:,j], seriestype = :steppre,
            linestyle = vls[j],
            linealpha = 0.5,
            linewidth = 2,
            linecolor = vcolor[i], label = vlabels[i,j], proj=:polar)
        if j==3 || j==4
            plot!(θ*d2r, -ff[:,j], seriestype = :steppre,
                linestyle = vls[j],
                linealpha = 0.5,
                linewidth = 2,
                linecolor = :green, label = "", proj=:polar)
        end
    end
    display(plt)
    println(fn[i])
    savefig(plt, fn[i])
end