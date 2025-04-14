using Libxc 
using Statistics
using Base.Threads

struct Connector end 

(connector::Connector)(basis)  = TermConnector()

struct TermConnector <: Term
end 

# HEG funtions from Libxc: Hexc , Hvxc, Hfxc 
function get_Hexc_Hvxc_Hfxc(ρ::Array{Float64})
    lda_x= Libxc.Functional(:lda_x)
    res_lda_x = Libxc.evaluate(lda_x, rho=reshape(ρ, 1,size(ρ,1),size(ρ,2),size(ρ,3)), derivatives=[0,1,2]) 

    lda_c= Libxc.Functional(:lda_c_pz)
    res_lda_c = Libxc.evaluate(lda_c, rho=reshape(ρ, 1,size(ρ,1),size(ρ,2),size(ρ,3)), derivatives=[0,1,2]) 

    Hexc= res_lda_x.zk + res_lda_c.zk
    Hvxc=  res_lda_x.vrho + res_lda_c.vrho 
    Hfxc = res_lda_x.v2rho2 + res_lda_c.v2rho2  
    Hexc , Hvxc[1, :, :, :], Hfxc[1, :, :, :]
end

function get_Hexc_Hvxc_Hfxc(ρ::Float64)
    lda_x= Libxc.Functional(:lda_x)
    res_lda_x = Libxc.evaluate(lda_x, rho=[ρ], derivatives=[0,1,2]) 

    lda_c= Libxc.Functional(:lda_c_pz)
    res_lda_c = Libxc.evaluate(lda_c, rho=[ρ], derivatives=[0,1,2]) 

    Hexc= res_lda_x.zk + res_lda_c.zk
    Hvxc=  res_lda_x.vrho + res_lda_c.vrho 
    Hfxc = res_lda_x.v2rho2 + res_lda_c.v2rho2  

    Hexc[1], Hvxc[1], Hfxc[1]
end

# Corradini fxc -----------------------------------------------------  
function diffvc(n)  # like that it doesn't support multidimensional Array 
    lda_c= Libxc.Functional(:lda_c_pz)
    res_lda_c = Libxc.evaluate(lda_c, rho=[n], derivatives=[0,1,2])
    return res_lda_c.v2rho2[1] 
end

function diffv_cep(n)
    rs = (3.0 / (4.0 * π * n))^(1.0 / 3.0)
    diff_n_rs = - 3 * (4.0 * π  / 3.)^(1.0 / 3.0) * n^(4.0 / 3.0)
    lda_c= Libxc.Functional(:lda_c_pz)
    res_lda_c = Libxc.evaluate(lda_c, rho=[n], derivatives=[0,1])
    ec = res_lda_c.zk[1]
    diff_ec_n = (res_lda_c.vrho[1] - ec) / n  
    diff_rsec_rs = ec + rs * diff_ec_n * diff_n_rs   
end


function fxcr_smooth_pz(n, r)
    third = 1.0 / 3.0
    rs = (3.0 / (4.0 * π * n))^third
    k_F = (3.0 * π^2 * n)^(1.0 / 3.0)
    e = 1.0  # atomic units
    diff_mu = diffvc(n)
    A = 1.0 / 4.0 - (k_F^2) / (4.0 * π * e^2) * diff_mu
    diff_rse = diffv_cep(n)
    C = π / (2.0 * e^2 * k_F) * (-diff_rse)
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    x = rs^(1.0 / 2.0)
    B = (1.0 + a1 * x + a2 * x^3) / (3.0 + b1 * x + b2 * x^3)
    g = B / (A - C)
    alpha = 1.5 / (rs^(1.0 / 4.0)) * A / (B * g)
    beta = 1.2 / (B * g)
    fxcr = alpha * k_F * π^1.5 / (4.0 * π^2 * beta^2.5) * (k_F^2 .* r.^2 / (2 * beta) .- 3) .* exp.(-k_F^2 .* r.^2 ./ (4 * beta))
    return fxcr
end

function fxcr_residual_pz(n, r, integral::Bool=false)
    third = 1.0 / 3.0
    rs = (3.0 / (4.0 * π * n))^third
    k_F = (3.0 * π^2 * n)^(1.0 / 3.0)
    e = 1.0  # atomic units
    diff_mu =  diffvc(n)
    A = 1.0 / 4.0 - (k_F^2) / (4.0 * π * e^2) * diff_mu
    diff_rse =  diffv_cep(n)
    C = π / (2.0 * e^2 * k_F) * (-diff_rse)
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    x = rs^(1.0 / 2.0)
    B = (1.0 + a1 * x + a2 * x^3) / (3.0 + b1 * x + b2 * x^3)
    g = B / (A - C)
    if integral
        return -B .* r .* exp.(-sqrt(g) * k_F .* r)
    else
        return -B .* exp.(-sqrt(g) * k_F .* r) ./ r
    end
end

function prefact_delta_term_fxc(n)
    k_F = (3.0 * π^2 * n)^(1.0 / 3.0)
    e = 1.0
    diff_rse =  diffv_cep(n) 
    C = π / (2.0 * e^2 * k_F) * (-diff_rse)
    return - 4π * C / k_F^2 
end

function fxc_corradini(nbar, norm_rdiff )  
    delta_term = 0  # must added in the integral
    norm_rdiff_2 = copy(norm_rdiff)
    idx = findfirst(x -> x < 5 * 1e-15, norm_rdiff_2)
    norm_rdiff_2[idx] = 1000.0 
    res1= fxcr_residual_pz(nbar, norm_rdiff_2) 
    res1[idx] = 0. 
    res2= fxcr_smooth_pz(nbar, norm_rdiff) 
    tot_fxc = res1 +  res2
    return tot_fxc 
end
# ----------------------------------------------------------------------



function get_cot_density(ρ::Array{Float64}, basis::PlaneWaveBasis{T}) where {T} 
    ρ = abs.(ρ) .+ 1e-16 
    fft_size= basis.fft_size[1] 
    r_vectors_cart_= r_vectors_cart(basis) 
    nbar= mean(ρ)
    nbar_sing=nbar
	ncon = Vector{Float64}(undef, fft_size^3)  # [ (sum(fxc_corradini(ρ, basis, vp ) .* ρ) * basis.dvol) for vp in r_vectors(basis) ]
    list_of_idx = [(i, j, k) for i in 1:fft_size, j in 1:fft_size, k in 1:fft_size] 
    dvol = basis.dvol
    lattice = basis.model.lattice
    @threads for idx in list_of_idx 
        i,j,k = idx    
        norm_rdiff = get_norm_rdiff(r_vectors_cart_[i,j,k], [i-1,j-1,k-1], fft_size, lattice)
        big_nr = get_big_nr(ρ, [i-1,j-1,k-1], fft_size, lattice) 
        int_fxc_nr = sum(fxc_corradini(nbar, norm_rdiff) .* big_nr) * dvol 
        # term with delta function 
        delta_term = prefact_delta_term_fxc(nbar)* ρ[i,j,k]  
        # Treating the singularity 
        nabr_sing=nbar 
        R_step = dvol^(1.0 / 3.0)
        Rlist=collect(range(0.0, stop=R_step, step=R_step / 1000.))
        sing_analy=  sum(fxcr_residual_pz(nbar_sing,Rlist,true)) * R_step / 1000 * 4π * ρ[i,j,k] 
        # sum all terms 
        tot_int_fxc = int_fxc_nr + delta_term + sing_analy
        ncon_i = tot_int_fxc / get_Hexc_Hvxc_Hfxc(nbar)[3]
        #append!(ncon, ncon_i )
        ncon[k + (j-1) * fft_size + (i-1) * fft_size^2] = ncon_i 
    end 
    res= abs.( convert(Array{Float64}, reshape(ncon, basis.fft_size) ) ) .+ 1e-16
    #print( "ncon is not nan? ", !(any(isnan, res)), "mean ncon " , mean(res))  
    return res 
end

function meshgrid(x, y, z)
    return ([k for i in x, j in y, k in z],
            [j for i in x, j in y, k in z],
            [i for i in x, j in y, k in z])
end 


function get_norm_rdiff(r_local,r_idx, fft_size, lattice)
    # Initial parameters
    intg_scope = round(Int, fft_size)
    #locz,locy,locx=r_idx  
    locx,locy,locz=r_idx 
    srx, erx = locx - intg_scope, locx + intg_scope 
    sry, ery =  locy - intg_scope, locy + intg_scope 
    srz, erz = locz - intg_scope, locz + intg_scope 

    # Generate the ranges
    z_range = srz:erz
    y_range = sry:ery
    x_range = srx:erx

    big_rp = [ (lattice' * [x/fft_size y/fft_size z/fft_size]')'  for x in x_range, y in y_range, z in z_range]
    norm_rdiff   = [norm(vec(v) - vec(r_local)) for v in big_rp]
    norm_rdiff=reshape(norm_rdiff,  2*intg_scope + 1, 2*intg_scope +1, 2*intg_scope +1 ) 
    
    return norm_rdiff

end


function get_big_nr(ρ,r_idx, fft_size, lattice)
    # Initial parameters
    intg_scope = round(Int, fft_size)
    #locz,locy,locx=r_idx  
    locx,locy,locz=r_idx 
    srx, erx = locx - intg_scope, locx + intg_scope 
    sry, ery =  locy - intg_scope, locy + intg_scope 
    srz, erz = locz - intg_scope, locz + intg_scope 

    # Generate the ranges
    z_range = srz:erz
    y_range = sry:ery
    x_range = srx:erx

    # Generate `big_nR` array similar to list comprehension in Python
    big_nr= [ ρ[(mod(xi, fft_size) + 1) ,  (mod(yi, fft_size) +1 ) , (mod(zi, fft_size) + 1 )]
            for zi in z_range, yi in y_range, xi in x_range]

    reshape(big_nr,  2*intg_scope+1, 2*intg_scope+1, 2*intg_scope +1)  

end





@timing "ene_ops: connector" function ene_ops(term::TermConnector, basis::PlaneWaveBasis{T},
                                            ψ, occupation; ρ, kwargs...) where {T}
    cot_density= get_cot_density(total_density(ρ),basis)
    exc,pot_real, Hfxc = get_Hexc_Hvxc_Hfxc( cot_density )
    
    E= sum(total_density(ρ) .* exc ) * basis.dvol  

    ops = [RealSpaceMultiplication(basis, kpt, pot_real) for kpt in basis.kpoints]
    (; E, ops)
end
