import sympy 
import symnum.numpy as snp


def ips_fhn_euler_maruyama_mean_and_covariance(
        drift_func_rough, diff_coeff_rough
):
    def mean_and_covariance(x, θ, t, empirical_mean):   
        dim_r = 1
        x_r= x[:dim_r]

        μ = snp.array(
            [
                x_r + drift_func_rough(x, θ, empirical_mean) * t
            ]
        )
        B_r = diff_coeff_rough(x, θ)

        Σ = B_r @ B_r.T * t

        return μ, Σ

    return mean_and_covariance


def ips_fhn_local_gaussian_mean_and_covariance(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
):
    def mean_and_covariance(x, θ, t, empirical_mean):   
        dim_r = 1
        x_r, x_s = x[:dim_r], x[dim_r:]
        a, b, τ, σ, κ = θ

        second_order_term = (drift_func_rough(x, θ, empirical_mean)/τ - b * drift_func_smooth(x, θ)/(τ**2)) 

        μ = snp.concatenate(
            [
                x_r + drift_func_rough(x, θ, empirical_mean) * t,
                x_s + drift_func_smooth(x, θ) * t + second_order_term*t**2/2,
            ]
        )
    
        B_r = diff_coeff_rough(x, θ)
        C_s = snp.array([[σ/τ]])

        Σ_11 = B_r @ B_r.T * t
        Σ_12 = B_r @ C_s.T * t**2 / 2
        Σ_22 = C_s @ C_s.T * t**3 / 3

        Σ = snp.concatenate(
            [
                snp.concatenate([Σ_11, Σ_12], axis=1),
                snp.concatenate([Σ_12, Σ_22], axis=1),
            ],
            axis=0,
        )
        return μ, Σ

    return mean_and_covariance

def euler_maruyama_log_transition_density_rough(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):

    mean_and_covariance = ips_fhn_euler_maruyama_mean_and_covariance(
        drift_func_rough, diff_coeff_rough
    )

    def log_transition_density(x_t, x_0, θ, t, empirical_mean):
        dim_x_r = 1
        x_t_r = x_t[:dim_x_r]
        μ, Σ = mean_and_covariance(x_0, θ, t, empirical_mean)
        Σ = sympy.Matrix(Σ)
        chol_Σ = Σ.cholesky(hermitian=False)
        x_t_r_minus_μ = sympy.Matrix(x_t_r - μ)
        return -(
            (
                x_t_r_minus_μ.T
                * chol_Σ.T.upper_triangular_solve(
                    chol_Σ.lower_triangular_solve(x_t_r_minus_μ)
                )
            )[0, 0]
            / 2
            + snp.log(chol_Σ.diagonal()).sum()
            + snp.log(2 * snp.pi) * (dim_x_r / 2)
        )

    return log_transition_density 

def local_gaussian_log_transition_density(
    drift_func_rough, drift_func_smooth, diff_coeff_rough
):

    mean_and_covariance = ips_fhn_local_gaussian_mean_and_covariance(
        drift_func_rough, drift_func_smooth, diff_coeff_rough
    )

    def log_transition_density(x_t, x_0, θ, t, empirical_mean):
        dim_x = x_0.shape[0]
        μ, Σ = mean_and_covariance(x_0, θ, t, empirical_mean)
        Σ = sympy.Matrix(Σ)
        chol_Σ = Σ.cholesky(hermitian=False)
        x_t_minus_μ = sympy.Matrix(x_t - μ)
        return -(
            (
                x_t_minus_μ.T
                * chol_Σ.T.upper_triangular_solve(
                    chol_Σ.lower_triangular_solve(x_t_minus_μ)
                )
            )[0, 0]
            / 2
            + snp.log(chol_Σ.diagonal()).sum()
            + snp.log(2 * snp.pi) * (dim_x / 2)
        )

    return log_transition_density