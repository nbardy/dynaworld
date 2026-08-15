#include <metal_stdlib>
using namespace metal;

// Keep these flags bit-identical to finite_element_material_transfer.py.
constant int BRANCH_SMALL_TAU = 1;
constant int BRANCH_LOG_LINEAR_SERIES = 2;
constant int BRANCH_LOG_QUADRATIC_SERIES = 4;
constant int BRANCH_LOG_QUADRATIC_ERF = 8;
constant int BRANCH_LOG_QUADRATIC_TAIL = 16;
constant int BRANCH_INVALID = 1073741824;

struct MaterialEval {
    float tau;
    float beta;
    float3 m;
    float2 density_bounds;
    float3 dtau_controls;
    float dtau_length;
    float3 color_front_grad_weight;
    float3 color_back_grad_weight;
    float3 dm_dtau;
    int status;
};

inline float erf_approx(float x) {
    // Abramowitz-Stegun 7.1.26; max absolute error is about 1.5e-7.
    float sign_x = x < 0.0f ? -1.0f : 1.0f;
    float z = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * z);
    float polynomial =
        (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
           - 0.284496736f) * t + 0.254829592f) * t;
    return sign_x * (1.0f - polynomial * exp(-z * z));
}

inline float erfcx_approx_positive(float x) {
    // Numerical-Recipes-style relative erfc approximation with exp(-x^2)
    // analytically removed. The input is nonnegative.
    float t = 1.0f / (1.0f + 0.5f * x);
    float polynomial = t * (
        1.00002368f + t * (
        0.37409196f + t * (
        0.09678418f + t * (
       -0.18628806f + t * (
        0.27886807f + t * (
       -1.13520398f + t * (
        1.48851587f + t * (
       -0.82215223f + t * 0.17087277f
    ))))))));
    return t * exp(-1.26551223f + polynomial);
}

inline float one_minus_exp_neg(float tau, thread bool& used_series) {
    if (fabs(tau) < 1.0e-4f) {
        used_series = true;
        float t2 = tau * tau;
        return tau - 0.5f * t2 + (tau * t2) / 6.0f
            - (t2 * t2) / 24.0f + (tau * t2 * t2) / 120.0f;
    }
    return 1.0f - exp(-tau);
}

inline float log_linear_moment(
    int order,
    float b,
    float c,
    thread bool& used_series
) {
    if (fabs(b) < 0.75f) {
        used_series = true;
        float term = 1.0f;
        float result = 1.0f / float(order + 1);
        for (int k = 1; k < 20; ++k) {
            term *= -b / float(k);
            result += term / float(order + k + 1);
        }
        return exp(-c) * result;
    }
    float density0 = exp(-c);
    float density1 = exp(-(b + c));
    float result = (density0 - density1) / b;
    for (int n = 1; n <= order; ++n) {
        result = (float(n) * result - density1) / b;
    }
    return result;
}

inline void quadratic_moments(
    float a,
    float b,
    float c,
    thread float& i0,
    thread float& i1,
    thread float& i2,
    thread int& status
) {
    if (a < 0.02f) {
        float sums[3] = {0.0f, 0.0f, 0.0f};
        float factor = 1.0f;
        bool used_linear_series = false;
        for (int k = 0; k < 7; ++k) {
            for (int n = 0; n < 3; ++n) {
                bool this_series = false;
                sums[n] += factor * log_linear_moment(
                    n + 2 * k, b, c, this_series
                );
                used_linear_series = used_linear_series || this_series;
            }
            factor *= -a / float(k + 1);
        }
        i0 = sums[0];
        i1 = sums[1];
        i2 = sums[2];
        status |= BRANCH_LOG_QUADRATIC_SERIES;
        if (used_linear_series) {
            status |= BRANCH_LOG_LINEAR_SERIES;
        }
        return;
    }

    float sqrt_a = sqrt(a);
    float u0 = b / (2.0f * sqrt_a);
    float u1 = sqrt_a + u0;
    float exponent = -c + b * b / (4.0f * a);
    bool straddles_zero = u0 <= 0.0f && u1 >= 0.0f;
    // The approximation has bounded absolute error, so same-sign erf
    // subtraction can still lose relative accuracy even for modest
    // arguments.  Reserve it for sign-straddling intervals and route every
    // same-sign interval through the scaled-tail form.
    bool safe_erf = straddles_zero
        && exponent >= -80.0f && exponent <= 80.0f;
    if (safe_erf) {
        float prefactor = exp(exponent) * 0.886226925452758f / sqrt_a;
        i0 = prefactor * (erf_approx(u1) - erf_approx(u0));
        float f0 = exp(-c);
        float f1 = exp(-(a + b + c));
        i1 = (f0 - f1 - b * i0) / (2.0f * a);
        i2 = (i0 - b * i1 - f1) / (2.0f * a);
        if (isfinite(i0) && isfinite(i1) && isfinite(i2)
            && i0 >= 0.0f && i1 >= 0.0f && i2 >= 0.0f) {
            status |= BRANCH_LOG_QUADRATIC_ERF;
            return;
        }
    }

    float q0 = c;
    float q1 = a + b + c;
    float tail_prefactor = 0.886226925452758f / sqrt_a;
    if (u0 > 0.0f) {
        i0 = tail_prefactor * (
            exp(-q0) * erfcx_approx_positive(u0)
            - exp(-q1) * erfcx_approx_positive(u1)
        );
    } else if (u1 < 0.0f) {
        i0 = tail_prefactor * (
            exp(-q1) * erfcx_approx_positive(-u1)
            - exp(-q0) * erfcx_approx_positive(-u0)
        );
    } else {
        i0 = 0.0f;
        i1 = 0.0f;
        i2 = 0.0f;
        status |= BRANCH_INVALID;
        return;
    }
    float f0 = exp(-q0);
    float f1 = exp(-q1);
    i1 = (f0 - f1 - b * i0) / (2.0f * a);
    i2 = (i0 - b * i1 - f1) / (2.0f * a);
    if (!isfinite(i0) || !isfinite(i1) || !isfinite(i2)
        || i0 < 0.0f || i1 < 0.0f || i2 < 0.0f) {
        i0 = 0.0f;
        i1 = 0.0f;
        i2 = 0.0f;
        status |= BRANCH_INVALID;
        return;
    }
    status |= BRANCH_LOG_QUADRATIC_TAIL;
}

inline MaterialEval evaluate_material(
    int mode,
    float3 controls,
    float length,
    float3 color_front,
    float3 color_back
) {
    MaterialEval out;
    out.tau = 0.0f;
    out.beta = 1.0f;
    out.m = float3(0.0f);
    out.density_bounds = float2(0.0f);
    out.dtau_controls = float3(0.0f);
    out.dtau_length = 0.0f;
    out.color_front_grad_weight = float3(0.0f);
    out.color_back_grad_weight = float3(0.0f);
    out.dm_dtau = float3(0.0f);
    out.status = 0;

    bool invalid = !isfinite(length) || length < 0.0f
        || any(!isfinite(controls)) || any(!isfinite(color_front))
        || any(!isfinite(color_back)) || mode < 0 || mode > 5;
    if ((mode == 0 || mode == 1) && controls.x < 0.0f) invalid = true;
    if (mode == 2 && (controls.x < 0.0f || controls.y < 0.0f)) invalid = true;
    if (mode == 3 && any(controls < float3(0.0f))) invalid = true;
    if (mode == 5 && controls.x < 0.0f) invalid = true;
    if (invalid) {
        out.status = BRANCH_INVALID;
        return out;
    }

    if (mode == 0 || mode == 1) {
        out.dtau_length = controls.x;
        out.dtau_controls.x = length;
        out.density_bounds = float2(controls.x);
    } else if (mode == 2) {
        out.dtau_length = 0.5f * (controls.x + controls.y);
        out.dtau_controls = float3(0.5f * length, 0.5f * length, 0.0f);
        out.density_bounds = float2(
            min(controls.x, controls.y),
            max(controls.x, controls.y)
        );
    } else if (mode == 3) {
        out.dtau_length = (controls.x + controls.y + controls.z) / 3.0f;
        out.dtau_controls = float3(length / 3.0f);
        out.density_bounds = float2(
            min(controls.x, min(controls.y, controls.z)),
            max(controls.x, max(controls.y, controls.z))
        );
    } else if (mode == 4) {
        bool used0 = false;
        bool used1 = false;
        float i0 = log_linear_moment(
            0, controls.x, controls.y, used0
        );
        float i1 = log_linear_moment(
            1, controls.x, controls.y, used1
        );
        out.dtau_length = i0;
        out.dtau_controls = float3(-length * i1, -length * i0, 0.0f);
        float density0 = exp(-controls.y);
        float density1 = exp(-(controls.x + controls.y));
        out.density_bounds = float2(
            min(density0, density1),
            max(density0, density1)
        );
        if (used0 || used1) out.status |= BRANCH_LOG_LINEAR_SERIES;
    } else {
        float i0, i1, i2;
        quadratic_moments(
            controls.x, controls.y, controls.z, i0, i1, i2, out.status
        );
        out.dtau_length = i0;
        out.dtau_controls = -length * float3(i2, i1, i0);
        float q0 = controls.z;
        float q1 = controls.x + controls.y + controls.z;
        float stationary_x = controls.x > 0.0f
            ? clamp(-controls.y / (2.0f * controls.x), 0.0f, 1.0f)
            : 0.0f;
        float stationary_q =
            controls.x * stationary_x * stationary_x
            + controls.y * stationary_x + controls.z;
        float q_min = min(min(q0, q1), stationary_q);
        float q_max = max(q0, q1);
        out.density_bounds = float2(exp(-q_max), exp(-q_min));
    }

    out.tau = length * out.dtau_length;
    out.beta = exp(-out.tau);
    bool used_tau_series = false;
    float alpha = one_minus_exp_neg(out.tau, used_tau_series);
    if (used_tau_series) out.status |= BRANCH_SMALL_TAU;

    if (mode == 1) {
        float w1, dw1;
        if (fabs(out.tau) < 1.0e-4f) {
            float t = out.tau;
            float t2 = t * t;
            w1 = 0.5f * t - t2 / 3.0f + t * t2 / 8.0f
                - t2 * t2 / 30.0f + t * t2 * t2 / 144.0f;
            dw1 = 0.5f - 2.0f * t / 3.0f + 3.0f * t2 / 8.0f
                - 2.0f * t * t2 / 15.0f + 5.0f * t2 * t2 / 144.0f;
        } else {
            float numerator = 1.0f - (1.0f + out.tau) * out.beta;
            w1 = numerator / out.tau;
            dw1 = (out.tau * out.tau * out.beta - numerator)
                / (out.tau * out.tau);
        }
        float w0 = alpha - w1;
        float dw0 = out.beta - dw1;
        out.m = w0 * color_front + w1 * color_back;
        out.color_front_grad_weight = float3(w0);
        out.color_back_grad_weight = float3(w1);
        out.dm_dtau = dw0 * color_front + dw1 * color_back;
    } else {
        out.m = alpha * color_front;
        out.color_front_grad_weight = float3(alpha);
        out.dm_dtau = out.beta * color_front;
    }
    bool nonfinite_output =
        !isfinite(out.tau) || !isfinite(out.beta) || any(!isfinite(out.m))
        || any(!isfinite(out.density_bounds))
        || any(!isfinite(out.dtau_controls)) || !isfinite(out.dtau_length)
        || any(!isfinite(out.dm_dtau));
    if (nonfinite_output) {
        out.tau = 0.0f;
        out.beta = 1.0f;
        out.m = float3(0.0f);
        out.density_bounds = float2(0.0f);
        out.dtau_controls = float3(0.0f);
        out.dtau_length = 0.0f;
        out.color_front_grad_weight = float3(0.0f);
        out.color_back_grad_weight = float3(0.0f);
        out.dm_dtau = float3(0.0f);
        out.status |= BRANCH_INVALID;
    }
    return out;
}

kernel void worldfoam_material_forward(
    device float* out_tau,
    device float* out_beta,
    device float* out_m,
    device float* out_density_bounds,
    device int* out_status,
    device const float* controls,
    device const float* lengths,
    device const float* color_front,
    device const float* color_back,
    device const int* modes,
    constant uint& count,
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    MaterialEval value = evaluate_material(
        modes[index],
        float3(controls[3 * index], controls[3 * index + 1], controls[3 * index + 2]),
        lengths[index],
        float3(color_front[3 * index], color_front[3 * index + 1], color_front[3 * index + 2]),
        float3(color_back[3 * index], color_back[3 * index + 1], color_back[3 * index + 2])
    );
    out_tau[index] = value.tau;
    out_beta[index] = value.beta;
    out_m[3 * index] = value.m.x;
    out_m[3 * index + 1] = value.m.y;
    out_m[3 * index + 2] = value.m.z;
    out_density_bounds[2 * index] = value.density_bounds.x;
    out_density_bounds[2 * index + 1] = value.density_bounds.y;
    out_status[index] = value.status;
}

kernel void worldfoam_material_vjp(
    device float* out_grad_controls,
    device float* out_grad_color_front,
    device float* out_grad_color_back,
    device float* out_grad_length,
    device int* out_status,
    device const float* controls,
    device const float* lengths,
    device const float* color_front,
    device const float* color_back,
    device const int* modes,
    device const float* grad_tau,
    device const float* grad_beta,
    device const float* grad_m,
    constant uint& count,
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    MaterialEval value = evaluate_material(
        modes[index],
        float3(controls[3 * index], controls[3 * index + 1], controls[3 * index + 2]),
        lengths[index],
        float3(color_front[3 * index], color_front[3 * index + 1], color_front[3 * index + 2]),
        float3(color_back[3 * index], color_back[3 * index + 1], color_back[3 * index + 2])
    );
    float3 gm = float3(
        grad_m[3 * index], grad_m[3 * index + 1], grad_m[3 * index + 2]
    );
    float effective_tau = grad_tau[index] - value.beta * grad_beta[index]
        + dot(gm, value.dm_dtau);
    float3 grad_controls_value = effective_tau * value.dtau_controls;
    float3 grad_front = value.color_front_grad_weight * gm;
    float3 grad_back = value.color_back_grad_weight * gm;

    out_grad_controls[3 * index] = grad_controls_value.x;
    out_grad_controls[3 * index + 1] = grad_controls_value.y;
    out_grad_controls[3 * index + 2] = grad_controls_value.z;
    out_grad_color_front[3 * index] = grad_front.x;
    out_grad_color_front[3 * index + 1] = grad_front.y;
    out_grad_color_front[3 * index + 2] = grad_front.z;
    out_grad_color_back[3 * index] = grad_back.x;
    out_grad_color_back[3 * index + 1] = grad_back.y;
    out_grad_color_back[3 * index + 2] = grad_back.z;
    out_grad_length[index] = effective_tau * value.dtau_length;
    out_status[index] = value.status;
}
