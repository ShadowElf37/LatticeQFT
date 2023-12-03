#define DERIV_STEP 0.01

__kernel void metropolis_sweep(
    __global const float *deltas_g, __global const float *probs_g, __global float *phi, __global int *acceptance_rate_g)
{
    // initialize variables
    int gid = get_global_id(0); // each gid will get its own lattice to play with - these will be added up by the cpu
    float delta;
    int accept, phi_i;

    int phi_i_dt0, phi_i_dt1;
    int phi_i_dx0, phi_i_dx1;
    int phi_i_dy0, phi_i_dy1;

    float a0,a1,a2,a3, N, phi_i_val, dS;

    for (int t = 1; t < {{Nt-1}}; t++){
        for (int x = 1; x < {{Nx-1}}; x++){
            for (int y = 1; y < {{Ny-1}}; y++){

                // compute some indices for ease of use
                phi_i = gid*{{Nt*Nx*Ny}} + t*{{Nx*Ny}} + x*{{Ny}} + y;
                phi_i_dt0 = phi_i - {{Nx*Ny}};
                phi_i_dt1 = phi_i + {{Nx*Ny}};
                phi_i_dx0 = phi_i - {{Ny}};
                phi_i_dx1 = phi_i + {{Ny}};
                phi_i_dy0 = phi_i - 1;
                phi_i_dy1 = phi_i + 1;

                // compute some things to add up to make dS
                phi_i_val = phi[phi_i];
                a3 = {{lam}};
                a2 = {{4*lam}}*phi_i_val;
                a1 = {{1-2*lam}}+{{6*lam}}*phi_i_val*phi_i_val;
                N = {{-2*kappa*zeta}}*(phi[phi_i_dt1] - phi[phi_i_dt0])
                 + {{-2*kappa/zeta}}*(
                    (phi[phi_i_dx1] - phi[phi_i_dx0]) +
                    (phi[phi_i_dy1] - phi[phi_i_dy0])
                    );
                a0 = N+2*phi_i_val*({{1-2*lam}}+{{2*lam}}*phi_i_val*phi_i_val);

                // compute dS
                delta = deltas_g[phi_i];
                dS = delta*(a0+delta*(a1+delta*(a2+delta*a3)));
                // calculate accept bool
                accept = probs_g[phi_i] < exp(-dS);
                
                // replace value in phi
                phi[phi_i] += accept*delta;
                // record accept
                acceptance_rate_g[gid] += accept;
            }
        }
    }
}

__kernel void microcanonical_sweep_lambda0(
    __global const float *deltas_g, __global const float *probs_g, __global const float *mu, __global float *phi, __global int *acceptance_rate_g)
{
    // initialize variables
    int gid = get_global_id(0); // each gid will get its own phi to play with - these will be added up later
    //float delta;
    int accept, phi_i;

    int phi_i_dt0, phi_i_dt1;
    int phi_i_dx0, phi_i_dx1;
    int phi_i_dy0, phi_i_dy1;

    float a0, N, phi_i_val, dS;
    float dSdp, dSdp_proposed;

    // Metropolis sweep
    for (int t = 1; t < {{Nt-1}}; t++){
        for (int x = 1; x < {{Nx-1}}; x++){
            for (int y = 1; y < {{Ny-1}}; y++){

                // compute some indices for ease of use
                phi_i = gid*{{Nt*Nx*Ny}} + t*{{Nx*Ny}} + x*{{Ny}} + y;

                if (mu[phi_i] < 0.98) {

                    phi_i_dt0 = phi_i - {{Nx*Ny}};
                    phi_i_dt1 = phi_i + {{Nx*Ny}};
                    phi_i_dx0 = phi_i - {{Ny}};
                    phi_i_dx1 = phi_i + {{Ny}};
                    phi_i_dy0 = phi_i - 1;
                    phi_i_dy1 = phi_i + 1;

                    // compute some things to add up to make dS
                    phi_i_val = phi[phi_i];
                    N = {{-2*kappa*zeta}}*(phi[phi_i_dt1] + phi[phi_i_dt0])
                     + {{-2*kappa/zeta}}*(
                        (phi[phi_i_dx1] + phi[phi_i_dx0]) +
                        (phi[phi_i_dy1] + phi[phi_i_dy0])
                        );

                    dSdp = N+2*phi_i_val+DERIV_STEP;
                    phi_i_val = -0.5*(DERIV_STEP+N);
                    dSdp_proposed = N+2*phi_i_val+DERIV_STEP;

                    // calculate accept bool
                    accept = probs_g[phi_i] < fabs(dSdp/dSdp_proposed);
                    
                    // replace value in phi
                    if (accept) {
                        phi[phi_i] = phi_i_val;
                    }
                    acceptance_rate_g[gid] += accept;
                }
            }
        }
    }
}

__kernel void microcanonical_sweep(
    __global const float *deltas_g, __global const float *probs_g, __global const float *mu, __global float *phi, __global int *acceptance_rate_g)
{
    // initialize variables
    int gid = get_global_id(0); // each gid will get its own phi to play with - these will be added up later
    //float delta;
    int accept, phi_i;

    int phi_i_dt0, phi_i_dt1;
    int phi_i_dx0, phi_i_dx1;
    int phi_i_dy0, phi_i_dy1;

    float a0,a1,a2,a3, N, phi_i_val, dS;
    float dSdp, dSdp_proposed;

    // Metropolis sweep
    for (int t = 1; t < {{Nt-1}}; t++){
        for (int x = 1; x < {{Nx-1}}; x++){
            for (int y = 1; y < {{Ny-1}}; y++){

                // compute some indices for ease of use
                phi_i = gid*{{Nt*Nx*Ny}} + t*{{Nx*Ny}} + x*{{Ny}} + y;

                if (mu[phi_i] < 0.98) {

                    phi_i_dt0 = phi_i - {{Nx*Ny}};
                    phi_i_dt1 = phi_i + {{Nx*Ny}};
                    phi_i_dx0 = phi_i - {{Ny}};
                    phi_i_dx1 = phi_i + {{Ny}};
                    phi_i_dy0 = phi_i - 1;
                    phi_i_dy1 = phi_i + 1;

                    // compute some things to add up to make dS
                    phi_i_val = phi[phi_i];
                    a3 = {{lam}};
                    a2 = {{4*lam}}*phi_i_val;
                    a1 = {{1-2*lam}}+{{6*lam}}*phi_i_val*phi_i_val;
                    N = {{-2*kappa*zeta}}*(phi[phi_i_dt1] + phi[phi_i_dt0])
                     + {{-2*kappa/zeta}}*(
                        (phi[phi_i_dx1] + phi[phi_i_dx0]) +
                        (phi[phi_i_dy1] + phi[phi_i_dy0])
                        );
                    a0 = N+2*phi_i_val*({{1-2*lam}}+{{2*lam}}*phi_i_val*phi_i_val);
                    dSdp = (a0+DERIV_STEP*(a1+DERIV_STEP*(a2+DERIV_STEP*a3)));


                    // TODO select this new value randomly from the 3 solutions to the cubic
                    // phi_i_val = 
                    // a\left(c+\left(a+2x\right)\left(1+b\left(\left(a+2x\right)a+2\left(x^{2}-1\right)\right)\right)\right)=0


                    a2 = {{4*lam}}*phi_i_val;
                    a1 = {{1-2*lam}}+{{6*lam}}*phi_i_val*phi_i_val;
                    a0 = N+2*phi_i_val*({{1-2*lam}}+{{2*lam}}*phi_i_val*phi_i_val);
                    dSdp_proposed = (a0+DERIV_STEP*(a1+DERIV_STEP*(a2+DERIV_STEP*a3)));

                    // calculate accept bool
                    accept = probs_g[phi_i] < fabs(dSdp/dSdp_proposed);
                    
                    // replace value in phi
                    if (accept) {
                        phi[phi_i] = phi_i_val;
                    }
                    acceptance_rate_g[gid] += accept;
                }
            }
        }
    }
}