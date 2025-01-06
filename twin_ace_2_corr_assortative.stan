data {
  // Meta-data
  int<lower=0> mz_N;// Number of mz-pairs total
  int<lower=0> dz_N;
  int<lower=0> cohort_N; // Number of distinct birth years
  int<lower = 1> outcomes_N;//Number of observed outcome types (ordinal categorical - written for education)
  int<lower = 0> factor_levels_N[2];//Number of levels within the hierarchical variable
  int factor_levels[cohort_N, 2];
  int sub_factor_levels_N[factor_levels_N[1]]; // Number of levels within each level of the top factor

  //Individual observations
  int<lower = 1, upper = outcomes_N> y_mz_1[mz_N]; // Sorted by twin-pair id
  int<lower = 1, upper = outcomes_N> y_mz_2[mz_N]; 
  int<lower = 1, upper = cohort_N> cohort_mz[mz_N]; 
  
  int<lower = 1, upper = outcomes_N> y_dz_1[dz_N]; 
  int<lower = 1, upper = outcomes_N> y_dz_2[dz_N]; 
  int<lower = 1, upper = cohort_N> cohort_dz[dz_N]; 
}
transformed data {
  int cutpoints_N = outcomes_N - 1;
}
parameters {
  // For cutpoints
  simplex[outcomes_N] outcome_shares[cohort_N];
  
  real mz_shared_mu; // shared variance of mz
  real dz_shared_mu; // shared variance of dz
  
  // Level 2
  real<lower = 0> mz_shared_sigma;
  real<lower = 0> dz_shared_sigma;
  
  // Level 3
  real<lower = 0> mz_shared_2_sigma;
  real<lower = 0> dz_shared_2_sigma;
  

  // Shared within twin-pair (random effect - unscaled)
  vector[mz_N + dz_N] twinpair_std;
  vector[factor_levels_N[1]] mz_shared_1_std;
  vector[factor_levels_N[1]] dz_shared_1_std;
  vector[factor_levels_N[2]] mz_shared_2_std;
  vector[factor_levels_N[2]] dz_shared_2_std;

}
transformed parameters {
  matrix[cutpoints_N, cohort_N] cohort_cutpoints;
  vector[cohort_N] MZ_shared;
  vector[cohort_N] DZ_shared;
  vector[cohort_N] MZ_coef;
  vector[cohort_N] DZ_coef;
  vector[cohort_N] mz_re;
  vector[cohort_N] dz_re;
  vector[cohort_N] MZ_sigma; // Unexplained variance
  vector[cohort_N] DZ_sigma;
  
  
  {
    vector[factor_levels_N[1]] factor_1_mz_re;
    vector[factor_levels_N[2]] factor_2_mz_re;
    vector[factor_levels_N[1]] factor_1_dz_re;
    vector[factor_levels_N[2]] factor_2_dz_re;
    int temp_passed = 0;

    factor_1_mz_re =  mz_shared_sigma * (mz_shared_1_std - mean(mz_shared_1_std));
    factor_1_dz_re =  dz_shared_sigma * (dz_shared_1_std - mean(dz_shared_1_std));

    //Separate sum-to-zero constraints for factor 2 within each level of factor 1
    for (level_i in 1:factor_levels_N[1]){
      factor_2_mz_re[(temp_passed + 1):(temp_passed + sub_factor_levels_N[level_i])] =
        mz_shared_2_sigma * (mz_shared_2_std[(temp_passed + 1):(temp_passed + sub_factor_levels_N[level_i])] -
                      mean(mz_shared_2_std[(temp_passed + 1):(temp_passed + sub_factor_levels_N[level_i])]));
      factor_2_dz_re[(temp_passed + 1):(temp_passed + sub_factor_levels_N[level_i])] =
        dz_shared_2_sigma * (dz_shared_2_std[(temp_passed + 1):(temp_passed + sub_factor_levels_N[level_i])] -
                      mean(dz_shared_2_std[(temp_passed + 1):(temp_passed + sub_factor_levels_N[level_i])]));
      temp_passed += sub_factor_levels_N[level_i];
    }
    for (cohort_i in 1:cohort_N){
        mz_re[cohort_i] = mz_shared_mu + factor_1_mz_re[factor_levels[cohort_i, 1]] +
            factor_2_mz_re[factor_levels[cohort_i, 2]];
        dz_re[cohort_i] = dz_shared_mu + factor_1_dz_re[factor_levels[cohort_i, 1]] +
            factor_2_dz_re[factor_levels[cohort_i, 2]];
    }

  }

  
  for (cohort_index in 1:cohort_N){

    cohort_cutpoints[, cohort_index] = inv_Phi(cumulative_sum(outcome_shares[cohort_index][1:cutpoints_N]));


  }
    MZ_shared = inv_logit(mz_re);
    DZ_shared = inv_logit(dz_re);
  
    MZ_coef = sqrt(MZ_shared);
    DZ_coef = sqrt(DZ_shared);
    MZ_sigma = sqrt(1 - MZ_shared);
    DZ_sigma = sqrt(1 - DZ_shared);

}
model {
  real temp_sum = 0;
  real upper_limit = 0;
  real lower_limit = 0;
  int twinpair_counter = 1;
  int twin_same = 0;
  
  // Priors

  for (cohort_index in 1:cohort_N){
    outcome_shares[cohort_index] ~ dirichlet(rep_vector(1,outcomes_N));
  }
  
  mz_shared_mu ~ normal(0, 1.5); 
  dz_shared_mu ~ normal(0, 1.5);
  mz_shared_sigma ~ normal(0, 0.5);
  dz_shared_sigma ~ normal(0, 0.5);
  mz_shared_2_sigma ~ normal(0, 0.25);
  dz_shared_2_sigma ~ normal(0, 0.25);
  
  
  mz_shared_1_std ~ std_normal();
  dz_shared_1_std ~ std_normal();
  mz_shared_2_std ~ std_normal();
  dz_shared_2_std ~ std_normal();
  twinpair_std ~ std_normal();
  
  // Model
  
  // MZ twins
  
  for (mz_index in 1:mz_N){
    twin_same = (y_mz_1[mz_index] == y_mz_2[mz_index])? 1 : 0; //If same outcome for both twins
    
    upper_limit = (y_mz_1[mz_index] == outcomes_N) ? 1 : 
                    exp(normal_lcdf(cohort_cutpoints[y_mz_1[mz_index], cohort_mz[mz_index]] | 
                                      MZ_coef[cohort_mz[mz_index]] * twinpair_std[mz_index],
                                      MZ_sigma[cohort_mz[mz_index]]));
    
    lower_limit = (y_mz_1[mz_index] == 1) ? 0 : 
                    exp(normal_lcdf(cohort_cutpoints[y_mz_1[mz_index] - 1, cohort_mz[mz_index]] | 
                                      MZ_coef[cohort_mz[mz_index]] * twinpair_std[mz_index],
                                      MZ_sigma[cohort_mz[mz_index]]));
    
    temp_sum += log(upper_limit - lower_limit) * ((twin_same == 1) ? 2 : 1);
    
    if (twin_same == 0){
      upper_limit = (y_mz_2[mz_index] == outcomes_N) ? 1 : 
                      exp(normal_lcdf(cohort_cutpoints[y_mz_2[mz_index], cohort_mz[mz_index]] | 
                                        MZ_coef[cohort_mz[mz_index]] * twinpair_std[mz_index],
                                        MZ_sigma[cohort_mz[mz_index]]));
      
      lower_limit = (y_mz_2[mz_index] == 1) ? 0 : 
                      exp(normal_lcdf(cohort_cutpoints[y_mz_2[mz_index] - 1, cohort_mz[mz_index]] | 
                                        MZ_coef[cohort_mz[mz_index]] * twinpair_std[mz_index],
                                        MZ_sigma[cohort_mz[mz_index]]));
      
      temp_sum += log(upper_limit - lower_limit);
    }
    
  }

  for (dz_index in 1:dz_N){
    twin_same = (y_dz_1[dz_index] == y_dz_2[dz_index])? 1 : 0; //If same outcome for both twins
    
    upper_limit = (y_dz_1[dz_index] == outcomes_N) ? 1 : 
                    exp(normal_lcdf(cohort_cutpoints[y_dz_1[dz_index], cohort_dz[dz_index]] | 
                                      DZ_coef[cohort_dz[dz_index]] * twinpair_std[dz_index + mz_N],
                                      DZ_sigma[cohort_dz[dz_index]]));
    
    lower_limit = (y_dz_1[dz_index] == 1) ? 0 : 
                    exp(normal_lcdf(cohort_cutpoints[y_dz_1[dz_index] - 1, cohort_dz[dz_index]] | 
                                      DZ_coef[cohort_dz[dz_index]] * twinpair_std[dz_index + mz_N],
                                      DZ_sigma[cohort_dz[dz_index]]));
    
    temp_sum += log(upper_limit - lower_limit) * ((twin_same == 1) ? 2 : 1);
    
    if (twin_same == 0){
      upper_limit = (y_dz_2[dz_index] == outcomes_N) ? 1 : 
                      exp(normal_lcdf(cohort_cutpoints[y_dz_2[dz_index], cohort_dz[dz_index]] | 
                                        DZ_coef[cohort_dz[dz_index]] * twinpair_std[dz_index + mz_N],
                                        DZ_sigma[cohort_dz[dz_index]]));
      
      lower_limit = (y_dz_2[dz_index] == 1) ? 0 : 
                      exp(normal_lcdf(cohort_cutpoints[y_dz_2[dz_index] - 1, cohort_dz[dz_index]] | 
                                        DZ_coef[cohort_dz[dz_index]] * twinpair_std[dz_index + mz_N],
                                        DZ_sigma[cohort_dz[dz_index]]));
      
      temp_sum += log(upper_limit - lower_limit);
    }
    
  }
  target += temp_sum;
  
  
}
generated quantities {
  int synth_y_mz_1[mz_N];
  int synth_y_mz_2[mz_N];
  int synth_y_dz_1[dz_N];
  int synth_y_dz_2[dz_N];
  vector[cohort_N] A_share;
  vector[cohort_N] C_share;
  vector[cohort_N] E_share;
  vector[cohort_N] A_share_assortative;
  vector[cohort_N] C_share_assortative;
  vector[cohort_N] E_share_assortative;
  
  for (cohort_i in 1:cohort_N){
    A_share[cohort_i] = 2 * (MZ_shared[cohort_i] - DZ_shared[cohort_i]);
    C_share[cohort_i] =  MZ_shared[cohort_i] - A_share[cohort_i];
    E_share[cohort_i] = 1 - MZ_shared[cohort_i];
  
    // With an assumption that DZ twins share 0.68 of their genes
    A_share_assortative[cohort_i] = (1/(1-0.68)) * (MZ_shared[cohort_i] - DZ_shared[cohort_i]);
    C_share_assortative[cohort_i] =  MZ_shared[cohort_i] - A_share_assortative[cohort_i];
    E_share_assortative[cohort_i] = 1 - MZ_shared[cohort_i];
    
  
  }


  {
    real temp_twin;
    real latent_pair;
    int temp_twin_state;

    for(i in 1:mz_N){
      temp_twin_state = 1;

      latent_pair = normal_rng(0,1);
      temp_twin = normal_rng(MZ_coef[cohort_mz[cohort_mz[i]]] * latent_pair, MZ_sigma[cohort_mz[i]]);

      while (temp_twin_state < outcomes_N && temp_twin > cohort_cutpoints[temp_twin_state, cohort_mz[i]]){
        temp_twin_state += 1;
      }

      synth_y_mz_1[i] = temp_twin_state;

      temp_twin_state = 1;

      temp_twin = normal_rng(MZ_coef[cohort_mz[cohort_mz[i]]] * latent_pair, MZ_sigma[cohort_mz[i]]);

      while (temp_twin_state < outcomes_N && temp_twin > cohort_cutpoints[temp_twin_state, cohort_mz[i]]){
        temp_twin_state += 1;
      }

      synth_y_mz_2[i] = temp_twin_state;

    }

    for(i in 1:dz_N){
      temp_twin_state = 1;

      latent_pair = normal_rng(0,1);
      temp_twin = normal_rng(DZ_coef[cohort_mz[cohort_dz[i]]] * latent_pair, DZ_sigma[cohort_dz[i]]);

      while (temp_twin_state < outcomes_N && temp_twin > cohort_cutpoints[temp_twin_state, cohort_dz[i]]){
        temp_twin_state += 1;
      }

      synth_y_dz_1[i] = temp_twin_state;

      temp_twin_state = 1;

      temp_twin = normal_rng(DZ_coef[cohort_dz[cohort_dz[i]]] * latent_pair, DZ_sigma[cohort_dz[i]]);

      while (temp_twin_state < outcomes_N && temp_twin > cohort_cutpoints[temp_twin_state, cohort_dz[i]]){
        temp_twin_state += 1;
      }

      synth_y_dz_2[i] = temp_twin_state;

    }
  }

}

