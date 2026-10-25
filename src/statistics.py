import warnings

import numpy as np
from scipy.integrate import quad


def _paired_differences(aw_vals, un_vals):
    aw, un = np.asarray(aw_vals, dtype=float), np.asarray(un_vals, dtype=float)
    if aw.ndim != 1 or un.shape != aw.shape or len(aw) == 0:
        raise ValueError('Paired samples must be non-empty, one-dimensional and equally sized')
    if not np.isfinite(aw).all() or not np.isfinite(un).all():
        raise ValueError('Paired samples must be finite; missing values cannot be replaced by zero')
    return un-aw


def permutation_test(aw_vals, un_vals, n_perm=10000):
    diffs = _paired_differences(aw_vals, un_vals)
    obs_diff = np.mean(diffs)
    n = len(diffs)
    if n <= 20:
        count = 0
        for start in range(0, 2**n, 4096):
            codes = np.arange(start, min(start+4096, 2**n), dtype=np.uint32)
            signs = 2*((codes[:, None] >> np.arange(n)) & 1).astype(np.int8)-1
            null_means = (signs @ diffs)/n
            count += np.count_nonzero(np.abs(null_means) >= abs(obs_diff)-1e-15)
        return float(count/2**n), float(obs_diff)

    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        if abs(np.mean(diffs * signs)) >= abs(obs_diff):
            count += 1
    return (count+1) / (n_perm+1), float(obs_diff)


def cohens_d(aw_vals, un_vals):
    return _paired_d_from_diffs(_paired_differences(aw_vals, un_vals))


def benjamini_hochberg(p_values, alpha=0.05):
    n = len(p_values)
    if n == 0:
        return [], []
    if not np.isfinite(p_values).all() or np.any(np.asarray(p_values) < 0) or np.any(np.asarray(p_values) > 1):
        raise ValueError('p-values must be finite and in [0, 1]')
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adj_p = np.empty(n)
    adj_p[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adj_p[sorted_idx[i]] = min(adj_p[sorted_idx[i + 1]], sorted_p[i] * n / (i + 1))
    return adj_p.tolist(), [ap < alpha for ap in adj_p]


def bootstrap_ci_cohens_d(aw_vals, un_vals, n_boot=5000, ci=0.95):
    diffs = _paired_differences(aw_vals, un_vals)
    n = len(diffs)
    rng = np.random.default_rng(42)
    if n < 2 or n_boot < 1 or not 0 < ci < 1:
        raise ValueError('Bootstrap requires >=2 pairs, positive draws and 0 < ci < 1')
    bd = diffs[rng.integers(0, n, size=(n_boot, n))]
    std = np.std(bd, axis=1, ddof=1)
    valid = std > 0
    if not valid.all():
        warnings.warn(f'{np.count_nonzero(~valid)}/{n_boot} bootstrap samples have undefined effect size', stacklevel=2)
    boot_d = bd.mean(axis=1)[valid]/std[valid]
    if not len(boot_d):
        return np.nan, np.nan
    lo = np.percentile(boot_d, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_d, (1 + ci) / 2 * 100)
    return lo, hi


def _paired_d_from_diffs(diffs):
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2:
        return np.nan
    std = np.std(diffs, ddof=1)
    return float(np.mean(diffs)/std) if std > 0 else np.nan


def compute_group_diagnostics(all_tests, datasets):
    """Summarize sensitivity to condition and monkey grouping."""
    dataset_names = list(datasets.keys())
    conditions = []
    monkeys = []
    for ds_name in dataset_names:
        cond = datasets[ds_name]['condition']
        monkey = datasets[ds_name]['monkey']
        if cond not in conditions:
            conditions.append(cond)
        if monkey not in monkeys:
            monkeys.append(monkey)

    diagnostics = {}
    for name, band, aw, un, d, p in all_tests:
        if len(aw) != len(dataset_names) or len(un) != len(dataset_names):
            raise ValueError('Grouping requires complete pairs in the declared dataset order')
        aw = np.asarray(aw, dtype=float)
        un = np.asarray(un, dtype=float)
        diffs = un - aw

        cond_diffs = []
        for cond in conditions:
            idx = [i for i, ds_name in enumerate(dataset_names)
                   if datasets[ds_name]['condition'] == cond]
            cond_diffs.append(float(np.mean(diffs[idx])))
        cond_p, _ = permutation_test(np.zeros(len(cond_diffs)), cond_diffs)

        monkey_diffs = []
        for monkey in monkeys:
            idx = [i for i, ds_name in enumerate(dataset_names)
                   if datasets[ds_name]['monkey'] == monkey]
            monkey_diffs.append(float(np.mean(diffs[idx])))

        loo_condition_d = {}
        for cond in conditions:
            idx = [i for i, ds_name in enumerate(dataset_names)
                   if datasets[ds_name]['condition'] != cond]
            loo_condition_d[cond] = _paired_d_from_diffs(diffs[idx])

        loo_monkey_d = {}
        for monkey in monkeys:
            idx = [i for i, ds_name in enumerate(dataset_names)
                   if datasets[ds_name]['monkey'] != monkey]
            loo_monkey_d[monkey] = _paired_d_from_diffs(diffs[idx])

        diagnostics[(name, band)] = {
            'dataset_diffs': diffs,
            'n_positive': int(np.sum(diffs > 0)),
            'n_negative': int(np.sum(diffs < 0)),
            'condition_diffs': dict(zip(conditions, cond_diffs)),
            'condition_d': _paired_d_from_diffs(cond_diffs),
            'condition_p': cond_p,
            'monkey_diffs': dict(zip(monkeys, monkey_diffs)),
            'monkey_mean_signflip_p': permutation_test(np.zeros(len(monkey_diffs)), monkey_diffs)[0],
            'loo_condition_d': loo_condition_d,
            'loo_monkey_d': loo_monkey_d,
        }
    return diagnostics


def bayes_factor_paired(aw_vals, un_vals, r=0.707):
    """Two-sided JZS BF10, Rouder et al. (2009), eq. 1.

    Matches the formula used by Pingouin/JASP/BayesFactor. The auxiliary
    inverse-gamma mixing density is g**(-3/2) exp(-1/(2g))/sqrt(2*pi).
    """
    diffs = _paired_differences(aw_vals, un_vals)
    n = len(diffs)
    if n < 2 or r <= 0 or not np.isfinite(r):
        raise ValueError('JZS requires >=2 pairs and a finite positive prior scale')
    std = np.std(diffs, ddof=1)
    if std == 0:
        return np.nan
    t_stat = np.mean(diffs) / (std / np.sqrt(n))
    df = n - 1

    def integrand(g):
        scale = 1+n*g*r*r
        # Integrate the likelihood ratio directly to avoid tiny H0 densities.
        log_value = (-.5*np.log(scale)
                     -(df+1)/2*np.log1p(t_stat*t_stat/(df*scale))
                     +(df+1)/2*np.log1p(t_stat*t_stat/df)
                     -.5*np.log(2*np.pi)-1.5*np.log(g)-1/(2*g))
        return np.exp(log_value)

    return float(quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-9, limit=200)[0])


def collect_tests(kuramoto_results, wpli_results, fcd_results, coal_results, datasets, analysis_bands):
    """Fail on incomplete/nonfinite pairs; never fabricate measurements or shift IDs."""
    if len(datasets) < 2:
        raise ValueError('At least two complete dataset pairs are required')
    specifications = []
    for band in analysis_bands:
        specifications.extend([('Kur_Sync', band, kuramoto_results, (band, 'sync')),
                               ('Kur_Meta', band, kuramoto_results, (band, 'meta'))])
    specifications.extend(('wPLI', band, wpli_results, (band,)) for band in analysis_bands)
    specifications.extend([('FCD_Var', 'delta', fcd_results, ()),
                           ('H_norm', 'delta', coal_results, ())])
    tests = []
    for metric, band, source, path in specifications:
        awake, unconscious = [], []
        for dataset in datasets:
            for state, target in [('Awake', awake), ('Unconscious', unconscious)]:
                try:
                    value = source[dataset][state]
                    for key in path:
                        value = value[key]
                    value = float(value)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f'Missing {dataset}/{state}/{metric}/{band}') from exc
                if not np.isfinite(value):
                    raise ValueError(f'Nonfinite {dataset}/{state}/{metric}/{band}')
                target.append(value)
        p, _ = permutation_test(awake, unconscious)
        tests.append((metric, band, awake, unconscious, cohens_d(awake, unconscious), p))
    adjusted, significant = benjamini_hochberg([test[5] for test in tests])
    factors = [bayes_factor_paired(test[2], test[3]) for test in tests]
    return tests, adjusted, significant, factors


def statistical_rows(kuramoto, wpli, fcd, coalition, datasets, bands):
    """Full-precision records for JSON/CSV and plotting."""
    tests, adjusted, significant, factors = collect_tests(kuramoto, wpli, fcd, coalition, datasets, bands)
    diagnostics = compute_group_diagnostics(tests, datasets)
    rows = []
    for index, (metric, band, aw, un, effect, p) in enumerate(tests):
        low, high = bootstrap_ci_cohens_d(aw, un)
        rows.append({'metric': metric, 'band': band, 'dataset_ids': list(datasets),
                     'n_pairs': len(aw), 'awake': aw, 'unconscious': un,
                     'mean_difference': float(np.mean(np.asarray(un)-aw)),
                     'd': effect, 'ci_lo': float(low), 'ci_hi': float(high),
                     'p_raw': p, 'p_fdr': adjusted[index], 'fdr_significant': bool(significant[index]),
                     'bf': factors[index], 'diagnostics': diagnostics[(metric, band)]})
    return rows


def print_stats_table(all_tests, p_fdr, sig_fdr, bf_cache):
    """Print full results table."""
    print(f'{"Metric":<12s} {"Band":<11s} | {"d":>7s} {"95% CI":>16s} | '
          f'{"p_raw":>7s} {"p_FDR":>7s} {"FDR":>4s} | {"BF10":>7s} {"Evidence":>12s}')
    print('-' * 100)
    for i, (name, band, aw, un, d, p) in enumerate(all_tests):
        ci_lo, ci_hi = bootstrap_ci_cohens_d(aw, un)
        bf = bf_cache[i]
        fdr_sig = '*' if sig_fdr[i] else ''
        if not np.isfinite(bf): ev = 'Undefined'
        elif bf > 10: ev = 'Strong H1'
        elif bf > 3: ev = 'Moderate H1'
        elif bf > 1: ev = 'Anecdotal H1'
        elif bf > 1/3: ev = 'Anecdotal H0'
        elif bf > 1/10: ev = 'Moderate H0'
        else: ev = 'Strong H0'
        raw_sig = '*' if p < 0.05 else '(+)' if p < 0.1 else ''
        print(f'{name:<12s} {band:<11s} | {d:>+7.3f} [{ci_lo:>+6.2f},{ci_hi:>+6.2f}] | '
              f'{p:>7.4f}{raw_sig:<3s} {p_fdr[i]:>7.4f} {fdr_sig:>4s} | {bf:>7.3f} {ev:>12s}')

    p_orig = [t[5] for t in all_tests]
    n_raw_sig = sum(1 for p in p_orig if p < 0.05)
    n_fdr_sig = sum(sig_fdr)
    n_bf_mod = sum(1 for bf in bf_cache if bf > 3)
    print(f'\nSummary: {len(all_tests)} tests | Raw p<0.05: {n_raw_sig} | '
          f'FDR p<0.05: {n_fdr_sig} | BF>3: {n_bf_mod}')


def print_group_diagnostics(all_tests, datasets, bf_cache, keys=None):
    """Print a compact grouping-aware robustness table for key effects."""
    diagnostics = compute_group_diagnostics(all_tests, datasets)
    if keys is None:
        keys = [(name, band) for (name, band, *_), bf in zip(all_tests, bf_cache) if bf > 3]

    if not keys:
        return diagnostics

    print('\nGrouping-aware diagnostics (exploratory; conditions and animals are not independent dataset replicates)')
    print(f'{"Metric":<12s} {"Band":<11s} | {"signs":>7s} | {"cond d":>8s} {"cond p":>8s} | '
          f'{"monkey mean diffs":>24s} | {"leave-one-condition d":>28s}')
    print('-' * 112)
    for key in keys:
        if key not in diagnostics:
            continue
        diag = diagnostics[key]
        signs = f"{diag['n_positive']}+/{diag['n_negative']}-"
        monkey_str = ', '.join(f'{k}:{v:+.3g}' for k, v in diag['monkey_diffs'].items())
        loo_str = ', '.join(f'{k}:{v:+.2f}' for k, v in diag['loo_condition_d'].items())
        print(f'{key[0]:<12s} {key[1]:<11s} | {signs:>7s} | '
              f"{diag['condition_d']:>+8.3f} {diag['condition_p']:>8.4f} | "
              f'{monkey_str:>24s} | {loo_str:>28s}')

    return diagnostics
